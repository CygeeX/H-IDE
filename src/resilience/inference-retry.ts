const MAX_INFERENCE_ATTEMPTS = 3;
const RETRY_DELAYS_MS = [250, 750] as const;

const RETRYABLE_CODES = new Set([
  "ECONNABORTED",
  "ECONNREFUSED",
  "ECONNRESET",
  "EHOSTUNREACH",
  "ENETDOWN",
  "ENETUNREACH",
  "EPIPE",
  "ETIMEDOUT",
  "EAI_AGAIN",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_SOCKET",
]);

function errorChain(error: unknown): unknown[] {
  const values: unknown[] = [];
  const seen = new Set<unknown>();
  let current: unknown = error;
  while (current != null && !seen.has(current) && values.length < 8) {
    seen.add(current);
    values.push(current);
    current = typeof current === "object" && "cause" in current
      ? (current as { cause?: unknown }).cause
      : undefined;
  }
  return values;
}

function numericStatus(value: unknown): number | undefined {
  if (typeof value !== "object" || value == null) return undefined;
  for (const candidate of [
    (value as { status?: unknown }).status,
    (value as { statusCode?: unknown }).statusCode,
    (value as { response?: { status?: unknown } }).response?.status,
  ]) {
    if (typeof candidate === "number" && Number.isInteger(candidate)) return candidate;
  }
  return undefined;
}

export function isRetryableInferenceError(error: unknown): boolean {
  for (const value of errorChain(error)) {
    const status = numericStatus(value);
    if (status === 429 || (status != null && status >= 500 && status <= 599)) return true;
    if (typeof value === "object" && value != null) {
      const code = (value as { code?: unknown }).code;
      if (typeof code === "string" && RETRYABLE_CODES.has(code.toUpperCase())) return true;
    }
    const message = value instanceof Error ? value.message : String(value);
    if (
      /cannot connect to api|client network socket disconnected|connecterror|connection (?:reset|refused)|fetch failed|network error|socket hang up|timed?\s*out|tls connection|too many requests|rate limit|bad gateway|service unavailable|gateway timeout/i.test(
        message,
      )
    ) return true;
  }
  return false;
}

function waitForRetry(delayMs: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason ?? new DOMException("aborted", "AbortError"));
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason ?? new DOMException("aborted", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/** Retry planning-only model I/O, never validation or robot-side operations. */
export async function withInferenceRetry<T>(
  operation: () => Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  for (let attempt = 1; attempt <= MAX_INFERENCE_ATTEMPTS; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      if (
        signal.aborted
        || attempt === MAX_INFERENCE_ATTEMPTS
        || !isRetryableInferenceError(error)
      ) throw error;
      await waitForRetry(RETRY_DELAYS_MS[attempt - 1]!, signal);
    }
  }
  throw new Error("unreachable inference retry state");
}
