export type StreamingSpeechPhase =
  | "idle"
  | "requesting_permission"
  | "connecting"
  | "listening"
  | "finalizing"
  | "closed"
  | "error";

export interface StreamingSpeechMetrics {
  captureStartedAt?: number;
  firstPartialMs?: number;
  endToFinalMs?: number;
  framesSent: number;
  audioMsSent: number;
  ttsFirstByteMs?: number;
}

export interface StreamingTranscriptEvent {
  text: string;
  confirmedText?: string;
  utteranceId?: string;
}

export interface StreamingSpeechCallbacks {
  onPhase?: (phase: StreamingSpeechPhase) => void;
  onPartial?: (event: StreamingTranscriptEvent) => void;
  onFinal?: (event: StreamingTranscriptEvent) => void;
  onMetrics?: (metrics: StreamingSpeechMetrics) => void;
  onError?: (message: string) => void;
}

interface SpeechServerEvent {
  type?: string;
  text?: string;
  confirmedText?: string;
  utteranceId?: string;
  message?: string;
  fatal?: boolean;
}

const TARGET_SAMPLE_RATE = 16_000;
const FRAME_DURATION_MS = 40;
const FRAME_SAMPLES = TARGET_SAMPLE_RATE * FRAME_DURATION_MS / 1000;

/**
 * Stateful resampling is important here: resampling every 128-sample
 * AudioWorklet callback independently accumulates rounding gaps. This framer
 * preserves the fractional source position between callbacks and emits only
 * contract-sized 40 ms PCM16 little-endian frames.
 */
export class StreamingPcm16Framer {
  private readonly sourceRate: number;
  private readonly step: number;
  private source = new Float32Array(0);
  private sourcePosition = 0;
  private targetSamples: number[] = [];

  constructor(sourceRate: number) {
    if (!Number.isFinite(sourceRate) || sourceRate <= 0) throw new Error("音频采样率无效");
    this.sourceRate = sourceRate;
    this.step = sourceRate / TARGET_SAMPLE_RATE;
  }

  push(chunk: Float32Array): ArrayBuffer[] {
    if (!chunk.length) return [];
    const joined = new Float32Array(this.source.length + chunk.length);
    joined.set(this.source);
    joined.set(chunk, this.source.length);
    this.source = joined;

    if (this.sourceRate === TARGET_SAMPLE_RATE) {
      for (let index = 0; index < this.source.length; index += 1) {
        this.targetSamples.push(this.source[index] ?? 0);
      }
      this.source = new Float32Array(0);
      this.sourcePosition = 0;
    } else {
      while (this.sourcePosition + 1 < this.source.length) {
        const left = Math.floor(this.sourcePosition);
        const fraction = this.sourcePosition - left;
        const sample = (this.source[left] ?? 0) * (1 - fraction)
          + (this.source[left + 1] ?? 0) * fraction;
        this.targetSamples.push(sample);
        this.sourcePosition += this.step;
      }
      const consumed = Math.floor(this.sourcePosition);
      if (consumed > 0) {
        this.source = this.source.slice(consumed);
        this.sourcePosition -= consumed;
      }
    }

    const frames: ArrayBuffer[] = [];
    while (this.targetSamples.length >= FRAME_SAMPLES) {
      const frame = this.targetSamples.splice(0, FRAME_SAMPLES);
      const bytes = new ArrayBuffer(FRAME_SAMPLES * 2);
      const view = new DataView(bytes);
      for (let index = 0; index < frame.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, frame[index] ?? 0));
        view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      }
      frames.push(bytes);
    }
    return frames;
  }

  reset(): void {
    this.source = new Float32Array(0);
    this.sourcePosition = 0;
    this.targetSamples = [];
  }
}

export function speechWebSocketUrl(apiBase = "", locationValue = window.location): string {
  const parsed = apiBase
    ? new URL(apiBase, locationValue.href)
    : new URL(locationValue.href);
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  parsed.pathname = `${apiBase ? parsed.pathname.replace(/\/$/, "") : ""}/api/v2/speech/stream`;
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString();
}

export function assertStreamingMicrophoneAvailable(): void {
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  if (!window.isSecureContext && !local) {
    throw new Error("移动端流式语音需要 HTTPS；请通过工作站的 HTTPS 地址打开 H-IDE");
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持麦克风录音");
  if (!("AudioWorkletNode" in window)) {
    throw new Error("当前浏览器不支持 AudioWorklet；开发者可显式启用离线 WAV 转写降级");
  }
}

export interface BrowserStreamingSpeechOptions {
  url: string;
  csrfToken: string;
  locale?: string;
  callbacks?: StreamingSpeechCallbacks;
  now?: () => number;
  socketFactory?: (url: string, protocols: string | string[]) => WebSocket;
  audioContextFactory?: () => AudioContext;
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
}

export class BrowserStreamingSpeechSession {
  private readonly options: BrowserStreamingSpeechOptions;
  private readonly now: () => number;
  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private framer: StreamingPcm16Framer | null = null;
  private started = false;
  private finishing = false;
  private terminalResolve: (() => void) | null = null;
  private terminalPromise: Promise<void> | null = null;
  private speechEndedAt: number | null = null;
  private metrics: StreamingSpeechMetrics = { framesSent: 0, audioMsSent: 0 };

  constructor(options: BrowserStreamingSpeechOptions) {
    this.options = options;
    this.now = options.now ?? (() => performance.now());
  }

  async start(): Promise<void> {
    if (this.socket || this.stream) throw new Error("流式语音会话已经开始");
    assertStreamingMicrophoneAvailable();
    this.emitPhase("requesting_permission");
    const getUserMedia = this.options.getUserMedia
      ?? ((constraints: MediaStreamConstraints) => navigator.mediaDevices.getUserMedia(constraints));
    try {
      this.stream = await getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const makeContext = this.options.audioContextFactory
        ?? (() => new AudioContext({ latencyHint: "interactive" }));
      this.context = makeContext();
      // Creation and resume both happen inside the microphone button gesture;
      // this is required by iOS Safari's audio autoplay policy.
      await this.context.resume();
      await this.context.audioWorklet.addModule(new URL(
        "/audio/pcm-capture-worklet.js",
        window.location.origin,
      ).toString());
      this.source = this.context.createMediaStreamSource(this.stream);
      this.worklet = new AudioWorkletNode(this.context, "hide-pcm-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
      });
      this.framer = new StreamingPcm16Framer(this.context.sampleRate);
      this.worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        if (!this.started || this.finishing || !(event.data instanceof Float32Array)) return;
        for (const frame of this.framer?.push(event.data) ?? []) this.sendFrame(frame);
      };
      this.source.connect(this.worklet);
      this.metrics = { captureStartedAt: this.now(), framesSent: 0, audioMsSent: 0 };
      this.emitMetrics();
      this.emitPhase("connecting");
      await this.connect();
    } catch (error) {
      await this.releaseCapture();
      const activeSocket = this.socket as WebSocket | null;
      activeSocket?.close();
      this.socket = null;
      this.emitPhase("error");
      throw error;
    }
  }

  async finish(): Promise<void> {
    if (this.finishing) return this.terminalPromise ?? Promise.resolve();
    this.finishing = true;
    this.emitPhase("finalizing");
    await this.releaseCapture();
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "session.finish" }));
    } else {
      this.resolveTerminal();
    }
    try {
      await this.waitForTerminal(14_000);
    } catch (error) {
      this.socket?.close(1011, "final transcript timeout");
      this.socket = null;
      this.resolveTerminal();
      this.emitPhase("error");
      throw error;
    }
  }

  async cancel(): Promise<void> {
    this.finishing = true;
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "session.cancel" }));
    }
    await this.releaseCapture();
    this.socket?.close(1000, "client cancelled");
    this.socket = null;
    this.resolveTerminal();
    this.emitPhase("closed");
  }

  snapshotMetrics(): StreamingSpeechMetrics {
    return { ...this.metrics };
  }

  private connect(): Promise<void> {
    const socketFactory = this.options.socketFactory
      ?? ((url: string, protocols: string | string[]) => new WebSocket(url, protocols));
    this.socket = socketFactory(this.options.url, "speech-stream.v2");
    this.socket.binaryType = "arraybuffer";
    this.terminalPromise = new Promise((resolve) => { this.terminalResolve = resolve; });
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const timeout = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        reject(new Error("流式语音连接超时"));
      }, 8_000);
      const failStart = (message: string) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        reject(new Error(message));
      };
      this.socket!.onopen = () => {
        this.socket?.send(JSON.stringify({
          type: "session.start",
          locale: this.options.locale ?? "zh-CN",
          csrfToken: this.options.csrfToken,
          audio: { encoding: "pcm_s16le", sampleRateHz: TARGET_SAMPLE_RATE, channels: 1 },
          serverVad: true,
        }));
      };
      this.socket!.onmessage = (message) => {
        if (typeof message.data !== "string") return;
        let event: SpeechServerEvent;
        try {
          event = JSON.parse(message.data) as SpeechServerEvent;
        } catch {
          return;
        }
        if (event.type === "session.started" && !settled) {
          settled = true;
          window.clearTimeout(timeout);
          this.started = true;
          this.emitPhase("listening");
          resolve();
        }
        if (event.type === "error" && event.fatal) failStart(event.message || "流式语音服务拒绝了会话");
        this.handleServerEvent(event);
      };
      this.socket!.onerror = () => {
        failStart("无法连接流式语音服务");
        this.options.callbacks?.onError?.("流式语音连接异常；没有自动切换到离线转写");
      };
      this.socket!.onclose = () => {
        failStart("流式语音连接已关闭");
        this.resolveTerminal();
        if (this.started) this.emitPhase("closed");
      };
    });
  }

  private handleServerEvent(event: SpeechServerEvent): void {
    if (event.type === "speech.ended") this.speechEndedAt = this.now();
    if (event.type === "transcript.partial" && event.text?.trim()) {
      if (this.metrics.firstPartialMs === undefined && this.metrics.captureStartedAt !== undefined) {
        this.metrics.firstPartialMs = Math.max(0, this.now() - this.metrics.captureStartedAt);
        this.emitMetrics();
      }
      this.options.callbacks?.onPartial?.({
        text: event.text.trim(),
        confirmedText: event.confirmedText,
        utteranceId: event.utteranceId,
      });
    }
    if (event.type === "transcript.final" && event.text?.trim()) {
      if (this.speechEndedAt !== null) {
        this.metrics.endToFinalMs = Math.max(0, this.now() - this.speechEndedAt);
        this.emitMetrics();
      }
      this.options.callbacks?.onFinal?.({
        text: event.text.trim(),
        confirmedText: event.confirmedText,
        utteranceId: event.utteranceId,
      });
    }
    if (event.type === "error") {
      const message = event.message || "流式语音服务发生错误";
      this.options.callbacks?.onError?.(message);
      if (event.fatal) this.emitPhase("error");
    }
    if (event.type === "session.finished") {
      this.resolveTerminal();
      this.socket?.close(1000, "session finished");
      this.socket = null;
      this.emitPhase("closed");
    }
  }

  private sendFrame(frame: ArrayBuffer): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(frame);
    this.metrics.framesSent += 1;
    this.metrics.audioMsSent += FRAME_DURATION_MS;
    if (this.metrics.framesSent % 5 === 0) this.emitMetrics();
  }

  private async releaseCapture(): Promise<void> {
    if (this.worklet) this.worklet.port.onmessage = null;
    this.worklet?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.worklet = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.framer?.reset();
    this.framer = null;
  }

  private async waitForTerminal(timeoutMs: number): Promise<void> {
    if (!this.terminalPromise) return;
    let timeout: number | undefined;
    try {
      await Promise.race([
        this.terminalPromise,
        new Promise<void>((_, reject) => {
          timeout = window.setTimeout(() => reject(new Error("等待最终转写超时")), timeoutMs);
        }),
      ]);
    } finally {
      if (timeout !== undefined) window.clearTimeout(timeout);
    }
  }

  private resolveTerminal(): void {
    this.terminalResolve?.();
    this.terminalResolve = null;
  }

  private emitPhase(phase: StreamingSpeechPhase): void {
    this.options.callbacks?.onPhase?.(phase);
  }

  private emitMetrics(): void {
    this.options.callbacks?.onMetrics?.({ ...this.metrics });
  }
}

export function splitSpeechSentences(text: string, maximumCharacters = 220): string[] {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [];
  const sentences = normalized.match(/[^。！？!?；;\n]+[。！？!?；;\n]?/g) ?? [normalized];
  return sentences.flatMap((sentence) => {
    const value = sentence.trim();
    if (value.length <= maximumCharacters) return value ? [value] : [];
    const chunks: string[] = [];
    for (let offset = 0; offset < value.length; offset += maximumCharacters) {
      chunks.push(value.slice(offset, offset + maximumCharacters));
    }
    return chunks;
  });
}

export type TtsRequest = (text: string, signal: AbortSignal) => Promise<Response>;

export class StreamingPcmPlayer {
  private context: AudioContext | null = null;
  private controller: AbortController | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextStartTime = 0;

  async unlock(): Promise<void> {
    if (!this.context || this.context.state === "closed") {
      this.context = new AudioContext({ latencyHint: "interactive" });
    }
    await this.context.resume();
    // A silent one-sample source unlocks future scheduled playback on iOS.
    const buffer = this.context.createBuffer(1, 1, this.context.sampleRate);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.start();
  }

  async speak(
    text: string,
    request: TtsRequest,
    callbacks: { onFirstByte?: (milliseconds: number) => void; onError?: (message: string) => void } = {},
  ): Promise<void> {
    this.stop();
    await this.unlock();
    const context = this.context;
    if (!context) return;
    this.controller = new AbortController();
    const controller = this.controller;
    this.nextStartTime = context.currentTime + 0.04;
    try {
      let firstByteReported = false;
      for (const sentence of splitSpeechSentences(text)) {
        const requestedAt = performance.now();
        const response = await request(sentence, controller.signal);
        if (!response.ok || !response.body) throw new Error(`语音回复请求失败 (${response.status})`);
        const reader = response.body.getReader();
        let pending = new Uint8Array(0);
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value?.length) continue;
          if (!firstByteReported) {
            firstByteReported = true;
            callbacks.onFirstByte?.(Math.max(0, performance.now() - requestedAt));
          }
          const joined = new Uint8Array(pending.length + value.length);
          joined.set(pending);
          joined.set(value, pending.length);
          pending = joined;
          // Schedule about 50 ms at a time, avoiding hundreds of tiny WebAudio nodes.
          while (pending.length >= 2_400) {
            this.schedulePcm(pending.slice(0, 2_400), 24_000);
            pending = pending.slice(2_400);
          }
        }
        if (pending.length >= 2) this.schedulePcm(pending.slice(0, pending.length - pending.length % 2), 24_000);
      }
    } catch (error) {
      if (!controller.signal.aborted) callbacks.onError?.(error instanceof Error ? error.message : "语音回复播放失败");
    } finally {
      if (this.controller === controller) this.controller = null;
    }
  }

  stop(): void {
    this.controller?.abort();
    this.controller = null;
    for (const source of this.sources) {
      try { source.stop(); } catch { /* A source may have already ended. */ }
    }
    this.sources.clear();
    this.nextStartTime = 0;
  }

  async close(): Promise<void> {
    this.stop();
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.context = null;
  }

  private schedulePcm(bytes: Uint8Array, sampleRate: number): void {
    const context = this.context;
    if (!context || bytes.length < 2) return;
    const sampleCount = Math.floor(bytes.length / 2);
    const buffer = context.createBuffer(1, sampleCount, sampleRate);
    const channel = buffer.getChannelData(0);
    const view = new DataView(bytes.buffer, bytes.byteOffset, sampleCount * 2);
    for (let index = 0; index < sampleCount; index += 1) {
      channel[index] = view.getInt16(index * 2, true) / 0x8000;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => this.sources.delete(source);
    this.sources.add(source);
    const startAt = Math.max(context.currentTime + 0.025, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + sampleCount / sampleRate;
  }
}
