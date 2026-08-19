import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BrowserStreamingSpeechSession,
  speechWebSocketUrl,
  splitSpeechSentences,
  StreamingPcmPlayer,
  StreamingPcm16Framer,
} from "./streaming-speech";

describe("streaming browser speech", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("resamples continuously and emits exact 40 ms PCM16 frames", () => {
    const framer = new StreamingPcm16Framer(48_000);
    const source = new Float32Array(48_004);
    for (let index = 0; index < source.length; index += 1) source[index] = Math.sin(index / 32);
    const frames: ArrayBuffer[] = [];
    for (let offset = 0; offset < source.length; offset += 128) {
      frames.push(...framer.push(source.slice(offset, offset + 128)));
    }

    expect(frames).toHaveLength(25);
    expect(frames.every((frame) => frame.byteLength === 1_280)).toBe(true);
    expect(new DataView(frames[0]).getInt16(2, true)).not.toBe(0);
  });

  it("builds a same-origin speech websocket without URL credentials", () => {
    const locationValue = {
      href: "https://workstation.local/app?token=must-not-leak",
    } as Location;
    const url = speechWebSocketUrl("https://workstation.local/control", locationValue);

    expect(url).toBe("wss://workstation.local/control/api/v2/speech/stream");
    expect(url).not.toContain("token");
  });

  it("splits TTS at sentence boundaries and bounds long requests", () => {
    expect(splitSpeechSentences("你好。现在开始任务！下一步呢？"))
      .toEqual(["你好。", "现在开始任务！", "下一步呢？"]);
    expect(splitSpeechSentences("长".repeat(500)).map((item) => item.length))
      .toEqual([220, 220, 60]);
  });

  it("requests TTS sentence by sentence and reports the first streamed byte", async () => {
    const started: number[] = [];
    const stopped = vi.fn();
    class FakeAudioContext {
      state = "running";
      sampleRate = 48_000;
      currentTime = 1;
      destination = {};
      resume = vi.fn().mockResolvedValue(undefined);
      close = vi.fn().mockImplementation(() => { this.state = "closed"; return Promise.resolve(); });
      createBuffer(_channels: number, samples: number) {
        const channel = new Float32Array(samples);
        return { getChannelData: () => channel };
      }
      createBufferSource() {
        return {
          buffer: null,
          onended: null,
          connect: vi.fn(),
          start: (when = 0) => started.push(when),
          stop: stopped,
        };
      }
    }
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const request = vi.fn().mockImplementation(async () => new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array(2_400));
          controller.close();
        },
      }),
      { status: 200 },
    ));
    const onFirstByte = vi.fn();
    const player = new StreamingPcmPlayer();

    await player.speak("第一句。第二句！", request, { onFirstByte });

    expect(request).toHaveBeenCalledTimes(2);
    expect(request.mock.calls.map(([text]) => text)).toEqual(["第一句。", "第二句！"]);
    expect(onFirstByte).toHaveBeenCalledOnce();
    expect(started.length).toBeGreaterThanOrEqual(3); // one iOS unlock source + two speech chunks
    player.stop();
    expect(stopped).toHaveBeenCalled();
  });

  it("sends CSRF in session.start then only contract-sized binary audio", async () => {
    const sent: unknown[] = [];
    let socketMessage: ((event: MessageEvent) => void) | null = null;
    class FakeSocket {
      static OPEN = 1;
      readyState = FakeSocket.OPEN;
      binaryType = "blob";
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      constructor() {
        queueMicrotask(() => this.onopen?.());
      }
      send(value: unknown) {
        sent.push(value);
        if (typeof value === "string") {
          const command = JSON.parse(value) as { type: string };
          if (command.type === "session.start") {
            queueMicrotask(() => this.onmessage?.(new MessageEvent("message", {
              data: JSON.stringify({ type: "session.started" }),
            })));
          }
          if (command.type === "session.finish") {
            queueMicrotask(() => this.onmessage?.(new MessageEvent("message", {
              data: JSON.stringify({ type: "session.finished" }),
            })));
          }
        }
      }
      close() { this.readyState = 3; this.onclose?.(); }
    }
    vi.stubGlobal("WebSocket", FakeSocket);

    const stop = vi.fn();
    vi.stubGlobal("navigator", {
      mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] }) },
    });
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: true });

    let workletPort: { onmessage: ((event: MessageEvent<Float32Array>) => void) | null } | null = null;
    class FakeAudioWorkletNode {
      port = { onmessage: null as ((event: MessageEvent<Float32Array>) => void) | null };
      constructor() { workletPort = this.port; }
      disconnect() { return undefined; }
    }
    vi.stubGlobal("AudioWorkletNode", FakeAudioWorkletNode);
    const context = {
      sampleRate: 16_000,
      state: "running",
      audioWorklet: { addModule: vi.fn().mockResolvedValue(undefined) },
      resume: vi.fn().mockResolvedValue(undefined),
      close: vi.fn().mockImplementation(function (this: { state: string }) {
        this.state = "closed";
        return Promise.resolve();
      }),
      createMediaStreamSource: vi.fn().mockReturnValue({ connect: vi.fn(), disconnect: vi.fn() }),
    };
    const session = new BrowserStreamingSpeechSession({
      url: "wss://workstation.local/api/v2/speech/stream",
      csrfToken: "csrf-browser-only",
      socketFactory: () => {
        const socket = new FakeSocket();
        socketMessage = socket.onmessage;
        return socket as unknown as WebSocket;
      },
      audioContextFactory: () => context as unknown as AudioContext,
    });

    await session.start();
    for (let index = 0; index < 5; index += 1) {
      workletPort!.onmessage?.(new MessageEvent("message", { data: new Float32Array(640).fill(0.2) }));
    }
    await session.finish();

    const start = JSON.parse(String(sent[0])) as Record<string, unknown>;
    expect(start).toMatchObject({
      type: "session.start",
      csrfToken: "csrf-browser-only",
      serverVad: true,
      audio: { encoding: "pcm_s16le", sampleRateHz: 16_000, channels: 1 },
    });
    expect(sent.filter((item) => item instanceof ArrayBuffer)).toHaveLength(5);
    expect(sent.filter((item) => item instanceof ArrayBuffer).every((item) => (item as ArrayBuffer).byteLength === 1_280)).toBe(true);
    expect(stop).toHaveBeenCalledOnce();
    expect(socketMessage).toBeNull();
  });
});
