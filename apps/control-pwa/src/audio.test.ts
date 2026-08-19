import { afterEach, describe, expect, it, vi } from "vitest";
import { BrowserWavRecorder, encodePcm16Wav, resampleMono } from "./audio";

describe("browser WAV capture", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("resamples mono audio to 16 kHz and emits 16-bit PCM WAV", () => {
    const source = new Float32Array(48_000).fill(0).map((_, index) => Math.sin(index / 20));
    const samples = resampleMono(source, 48_000, 16_000);
    const wav = encodePcm16Wav(samples, 16_000);
    const view = new DataView(wav);

    expect(samples.length).toBe(16_000);
    expect(String.fromCharCode(...new Uint8Array(wav.slice(0, 4)))).toBe("RIFF");
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16_000);
    expect(view.getUint16(34, true)).toBe(16);
  });

  it("rejects invalid sample rates", () => {
    expect(() => resampleMono(new Float32Array([0]), 0, 16_000)).toThrow();
  });

  it("releases the microphone when audio graph setup fails", async () => {
    const stop = vi.fn();
    vi.stubGlobal("navigator", {
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] }),
      },
    });
    vi.stubGlobal("AudioContext", class {
      constructor() {
        throw new Error("audio context unavailable");
      }
    });

    await expect(new BrowserWavRecorder().start()).rejects.toThrow("audio context unavailable");
    expect(stop).toHaveBeenCalledOnce();
  });
});
