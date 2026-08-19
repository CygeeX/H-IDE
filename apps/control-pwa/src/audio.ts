export function resampleMono(
  source: Float32Array,
  sourceRate: number,
  targetRate = 16_000,
): Float32Array {
  if (!Number.isFinite(sourceRate) || sourceRate <= 0 || !Number.isFinite(targetRate) || targetRate <= 0) {
    throw new Error("音频采样率无效");
  }
  if (sourceRate === targetRate) return new Float32Array(source);
  const targetLength = Math.max(1, Math.round((source.length * targetRate) / sourceRate));
  const output = new Float32Array(targetLength);
  const scale = sourceRate / targetRate;
  for (let index = 0; index < targetLength; index += 1) {
    const position = index * scale;
    const left = Math.min(source.length - 1, Math.floor(position));
    const right = Math.min(source.length - 1, left + 1);
    const fraction = position - left;
    output[index] = (source[left] ?? 0) * (1 - fraction) + (source[right] ?? 0) * fraction;
  }
  return output;
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export function encodePcm16Wav(samples: Float32Array, sampleRate = 16_000): ArrayBuffer {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error("音频采样率无效");
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index] ?? 0));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

export class BrowserWavRecorder {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private chunks: Float32Array[] = [];

  async start() {
    if (this.stream) throw new Error("语音录制已经开始");
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持麦克风录音");
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16_000, echoCancellation: true, noiseSuppression: true },
    });
    try {
      this.context = new AudioContext();
      this.source = this.context.createMediaStreamSource(this.stream);
      this.processor = this.context.createScriptProcessor(4096, 1, 1);
      this.chunks = [];
      this.processor.onaudioprocess = (event) => {
        this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      this.source.connect(this.processor);
      this.processor.connect(this.context.destination);
    } catch (error) {
      try {
        await this.release();
      } catch {
        // Preserve the graph setup error while still making a best effort to
        // stop every microphone track acquired above.
        this.stream?.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      throw error;
    }
  }

  async stop(): Promise<Blob> {
    if (!this.stream || !this.context) throw new Error("语音录制尚未开始");
    const sampleRate = this.context.sampleRate;
    const totalLength = this.chunks.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    await this.release();
    if (!merged.length) throw new Error("没有录到有效语音");
    const samples = resampleMono(merged, sampleRate, 16_000);
    return new Blob([encodePcm16Wav(samples, 16_000)], { type: "audio/wav" });
  }

  async cancel() {
    await this.release();
    this.chunks = [];
  }

  private async release() {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.processor = null;
    this.source = null;
    this.stream = null;
    this.context = null;
  }
}
