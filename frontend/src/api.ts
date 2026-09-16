export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type LanguageCode = "hi" | "en" | "ta" | "te" | "kn" | "ml" | "bn" | "mr";

export interface Language {
  code: LanguageCode;
  name: string;
}

export interface SourceResponse {
  source_id: string;
  input_type: "upload" | "url";
  status: string;
  metadata?: MediaMetadata | null;
  preview_url?: string | null;
  message?: string | null;
}

export interface HealthResponse {
  status: string;
  ffmpeg: boolean;
  ffprobe: boolean;
  engine_mode: "production" | "demo" | string;
}

export interface MediaMetadata {
  duration: number;
  size_bytes?: number;
  format?: string;
  video?: {
    width?: number;
    height?: number;
    fps?: number;
    codec?: string;
  };
  audio?: {
    codec?: string;
    sample_rate?: number;
    channels?: number;
  };
}

export interface TranscriptSegment {
  id: string;
  speaker: string;
  start: number;
  end: number;
  text: string;
  translated_text?: string | null;
}

export interface TranscriptPayload {
  detected_language?: string | null;
  confidence?: number | null;
  segments: TranscriptSegment[];
}

export interface JobResponse {
  id: string;
  source_id: string;
  status: string;
  current_stage?: string | null;
  progress: number;
  error?: string | null;
  target_language: LanguageCode;
  source_language_override?: LanguageCode | null;
  model_preset: "fast" | "balanced" | "quality";
  audio_mode: "full_replacement" | "preserve_background";
  subtitle_mode: "none" | "original" | "translated" | "dual";
  speech_speed: number;
  original_volume: number;
  translated_volume: number;
  voice_map: Record<string, string>;
  source?: SourceResponse | null;
  metadata?: MediaMetadata | null;
  transcript?: TranscriptPayload | null;
  translated_transcript?: TranscriptPayload | null;
  artifacts: Record<string, { name: string; url: string }>;
  stage_log: Array<{ stage: string; status: string; message?: string; seconds?: number; at: string }>;
  created_at: string;
  updated_at: string;
}

export interface JobOptions {
  target_language?: LanguageCode;
  model_preset?: "fast" | "balanced" | "quality";
  audio_mode?: "full_replacement" | "preserve_background";
  subtitle_mode?: "none" | "original" | "translated" | "dual";
  speech_speed?: number;
  original_volume?: number;
  translated_volume?: number;
  voice_map?: Record<string, string>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function absoluteUrl(path?: string | null): string | undefined {
  if (!path) return undefined;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export const api = {
  async health(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
  },
  async languages(): Promise<Language[]> {
    const payload = await request<{ languages: Language[] }>("/api/languages");
    return payload.languages;
  },
  async uploadVideo(file: File): Promise<SourceResponse> {
    const form = new FormData();
    form.append("file", file);
    return request<SourceResponse>("/api/video/upload", {
      method: "POST",
      body: form,
    });
  },
  async submitUrl(url: string): Promise<SourceResponse> {
    return request<SourceResponse>("/api/video/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  },
  async createJob(sourceId: string, options: Required<JobOptions>): Promise<JobResponse> {
    return request<JobResponse>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, ...options }),
    });
  },
  async getJob(jobId: string): Promise<JobResponse> {
    return request<JobResponse>(`/api/jobs/${jobId}`);
  },
  async translate(jobId: string, options: JobOptions): Promise<JobResponse> {
    return request<JobResponse>(`/api/jobs/${jobId}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
  },
  async saveTranscript(
    jobId: string,
    payload: TranscriptPayload,
    kind: "original" | "translated",
  ): Promise<JobResponse> {
    return request<JobResponse>(`/api/jobs/${jobId}/transcript?kind=${kind}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  async generateAudio(jobId: string, options: JobOptions): Promise<JobResponse> {
    return request<JobResponse>(`/api/jobs/${jobId}/generate-audio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
  },
  async render(jobId: string): Promise<JobResponse> {
    return request<JobResponse>(`/api/jobs/${jobId}/render`, { method: "POST" });
  },
};
