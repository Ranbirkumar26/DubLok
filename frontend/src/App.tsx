import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  Languages,
  Link2,
  Mic2,
  Play,
  SlidersHorizontal,
  Upload,
  Video,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  API_BASE,
  JobOptions,
  JobResponse,
  HealthResponse,
  Language,
  LanguageCode,
  SourceResponse,
  TranscriptPayload,
  api,
  absoluteUrl,
} from "./api";

const defaultOptions: Required<JobOptions> = {
  target_language: "hi",
  model_preset: "balanced",
  audio_mode: "full_replacement",
  subtitle_mode: "translated",
  speech_speed: 1,
  original_volume: 0.2,
  translated_volume: 1,
  voice_map: {},
};

const orderedStages = [
  "Video Input",
  "Analysis",
  "Translation",
  "Voice Settings",
  "Generate",
  "Output",
];

export function App() {
  const [languages, setLanguages] = useState<Language[]>([]);
  const [source, setSource] = useState<SourceResponse | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [options, setOptions] = useState<Required<JobOptions>>(defaultOptions);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.languages().then(setLanguages).catch((err) => setError(err.message));
    api.health().then(setHealth).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!job || ["completed", "failed"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      api.getJob(job.id).then(setJob).catch((err) => setError(err.message));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job]);

  const activeIndex = useMemo(() => {
    if (!source) return 0;
    if (!job) return 1;
    if (job.status === "completed") return 5;
    if (job.status.includes("audio") || job.status.includes("render")) return 4;
    if (job.translated_transcript) return 3;
    if (job.transcript) return 2;
    return 1;
  }, [source, job]);

  async function withBusy(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  function updateOption<K extends keyof Required<JobOptions>>(key: K, value: Required<JobOptions>[K]) {
    setOptions((current) => ({ ...current, [key]: value }));
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">
          <Languages size={28} />
        </div>
        <div>
          <h1>Local Video Dubbing Studio</h1>
          <p>Zero paid APIs. Local processing. Segment-synced dubbing.</p>
        </div>
      </header>

      <nav className="stepper" aria-label="Workflow steps">
        {orderedStages.map((step, index) => (
          <div className={`step ${index <= activeIndex ? "active" : ""}`} key={step}>
            <span>{index + 1}</span>
            {step}
          </div>
        ))}
      </nav>

      {error && (
        <div className="alert" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}
      {health?.engine_mode === "demo" && (
        <div className="alert" role="alert">
          <AlertTriangle size={18} />
          <span>Demo mode is active. Production dubbing requires ENGINE_MODE=production and local ML models.</span>
        </div>
      )}

      <div className="workspace">
        <section className="panel input-panel">
          <div className="panel-title">
            <Video size={20} />
            <h2>Video Input</h2>
          </div>
          <div className="input-grid">
            <label className="dropzone">
              <Upload size={28} />
              <span>Upload Video</span>
              <input
                type="file"
                accept=".mp4,.mov,.mkv,.webm,video/mp4,video/webm"
                disabled={busy}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  withBusy(async () => {
                    const uploaded = await api.uploadVideo(file);
                    setSource(uploaded);
                    setJob(null);
                  });
                }}
              />
            </label>
            <form
              className="url-form"
              onSubmit={(event) => {
                event.preventDefault();
                withBusy(async () => {
                  const accepted = await api.submitUrl(url);
                  setSource(accepted);
                  setJob(null);
                });
              }}
            >
              <label>
                <span>YouTube or Google Drive URL</span>
                <div className="url-row">
                  <Link2 size={18} />
                  <input
                    value={url}
                    onChange={(event) => setUrl(event.target.value)}
                    placeholder="https://..."
                    disabled={busy}
                  />
                  <button type="submit" disabled={busy || !url} title="Use URL">
                    <Play size={18} />
                  </button>
                </div>
              </label>
            </form>
          </div>

          {source && (
            <div className="source-preview">
              {source.preview_url ? (
                <video controls src={absoluteUrl(source.preview_url)} />
              ) : (
                <div className="pending-preview">
                  <Link2 size={32} />
                  <span>Public URL accepted</span>
                </div>
              )}
              <Metadata metadata={source.metadata || job?.metadata || null} />
            </div>
          )}

          {source && !job && (
            <button
              className="primary"
              disabled={busy}
              onClick={() =>
                withBusy(async () => {
                  const created = await api.createJob(source.source_id, options);
                  setJob(created);
                })
              }
            >
              <Mic2 size={18} />
              Start Analysis
            </button>
          )}
        </section>

        <section className="panel progress-panel">
          <div className="panel-title">
            <SlidersHorizontal size={20} />
            <h2>Progress</h2>
          </div>
          <div className="progress-bar">
            <span style={{ width: `${Math.max(0, Math.min(job?.progress || 0, 100))}%` }} />
          </div>
          <div className="status-line">
            <strong>{job?.status || "waiting"}</strong>
            <span>{job?.current_stage || "No job queued"}</span>
          </div>
          {job?.error && (
            <div className="inline-error">
              <AlertTriangle size={16} />
              {job.error}
            </div>
          )}
          <ol className="stage-log">
            {(job?.stage_log || []).slice(-8).map((entry, index) => (
              <li key={`${entry.stage}-${index}`}>
                <CheckCircle2 size={15} />
                <span>{entry.stage}</span>
                {entry.seconds !== undefined && <em>{entry.seconds.toFixed(1)}s</em>}
              </li>
            ))}
          </ol>
        </section>
      </div>

      <div className="workspace lower">
        <section className="panel transcript-panel">
          <div className="panel-title">
            <FileText size={20} />
            <h2>Analysis And Translation</h2>
          </div>

          <div className="settings-row">
            <Select
              label="Target"
              value={options.target_language}
              onChange={(value) => updateOption("target_language", value as LanguageCode)}
              options={languages.map((language) => ({ value: language.code, label: language.name }))}
            />
            <Select
              label="Quality"
              value={options.model_preset}
              onChange={(value) => updateOption("model_preset", value as Required<JobOptions>["model_preset"])}
              options={[
                { value: "fast", label: "Fast" },
                { value: "balanced", label: "Balanced" },
                { value: "quality", label: "High Quality" },
              ]}
            />
            <Select
              label="Subtitles"
              value={options.subtitle_mode}
              onChange={(value) => updateOption("subtitle_mode", value as Required<JobOptions>["subtitle_mode"])}
              options={[
                { value: "translated", label: "Translated" },
                { value: "original", label: "Original" },
                { value: "dual", label: "Original + Translated" },
                { value: "none", label: "None" },
              ]}
            />
          </div>

          {job?.transcript && (
            <div className="language-detect">
              <span>Detected language</span>
              <strong>{job.transcript.detected_language || "unknown"}</strong>
              {job.transcript.confidence !== null && job.transcript.confidence !== undefined && (
                <em>{Math.round(job.transcript.confidence * 100)}%</em>
              )}
            </div>
          )}

          <TranscriptEditor
            transcript={job?.translated_transcript || job?.transcript || null}
            translated={Boolean(job?.translated_transcript)}
            onChange={(payload) => {
              if (!job) return;
              setJob({
                ...job,
                [job.translated_transcript ? "translated_transcript" : "transcript"]: payload,
              });
            }}
          />

          <div className="button-row">
            <button
              disabled={!job?.transcript || busy || job.status.startsWith("queued") || job.status.startsWith("running")}
              onClick={() =>
                job &&
                withBusy(async () => {
                  const queued = await api.translate(job.id, options);
                  setJob(queued);
                })
              }
            >
              <Languages size={18} />
              Translate
            </button>
            <button
              disabled={!job?.translated_transcript || busy}
              onClick={() =>
                job?.translated_transcript &&
                withBusy(async () => {
                  const saved = await api.saveTranscript(job.id, job.translated_transcript!, "translated");
                  setJob(saved);
                })
              }
            >
              <FileText size={18} />
              Save Edits
            </button>
          </div>
        </section>

        <section className="panel voice-panel">
          <div className="panel-title">
            <Mic2 size={20} />
            <h2>Voice Settings</h2>
          </div>
          <Select
            label="Audio Mode"
            value={options.audio_mode}
            onChange={(value) => updateOption("audio_mode", value as Required<JobOptions>["audio_mode"])}
            options={[
              { value: "full_replacement", label: "Full Replacement (mute original)" },
              { value: "preserve_background", label: "Background Preservation" },
            ]}
          />
          {options.audio_mode === "full_replacement" && (
            <p className="mode-note">Final video uses generated speech only; the original audio track is replaced.</p>
          )}
          <Range
            label="Speech Speed"
            value={options.speech_speed}
            min={0.5}
            max={2}
            step={0.05}
            onChange={(value) => updateOption("speech_speed", value)}
          />
          {options.audio_mode === "preserve_background" && (
            <Range
              label="Background Volume"
              value={options.original_volume}
              min={0}
              max={1.5}
              step={0.05}
              onChange={(value) => updateOption("original_volume", value)}
            />
          )}
          <Range
            label="Translated Speech Volume"
            value={options.translated_volume}
            min={0}
            max={2}
            step={0.05}
            onChange={(value) => updateOption("translated_volume", value)}
          />
          <VoiceMap
            job={job}
            value={options.voice_map}
            onChange={(voice_map) => updateOption("voice_map", voice_map)}
          />
          <div className="button-row vertical">
            <button
              disabled={!job?.translated_transcript || busy}
              onClick={() =>
                job &&
                withBusy(async () => {
                  const queued = await api.generateAudio(job.id, options);
                  setJob(queued);
                })
              }
            >
              <Mic2 size={18} />
              Generate Audio
            </button>
            <button
              className="primary"
              disabled={!job || !["audio_ready", "queued_render", "running_render", "completed"].includes(job.status) || busy}
              onClick={() =>
                job &&
                withBusy(async () => {
                  const queued = await api.render(job.id);
                  setJob(queued);
                })
              }
            >
              <Video size={18} />
              Render Video
            </button>
          </div>
        </section>
      </div>

      <section className="panel output-panel">
        <div className="panel-title">
          <Download size={20} />
          <h2>Output</h2>
        </div>
        {job?.status === "completed" && job.artifacts.video ? (
          <div className="output-grid">
            <video controls src={absoluteUrl(job.artifacts.video.url)} />
            <div className="downloads">
              {Object.entries(job.artifacts).map(([key, artifact]) => (
                <a key={key} href={`${API_BASE}${artifact.url}`}>
                  <Download size={16} />
                  {artifact.name}
                </a>
              ))}
            </div>
          </div>
        ) : (
          <p className="empty-state">Final files appear here after rendering completes.</p>
        )}
      </section>
    </main>
  );
}

function Metadata({ metadata }: { metadata: SourceResponse["metadata"] | null }) {
  if (!metadata || typeof metadata.duration !== "number") {
    return <p className="empty-state">Metadata appears after validation.</p>;
  }
  const resolution = metadata.video?.width && metadata.video?.height
    ? `${metadata.video.width}x${metadata.video.height}`
    : "unknown";
  return (
    <dl className="metadata">
      <div>
        <dt>Duration</dt>
        <dd>{metadata.duration.toFixed(1)}s</dd>
      </div>
      <div>
        <dt>Resolution</dt>
        <dd>{resolution}</dd>
      </div>
      <div>
        <dt>FPS</dt>
        <dd>{metadata.video?.fps?.toFixed(2) || "unknown"}</dd>
      </div>
      <div>
        <dt>Codec</dt>
        <dd>{metadata.video?.codec || "unknown"}</dd>
      </div>
    </dl>
  );
}

function TranscriptEditor({
  transcript,
  translated,
  onChange,
}: {
  transcript: TranscriptPayload | null;
  translated: boolean;
  onChange: (payload: TranscriptPayload) => void;
}) {
  if (!transcript) {
    return <p className="empty-state">Transcript segments appear after analysis.</p>;
  }
  return (
    <div className="segments">
      {transcript.segments.map((segment, index) => (
        <div className="segment-row" key={segment.id}>
          <div className="segment-meta">
            <strong>{segment.speaker}</strong>
            <span>
              {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
            </span>
          </div>
          <textarea
            value={translated ? segment.translated_text || "" : segment.text}
            onChange={(event) => {
              const segments = transcript.segments.map((item, itemIndex) =>
                itemIndex === index
                  ? translated
                    ? { ...item, translated_text: event.target.value }
                    : { ...item, text: event.target.value }
                  : item,
              );
              onChange({ ...transcript, segments });
            }}
          />
          {translated && <small>{segment.text}</small>}
        </div>
      ))}
    </div>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Range({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="control">
      <span>
        {label} <em>{value.toFixed(2)}</em>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function VoiceMap({
  job,
  value,
  onChange,
}: {
  job: JobResponse | null;
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
}) {
  const speakers = Array.from(
    new Set((job?.translated_transcript?.segments || job?.transcript?.segments || []).map((segment) => segment.speaker)),
  );
  if (!speakers.length) return null;
  return (
    <div className="voice-map">
      {speakers.map((speaker) => (
        <label className="control" key={speaker}>
          <span>{speaker}</span>
          <input
            value={value[speaker] || ""}
            placeholder="Neutral voice"
            onChange={(event) => onChange({ ...value, [speaker]: event.target.value })}
          />
        </label>
      ))}
    </div>
  );
}
