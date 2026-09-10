"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Field,
  Input,
  LoadingRows,
  PageHeader,
  ScoreBadge,
  Select,
  StatusBadge,
  TextArea,
  Toast,
  cx,
} from "@/components/ui";
import { useJobWatcher, useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatDuration, formatTimecode, titleCase } from "@/lib/format";
import type {
  CaptionPosition,
  CaptionStyle,
  ClipDetail,
  CropMode,
} from "@/types/api";

const CROP_MODES: CropMode[] = ["SMART", "CENTER", "BLUR_PAD", "SPLIT_SPEAKERS"];
const CAPTION_STYLES: CaptionStyle[] = [
  "WORD_HIGHLIGHT",
  "KARAOKE",
  "BOLD",
  "NORMAL",
  "MINIMAL",
  "PODCAST",
  "NONE",
];
const CAPTION_POSITIONS: CaptionPosition[] = [
  "LOWER_THIRD",
  "CENTER",
  "UPPER_THIRD",
  "BOTTOM",
  "TOP",
];

export default function ClipEditorPage() {
  return (
    <Shell>
      <ClipEditor />
    </Shell>
  );
}

function ClipEditor() {
  const params = useParams<{ id: string }>();
  const clipId = params.id;
  const [toast, setToast] = useState<{ message: string; tone: "good" | "bad" } | null>(
    null,
  );
  const [jobId, setJobId] = useState<string | null>(null);

  const clip = useQuery(() => api.clips.get(clipId), [clipId]);
  const { job, active } = useJobWatcher(jobId, (id) => api.jobs.get(id), (settled) => {
    setJobId(null);
    if (settled.status === "COMPLETED") {
      const newId = (settled.result?.clip_id as string) ?? null;
      setToast({ message: "Re-render finished.", tone: "good" });
      if (newId && newId !== clipId) {
        window.location.href = `/clips/${newId}`;
        return;
      }
      clip.refetch();
    } else {
      setToast({
        message: settled.error_message ?? "The re-render failed.",
        tone: "bad",
      });
    }
  });

  if (clip.loading && !clip.data) return <LoadingRows rows={4} />;
  if (clip.error) return <ErrorState error={clip.error} onRetry={clip.refetch} />;
  const data = clip.data;
  if (!data) return null;

  return (
    <>
      <div className="mb-2">
        <Link href="/clips" className="text-xs text-ink-400 hover:text-ink-200">
          ← Clips
        </Link>
      </div>

      <PageHeader
        title={data.title ?? "Untitled clip"}
        description={`${formatDuration(data.duration_seconds)} · ${data.width}×${data.height} · from ${data.video_title ?? "source"}`}
        actions={
          <>
            <a href={api.clips.downloadUrl(data.id)} download>
              <Button>Download</Button>
            </a>
            <Decision clip={data} onChanged={(m) => {
              setToast({ message: m, tone: "good" });
              clip.refetch();
            }} />
          </>
        }
      />

      {active && job && (
        <div className="mb-4 rounded-lg border border-accent-500/30 bg-accent-500/10 px-4 py-3 text-sm text-accent-300">
          Re-rendering — {job.stage ?? job.status.toLowerCase()} (
          {Math.round(job.progress * 100)}%)
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,340px)_1fr]">
        <div className="space-y-3">
          <Card className="overflow-hidden">
            <div className="aspect-[9/16] bg-ink-950">
              {data.media_url ? (
                <video
                  src={data.media_url}
                  controls
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-ink-600">
                  No rendered file
                </div>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-1.5 border-t border-ink-800 p-3">
              <StatusBadge status={data.status} />
              {data.safety_verdict && <StatusBadge status={data.safety_verdict} />}
              <ScoreBadge score={data.viral_score} />
              {data.quality_score != null && (
                <Badge tone="muted">QC {data.quality_score.toFixed(0)}</Badge>
              )}
              {data.regeneration_count > 0 && (
                <Badge tone="muted">{data.regeneration_count} re-render(s)</Badge>
              )}
            </div>
          </Card>

          <VariantsCard clip={data} onQueued={(id) => setJobId(id)} />
          <ChecksCard clip={data} />
        </div>

        <div className="space-y-4">
          <RenderPanel
            clip={data}
            disabled={active}
            onQueued={(id) => setJobId(id)}
          />
          <CopyPanel
            clip={data}
            onSaved={() => {
              setToast({ message: "Copy saved.", tone: "good" });
              clip.refetch();
            }}
          />
          <CaptionsPanel clip={data} />
        </div>
      </div>

      {toast && (
        <Toast
          message={toast.message}
          tone={toast.tone}
          onDismiss={() => setToast(null)}
        />
      )}
    </>
  );
}

function Decision({
  clip,
  onChanged,
}: {
  clip: ClipDetail;
  onChanged: (message: string) => void;
}) {
  const approve = useMutation(() => api.clips.approve(clip.id));
  const reject = useMutation(() => api.clips.reject(clip.id));
  const blocked = clip.safety_verdict === "BLOCK";

  if (clip.status === "PUBLISHED") return <Badge tone="good">Published</Badge>;

  return (
    <>
      <Button
        variant="danger"
        loading={reject.pending}
        onClick={async () => {
          if (await reject.run()) onChanged("Clip rejected.");
        }}
      >
        Reject
      </Button>
      <Button
        variant="success"
        loading={approve.pending}
        disabled={blocked}
        title={blocked ? "Blocked by the safety review" : undefined}
        onClick={async () => {
          if (await approve.run()) onChanged("Clip approved.");
        }}
      >
        Approve
      </Button>
    </>
  );
}

/**
 * Trim + framing + caption style. Any of these changes the picture, so the
 * panel queues a re-render rather than pretending to edit in place.
 */
function RenderPanel({
  clip,
  disabled,
  onQueued,
}: {
  clip: ClipDetail;
  disabled: boolean;
  onQueued: (jobId: string) => void;
}) {
  const [start, setStart] = useState(clip.start_time);
  const [end, setEnd] = useState(clip.end_time);
  const [cropMode, setCropMode] = useState<CropMode>(clip.crop_mode);
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>(clip.caption_style);
  const [captionPosition, setCaptionPosition] = useState<CaptionPosition>(
    clip.caption_position,
  );
  const [hook, setHook] = useState(clip.hook_text ?? "");
  const [regenerateHook, setRegenerateHook] = useState(false);

  const regenerate = useMutation(() =>
    api.clips.regenerate(clip.id, {
      crop_mode: cropMode,
      caption_style: captionStyle,
      caption_position: captionPosition,
      hook_text: regenerateHook ? null : hook || null,
      regenerate_hook: regenerateHook,
    }),
  );

  const duration = Math.max(0, end - start);
  const changed =
    cropMode !== clip.crop_mode ||
    captionStyle !== clip.caption_style ||
    captionPosition !== clip.caption_position ||
    hook !== (clip.hook_text ?? "") ||
    regenerateHook;

  return (
    <Card>
      <CardHeader
        title="Render settings"
        description="Changing framing, captions or the hook re-renders the clip."
      />
      <div className="space-y-4 p-5">
        <Timeline
          clip={clip}
          start={start}
          end={end}
          onChange={(nextStart, nextEnd) => {
            setStart(nextStart);
            setEnd(nextEnd);
          }}
        />
        <div className="flex flex-wrap items-center gap-4 text-xs text-ink-400">
          <span>
            In <span className="font-mono text-ink-200">{formatTimecode(start)}</span>
          </span>
          <span>
            Out <span className="font-mono text-ink-200">{formatTimecode(end)}</span>
          </span>
          <span>
            Length{" "}
            <span className="font-mono text-ink-200">{duration.toFixed(2)}s</span>
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Framing">
            <Select
              value={cropMode}
              onChange={(e) => setCropMode(e.target.value as CropMode)}
            >
              {CROP_MODES.map((value) => (
                <option key={value} value={value}>
                  {titleCase(value)}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Caption style">
            <Select
              value={captionStyle}
              onChange={(e) => setCaptionStyle(e.target.value as CaptionStyle)}
            >
              {CAPTION_STYLES.map((value) => (
                <option key={value} value={value}>
                  {titleCase(value)}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Caption position">
            <Select
              value={captionPosition}
              onChange={(e) =>
                setCaptionPosition(e.target.value as CaptionPosition)
              }
            >
              {CAPTION_POSITIONS.map((value) => (
                <option key={value} value={value}>
                  {titleCase(value)}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <Field
          label="Opening hook overlay"
          hint="Must describe the clip truthfully. Leave blank for no overlay."
        >
          <Input
            value={hook}
            disabled={regenerateHook}
            onChange={(e) => setHook(e.target.value)}
            maxLength={60}
            placeholder="Most people get this backwards"
          />
        </Field>

        {clip.hook_alternatives.length > 0 && !regenerateHook && (
          <div className="flex flex-wrap gap-1.5">
            {clip.hook_alternatives.map((alternative, index) => (
              <button
                key={index}
                onClick={() => setHook(alternative.text)}
                className={cx(
                  "rounded-md border px-2 py-1 text-[11px] transition-colors",
                  hook === alternative.text
                    ? "border-accent-500 bg-accent-500/15 text-accent-200"
                    : "border-ink-700 bg-ink-850 text-ink-300 hover:border-ink-600",
                )}
                title={alternative.rationale}
              >
                {alternative.text}
              </button>
            ))}
          </div>
        )}

        <label className="flex items-center gap-2 text-xs text-ink-300">
          <input
            type="checkbox"
            checked={regenerateHook}
            onChange={(e) => setRegenerateHook(e.target.checked)}
            className="h-3.5 w-3.5 accent-indigo-500"
          />
          Write a new hook from the transcript
        </label>

        <ErrorState error={regenerate.error} />

        <div className="flex items-center gap-2">
          <Button
            variant="primary"
            loading={regenerate.pending}
            disabled={disabled || (!changed && start === clip.start_time && end === clip.end_time)}
            onClick={async () => {
              const result = await regenerate.run();
              if (result) onQueued(result.job_id);
            }}
          >
            Re-render clip
          </Button>
          <span className="text-xs text-ink-500">
            The original stays in history until the new render succeeds.
          </span>
        </div>
      </div>
    </Card>
  );
}

/** Lightweight trim strip: drag the handles, scrub with the ruler. */
function Timeline({
  clip,
  start,
  end,
  onChange,
}: {
  clip: ClipDetail;
  start: number;
  end: number;
  onChange: (start: number, end: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<"start" | "end" | null>(null);

  // The visible window is the original cut plus a few seconds of headroom, so
  // the handles have somewhere to travel in both directions.
  const padding = Math.max(4, clip.duration_seconds * 0.35);
  const windowStart = Math.max(0, clip.start_time - padding);
  const windowEnd = clip.end_time + padding;
  const span = Math.max(0.1, windowEnd - windowStart);

  const toRatio = (value: number) => (value - windowStart) / span;

  useEffect(() => {
    if (!dragging) return;

    const move = (event: MouseEvent) => {
      const track = trackRef.current;
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      const value = windowStart + ratio * span;
      if (dragging === "start") onChange(Math.min(value, end - 1), end);
      else onChange(start, Math.max(value, start + 1));
    };
    const up = () => setDragging(null);

    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragging, start, end, onChange, span, windowStart]);

  return (
    <div>
      <div
        ref={trackRef}
        className="relative h-12 select-none rounded-lg border border-ink-700 bg-ink-850"
      >
        <div
          className="absolute inset-y-0 rounded-md bg-accent-500/25 ring-1 ring-inset ring-accent-500/60"
          style={{
            left: `${toRatio(start) * 100}%`,
            width: `${(toRatio(end) - toRatio(start)) * 100}%`,
          }}
        />
        {(["start", "end"] as const).map((handle) => (
          <button
            key={handle}
            onMouseDown={() => setDragging(handle)}
            aria-label={`Drag ${handle} handle`}
            className="absolute inset-y-0 w-3 -translate-x-1/2 cursor-ew-resize rounded bg-accent-400 hover:bg-accent-300"
            style={{
              left: `${toRatio(handle === "start" ? start : end) * 100}%`,
            }}
          />
        ))}
        {/* Caption cue marks, so trimming does not silently orphan a line. */}
        {clip.captions.map((cue, index) => (
          <span
            key={index}
            className="absolute bottom-0 h-1.5 bg-ink-400/60"
            style={{
              left: `${toRatio(clip.start_time + cue.start) * 100}%`,
              width: `${((cue.end - cue.start) / span) * 100}%`,
            }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-600">
        <span>{formatTimecode(windowStart)}</span>
        <span>{formatTimecode(windowEnd)}</span>
      </div>
    </div>
  );
}

function CopyPanel({
  clip,
  onSaved,
}: {
  clip: ClipDetail;
  onSaved: () => void;
}) {
  const [title, setTitle] = useState(clip.title ?? "");
  const [description, setDescription] = useState(clip.description ?? "");
  const [caption, setCaption] = useState(clip.instagram_caption ?? "");
  const [hashtags, setHashtags] = useState((clip.hashtags ?? []).join(" "));

  const save = useMutation(() =>
    api.clips.update(clip.id, {
      title,
      description,
      instagram_caption: caption,
      hashtags: hashtags
        .split(/[\s,]+/)
        .map((t) => t.replace(/^#/, ""))
        .filter(Boolean),
    }),
  );

  return (
    <Card>
      <CardHeader
        title="Packaging"
        description="Text-only edits save immediately; no re-render needed."
      />
      <div className="space-y-4 p-5">
        <Field label="YouTube Shorts title" hint={`${title.length}/100`}>
          <Input
            value={title}
            maxLength={100}
            onChange={(e) => setTitle(e.target.value)}
          />
        </Field>
        <Field label="Instagram caption" hint={`${caption.length}/300`}>
          <TextArea
            rows={3}
            maxLength={300}
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
          />
        </Field>
        <Field label="Description">
          <TextArea
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </Field>
        <Field label="Hashtags" hint="Space separated, without the #.">
          <Input value={hashtags} onChange={(e) => setHashtags(e.target.value)} />
        </Field>

        <ErrorState error={save.error} />
        <Button
          variant="primary"
          loading={save.pending}
          onClick={async () => {
            if (await save.run()) onSaved();
          }}
        >
          Save copy
        </Button>
      </div>
    </Card>
  );
}

function CaptionsPanel({ clip }: { clip: ClipDetail }) {
  if (!clip.captions.length) {
    return (
      <Card className="p-5">
        <EmptyState
          title="No captions"
          description="This clip was rendered without burned-in captions."
        />
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader
        title={`Captions · ${clip.captions.length} cues`}
        description="Timings come from word-level transcription."
        action={
          clip.subtitle_url ? (
            <a
              href={clip.subtitle_url}
              className="text-xs text-accent-300 hover:text-accent-200"
              download
            >
              Download .srt
            </a>
          ) : undefined
        }
      />
      <div className="max-h-72 overflow-y-auto p-4">
        <ul className="space-y-1">
          {clip.captions.map((cue, index) => (
            <li key={index} className="flex gap-3 rounded px-2 py-1 hover:bg-ink-850">
              <span className="w-20 shrink-0 font-mono text-[11px] text-ink-500">
                {formatTimecode(cue.start)}
              </span>
              <span className="text-xs text-ink-200">{cue.text}</span>
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}

function VariantsCard({
  clip,
  onQueued,
}: {
  clip: ClipDetail;
  onQueued: (jobId: string) => void;
}) {
  const create = useMutation(() => api.clips.variants(clip.id));

  return (
    <Card>
      <CardHeader
        title="Variants"
        description="Alternative framing and openings for the same words."
      />
      <div className="space-y-2 p-4">
        {clip.variants.length > 0 ? (
          clip.variants.map((variant) => (
            <div
              key={variant.id}
              className="flex items-center gap-3 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2"
            >
              <Badge tone="accent">{variant.label}</Badge>
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs text-ink-300">
                  {variant.description}
                </p>
              </div>
              {variant.media_url && (
                <a
                  href={variant.media_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-accent-300 hover:text-accent-200"
                >
                  Preview
                </a>
              )}
            </div>
          ))
        ) : (
          <p className="text-xs text-ink-500">
            No variants yet. Variants change framing, captions and lead-in only —
            never what is said.
          </p>
        )}
        <Button
          size="sm"
          loading={create.pending}
          onClick={async () => {
            const result = await create.run();
            if (result) onQueued(result.job_id);
          }}
        >
          Generate variants
        </Button>
      </div>
    </Card>
  );
}

function ChecksCard({ clip }: { clip: ClipDetail }) {
  const latestQuality = clip.quality_checks.at(-1);
  const latestSafety = clip.safety_checks.at(0);

  return (
    <Card>
      <CardHeader title="Automated checks" />
      <div className="space-y-3 p-4 text-xs">
        {latestQuality ? (
          <div>
            <div className="flex items-center justify-between">
              <span className="text-ink-400">Quality</span>
              <span className="flex items-center gap-2">
                <StatusBadge status={latestQuality.status} />
                <span className="tabular-nums text-ink-200">
                  {latestQuality.score.toFixed(0)}/{latestQuality.threshold.toFixed(0)}
                </span>
              </span>
            </div>
            {latestQuality.issues.length > 0 && (
              <ul className="mt-2 space-y-1">
                {latestQuality.issues.map((issue, index) => (
                  <li key={index} className="flex gap-2 text-[11px]">
                    <Badge
                      tone={
                        issue.severity === "CRITICAL" || issue.severity === "MAJOR"
                          ? "bad"
                          : "warn"
                      }
                    >
                      {issue.severity}
                    </Badge>
                    <span className="text-ink-400">{issue.message}</span>
                  </li>
                ))}
              </ul>
            )}
            {latestQuality.recommendations.length > 0 && (
              <ul className="mt-2 list-inside list-disc text-[11px] text-ink-500">
                {latestQuality.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <p className="text-ink-500">No quality check recorded.</p>
        )}

        <div className="border-t border-ink-800 pt-3">
          <div className="flex items-center justify-between">
            <span className="text-ink-400">Safety</span>
            <StatusBadge status={latestSafety?.verdict} />
          </div>
          {latestSafety?.rationale && (
            <p className="mt-1.5 text-[11px] leading-relaxed text-ink-500">
              {latestSafety.rationale}
            </p>
          )}
          {latestSafety && latestSafety.flagged_categories.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {latestSafety.flagged_categories.map((category) => (
                <Badge key={category} tone="warn">
                  {category.replace(/_/g, " ")}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
