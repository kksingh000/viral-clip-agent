"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  DimensionBar,
  EmptyState,
  ErrorState,
  LoadingRows,
  PageHeader,
  ScoreBadge,
  StatusBadge,
  Toast,
  cx,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import {
  formatCount,
  formatDuration,
  formatRelative,
  formatTimecode,
  titleCase,
} from "@/lib/format";
import type { Candidate, Clip, Page } from "@/types/api";

type Tab = "candidates" | "transcript" | "clips" | "scenes";

export default function VideoDetailPage() {
  return (
    <Shell>
      <VideoDetail />
    </Shell>
  );
}

function VideoDetail() {
  const params = useParams<{ id: string }>();
  const videoId = params.id;
  const [tab, setTab] = useState<Tab>("candidates");
  const [toast, setToast] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const video = useQuery(() => api.videos.get(videoId), [videoId], {
    pollMs: 8000,
  });
  const candidates = useQuery(() => api.videos.candidates(videoId), [videoId]);
  const clips = useQuery(
    () => api.clips.list({ video_id: videoId, limit: 50 }),
    [videoId],
  );

  const generate = useMutation((ids: string[]) =>
    api.clips.generate({ video_id: videoId, candidate_ids: ids }),
  );
  const analyze = useMutation(() => api.videos.analyze(videoId, true, false));

  const data = video.data;

  if (video.loading && !data) return <LoadingRows rows={4} />;
  if (video.error) return <ErrorState error={video.error} onRetry={video.refetch} />;
  if (!data) return null;

  const processable = [
    "USER_OWNED",
    "LICENSED",
    "AUTHORIZED",
    "USER_UPLOADED",
  ].includes(data.authorization_status);

  return (
    <>
      <div className="mb-2">
        <Link href="/videos" className="text-xs text-ink-400 hover:text-ink-200">
          ← Videos
        </Link>
      </div>

      <PageHeader
        title={data.title}
        description={[data.creator_name, data.category, data.language]
          .filter(Boolean)
          .join(" · ")}
        actions={
          <>
            <Button
              loading={analyze.pending}
              disabled={!processable || !data.has_media}
              onClick={async () => {
                await analyze.run();
                setToast("Re-analysis queued.");
                video.refetch();
              }}
            >
              Re-analyse
            </Button>
            <Button
              variant="primary"
              loading={generate.pending}
              disabled={!processable || selected.size === 0}
              onClick={async () => {
                const result = await generate.run([...selected]);
                if (result) {
                  setToast(`Generating ${selected.size} clip(s).`);
                  setSelected(new Set());
                  clips.refetch();
                }
              }}
            >
              Generate {selected.size > 0 ? `${selected.size} ` : ""}clip
              {selected.size === 1 ? "" : "s"}
            </Button>
          </>
        }
      />

      {!processable && (
        <div className="mb-4 rounded-lg border border-warn-500/30 bg-warn-500/10 px-4 py-3 text-sm text-warn-500">
          This video&apos;s rights status is{" "}
          <strong>{data.authorization_status}</strong>. It is stored as metadata
          only and cannot be analysed or published until you record that you own
          it or hold a licence.
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <div className="space-y-3">
          <Card className="overflow-hidden">
            <div className="aspect-video bg-ink-850">
              {data.media_url ? (
                <video
                  src={data.media_url}
                  controls
                  className="h-full w-full object-contain"
                />
              ) : data.thumbnail_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={data.thumbnail_url}
                  alt=""
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-ink-500">
                  No media attached
                </div>
              )}
            </div>
            <dl className="divide-y divide-ink-800 text-xs">
              <Row label="Status">
                <StatusBadge status={data.status} />
              </Row>
              <Row label="Rights">
                <StatusBadge status={data.authorization_status} />
              </Row>
              <Row label="Duration">{formatDuration(data.duration_seconds)}</Row>
              <Row label="Resolution">
                {data.width ? `${data.width}×${data.height}` : "--"}
                {data.fps ? ` @ ${Math.round(data.fps)}fps` : ""}
              </Row>
              <Row label="Audio">{data.audio_available ? "present" : "none"}</Row>
              {data.view_count != null && (
                <Row label="Views">{formatCount(data.view_count)}</Row>
              )}
              {data.trend_score != null && (
                <Row label="Trend score">
                  <ScoreBadge score={data.trend_score} />
                </Row>
              )}
              <Row label="Analysed">{formatRelative(data.analyzed_at)}</Row>
            </dl>
          </Card>

          {data.authorization_note && (
            <Card className="px-4 py-3">
              <div className="text-[11px] font-medium uppercase tracking-wide text-ink-400">
                Rights record
              </div>
              <p className="mt-1 text-xs text-ink-300">{data.authorization_note}</p>
            </Card>
          )}
        </div>

        <div>
          <div className="mb-3 flex gap-1 border-b border-ink-800">
            {(
              [
                ["candidates", `Moments (${candidates.data?.length ?? 0})`],
                ["clips", `Clips (${clips.data?.total ?? 0})`],
                ["transcript", "Transcript"],
                ["scenes", "Scenes"],
              ] as [Tab, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={cx(
                  "-mb-px border-b-2 px-3 py-2 text-sm transition-colors",
                  tab === key
                    ? "border-accent-500 font-medium text-ink-50"
                    : "border-transparent text-ink-400 hover:text-ink-200",
                )}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "candidates" && (
            <CandidatesTab
              videoId={videoId}
              candidates={candidates}
              selected={selected}
              onToggle={(id) =>
                setSelected((prev) => {
                  const next = new Set(prev);
                  if (next.has(id)) next.delete(id);
                  else next.add(id);
                  return next;
                })
              }
            />
          )}
          {tab === "clips" && <ClipsTab clips={clips} />}
          {tab === "transcript" && <TranscriptTab videoId={videoId} />}
          {tab === "scenes" && <ScenesTab videoId={videoId} />}
        </div>
      </div>

      {toast && <Toast message={toast} tone="good" onDismiss={() => setToast(null)} />}
    </>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-4 py-2">
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-ink-200">{children}</dd>
    </div>
  );
}

function CandidatesTab({
  candidates,
  selected,
  onToggle,
}: {
  videoId: string;
  candidates: ReturnType<typeof useQuery<Candidate[]>>;
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  if (candidates.loading && !candidates.data) return <LoadingRows rows={3} />;
  if (candidates.error)
    return <ErrorState error={candidates.error} onRetry={candidates.refetch} />;
  if (!candidates.data || candidates.data.length === 0) {
    return (
      <EmptyState
        title="No moments detected yet"
        description="Run an analysis. The agent reads the transcript and proposes self-contained spans."
      />
    );
  }

  return (
    <div className="space-y-3">
      {candidates.data.map((candidate) => (
        <CandidateCard
          key={candidate.id}
          candidate={candidate}
          selected={selected.has(candidate.id)}
          onToggle={() => onToggle(candidate.id)}
        />
      ))}
    </div>
  );
}

function CandidateCard({
  candidate,
  selected,
  onToggle,
}: {
  candidate: Candidate;
  selected: boolean;
  onToggle: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const dimensions = useMemo(
    () => Object.entries(candidate.score_breakdown?.dimensions ?? {}),
    [candidate],
  );
  const penalties = Object.entries(candidate.score_breakdown?.penalties ?? {});

  return (
    <Card className={cx(selected && "border-accent-500/60")}>
      <div className="flex items-start gap-3 px-4 py-3.5">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          className="mt-1 h-4 w-4 accent-indigo-500"
          aria-label="Select this moment"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <ScoreBadge score={candidate.viral_score} />
            <span className="font-mono text-xs text-ink-400">
              {formatTimecode(candidate.start_time)} –{" "}
              {formatTimecode(candidate.end_time)}
            </span>
            <Badge tone="muted">
              {candidate.duration_seconds.toFixed(1)}s
            </Badge>
            {candidate.context_lead_in > 0.2 && (
              <Badge tone="accent" title="Extra lead-in included for context">
                +{candidate.context_lead_in.toFixed(1)}s context
              </Badge>
            )}
            <StatusBadge status={candidate.status} />
          </div>

          {candidate.hook && (
            <p className="mt-2 text-sm font-medium text-ink-50">
              &ldquo;{candidate.hook}&rdquo;
            </p>
          )}
          {candidate.summary && (
            <p className="mt-1 text-xs text-ink-400">{candidate.summary}</p>
          )}

          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 text-xs text-accent-300 hover:text-accent-200"
          >
            {expanded ? "Hide reasoning" : "Why was this selected?"}
          </button>

          {expanded && (
            <div className="mt-3 space-y-3 rounded-lg border border-ink-700 bg-ink-850 p-3.5">
              {candidate.reason && (
                <p className="text-xs leading-relaxed text-ink-300">
                  {candidate.reason}
                </p>
              )}
              {dimensions.length > 0 && (
                <div className="space-y-1.5">
                  {dimensions.map(([name, value]) => (
                    <DimensionBar
                      key={name}
                      label={titleCase(name)}
                      value={Number(value)}
                    />
                  ))}
                </div>
              )}
              {penalties.length > 0 && (
                <div className="text-xs text-bad-500">
                  Penalties:{" "}
                  {penalties
                    .map(([name, value]) => `${titleCase(name)} −${value}`)
                    .join(", ")}
                </div>
              )}
              {candidate.score_breakdown?.adjustments?.length ? (
                <div className="text-xs text-ink-500">
                  Boundary adjustments:{" "}
                  {candidate.score_breakdown.adjustments.join("; ")}
                </div>
              ) : null}
              {candidate.transcript_excerpt && (
                <p className="border-l-2 border-ink-600 pl-3 text-xs italic leading-relaxed text-ink-400">
                  {candidate.transcript_excerpt}
                </p>
              )}
              <div className="text-[11px] text-ink-600">
                Scored by {candidate.model_name ?? "unknown"}
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function ClipsTab({
  clips,
}: {
  clips: ReturnType<typeof useQuery<Page<Clip>>>;
}) {
  if (clips.loading && !clips.data) return <LoadingRows rows={3} />;
  if (!clips.data || clips.data.items.length === 0) {
    return (
      <EmptyState
        title="No clips rendered"
        description="Select one or more moments above and generate."
      />
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {clips.data.items.map((clip) => (
        <Link key={clip.id} href={`/clips/${clip.id}`}>
          <Card className="overflow-hidden transition-colors hover:border-ink-600">
            <div className="aspect-[9/16] max-h-64 bg-ink-850">
              {clip.thumbnail_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={clip.thumbnail_url}
                  alt=""
                  className="h-full w-full object-cover"
                />
              )}
            </div>
            <div className="p-3">
              <div className="flex items-center justify-between gap-2">
                <StatusBadge status={clip.status} />
                <ScoreBadge score={clip.viral_score} />
              </div>
              <p className="mt-1.5 line-clamp-2 text-xs text-ink-200">
                {clip.title ?? clip.hook_text ?? "Untitled"}
              </p>
            </div>
          </Card>
        </Link>
      ))}
    </div>
  );
}

function TranscriptTab({ videoId }: { videoId: string }) {
  const transcript = useQuery(() => api.videos.transcript(videoId), [videoId]);

  if (transcript.loading && !transcript.data) return <LoadingRows rows={4} />;
  if (transcript.error) {
    return (
      <EmptyState
        title="No transcript"
        description="Run an analysis to transcribe this video."
      />
    );
  }
  const data = transcript.data;
  if (!data) return null;

  return (
    <Card>
      <CardHeader
        title={`Transcript · ${data.word_count} words`}
        description={`${data.provider}${data.model ? ` (${data.model})` : ""} · ${
          data.language ?? "unknown language"
        }`}
      />
      {data.is_synthetic && (
        <div className="border-b border-warn-500/25 bg-warn-500/10 px-5 py-2.5 text-xs text-warn-500">
          No transcription provider is configured. Segment timings are real, but
          the words are placeholders — moment detection cannot read what is said.
        </div>
      )}
      <div className="max-h-[32rem] overflow-y-auto p-4">
        <ul className="space-y-2">
          {data.segments.map((segment) => (
            <li
              key={segment.id}
              className="flex gap-3 rounded-lg px-2 py-1.5 hover:bg-ink-850"
            >
              <span className="w-16 shrink-0 pt-0.5 font-mono text-[11px] text-ink-500">
                {formatTimecode(segment.start_time)}
              </span>
              <p className="flex-1 text-sm leading-relaxed text-ink-200">
                {segment.text}
              </p>
              {segment.prefilter_score != null && (
                <span
                  className="w-8 shrink-0 text-right text-[11px] tabular-nums text-ink-600"
                  title="Deterministic signal prior used to pre-filter before the model call"
                >
                  {segment.prefilter_score.toFixed(2)}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}

function ScenesTab({ videoId }: { videoId: string }) {
  const scenes = useQuery(() => api.videos.scenes(videoId), [videoId]);
  if (scenes.loading && !scenes.data) return <LoadingRows rows={3} />;
  if (!scenes.data || scenes.data.length === 0) {
    return <EmptyState title="No scenes detected" />;
  }
  return (
    <Card className="p-4">
      <div className="flex flex-wrap gap-1.5">
        {scenes.data.map((scene) => (
          <span
            key={scene.id}
            className="rounded border border-ink-700 bg-ink-850 px-2 py-1 font-mono text-[11px] text-ink-300"
            title={`Scene ${scene.index}`}
          >
            {formatTimecode(scene.start_time)} → {formatTimecode(scene.end_time)}
          </span>
        ))}
      </div>
    </Card>
  );
}
