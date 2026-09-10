"use client";

import Link from "next/link";
import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  LoadingRows,
  PageHeader,
  ScoreBadge,
  Select,
  StatusBadge,
  Toast,
  cx,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatDuration, formatRelative } from "@/lib/format";
import type { Clip } from "@/types/api";

const STATUSES = [
  "NEEDS_REVIEW",
  "APPROVED",
  "REJECTED",
  "PUBLISHED",
  "RENDERING",
  "FAILED",
];

export default function ClipsPage() {
  return (
    <Shell>
      <ClipsContent />
    </Shell>
  );
}

function ClipsContent() {
  const [status, setStatus] = useState("");
  const [order, setOrder] = useState("created_at");
  const [toast, setToast] = useState<string | null>(null);

  const clips = useQuery(
    () =>
      api.clips.list({
        status: status || undefined,
        order_by: order,
        limit: 60,
      }),
    [status, order],
  );

  return (
    <>
      <PageHeader
        title="Clips"
        description="Preview, approve or reject. Nothing is published without a decision."
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="max-w-[12rem]"
        >
          <option value="">All statuses</option>
          {STATUSES.map((value) => (
            <option key={value} value={value}>
              {value.replace(/_/g, " ")}
            </option>
          ))}
        </Select>
        <Select
          value={order}
          onChange={(e) => setOrder(e.target.value)}
          className="max-w-[12rem]"
        >
          <option value="created_at">Newest first</option>
          <option value="viral_score">Highest viral score</option>
          <option value="quality_score">Highest quality score</option>
        </Select>
        <span className="text-xs text-ink-500">
          {clips.data?.total ?? 0} clip{clips.data?.total === 1 ? "" : "s"}
        </span>
      </div>

      <ErrorState error={clips.error} onRetry={clips.refetch} />

      {clips.loading && !clips.data ? (
        <LoadingRows rows={4} />
      ) : clips.data && clips.data.items.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {clips.data.items.map((clip) => (
            <ClipCard
              key={clip.id}
              clip={clip}
              onChanged={(message) => {
                setToast(message);
                clips.refetch();
              }}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          title="No clips yet"
          description="Analyse a video and generate clips from the detected moments."
          action={
            <Link href="/videos">
              <Button variant="primary">Go to videos</Button>
            </Link>
          }
        />
      )}

      {toast && <Toast message={toast} tone="good" onDismiss={() => setToast(null)} />}
    </>
  );
}

function ClipCard({
  clip,
  onChanged,
}: {
  clip: Clip;
  onChanged: (message: string) => void;
}) {
  const approve = useMutation(() => api.clips.approve(clip.id));
  const reject = useMutation(() => api.clips.reject(clip.id));
  const [playing, setPlaying] = useState(false);

  const blocked = clip.safety_verdict === "BLOCK";
  const decided = ["APPROVED", "REJECTED", "PUBLISHED"].includes(clip.status);

  return (
    <Card className="flex flex-col overflow-hidden">
      <div className="relative aspect-[9/16] bg-ink-950">
        {playing && clip.media_url ? (
          <video
            src={clip.media_url}
            controls
            autoPlay
            className="h-full w-full object-contain"
          />
        ) : (
          <button
            onClick={() => setPlaying(true)}
            disabled={!clip.media_url}
            className="group relative h-full w-full"
            aria-label="Play preview"
          >
            {clip.thumbnail_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={clip.thumbnail_url}
                alt=""
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-ink-600">
                {clip.status === "RENDERING" ? "Rendering..." : "No preview"}
              </div>
            )}
            {clip.media_url && (
              <span className="absolute inset-0 flex items-center justify-center bg-ink-950/30 opacity-0 transition-opacity group-hover:opacity-100">
                <span className="flex h-11 w-11 items-center justify-center rounded-full bg-white/90 text-ink-950">
                  ▶
                </span>
              </span>
            )}
          </button>
        )}
        <div className="pointer-events-none absolute left-2 top-2 flex flex-wrap gap-1">
          <ScoreBadge score={clip.viral_score} />
          {clip.auto_approved && <Badge tone="accent">auto</Badge>}
        </div>
        <div className="pointer-events-none absolute bottom-2 right-2">
          <Badge tone="muted">{formatDuration(clip.duration_seconds)}</Badge>
        </div>
      </div>

      <div className="flex flex-1 flex-col p-3.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <StatusBadge status={clip.status} />
          {clip.safety_verdict && (
            <StatusBadge status={clip.safety_verdict} />
          )}
          {clip.quality_score != null && (
            <Badge tone="muted" title="Automated quality score">
              QC {clip.quality_score.toFixed(0)}
            </Badge>
          )}
        </div>

        <Link
          href={`/clips/${clip.id}`}
          className="mt-2 line-clamp-2 text-sm font-medium text-ink-50 hover:text-accent-300"
        >
          {clip.title ?? clip.hook_text ?? "Untitled clip"}
        </Link>
        <p className="mt-1 line-clamp-2 text-xs text-ink-500">
          {clip.video_title ?? ""}
        </p>

        {clip.review_note && (
          <p
            className={cx(
              "mt-2 rounded border px-2 py-1.5 text-[11px]",
              blocked
                ? "border-bad-500/30 bg-bad-500/10 text-bad-500"
                : "border-ink-700 bg-ink-850 text-ink-400",
            )}
          >
            {clip.review_note}
          </p>
        )}

        <div className="mt-auto flex items-center gap-1.5 pt-3">
          {!decided && (
            <>
              <Button
                size="sm"
                variant="success"
                loading={approve.pending}
                disabled={blocked}
                title={
                  blocked
                    ? "Blocked by the safety review and cannot be approved"
                    : undefined
                }
                onClick={async () => {
                  const result = await approve.run();
                  if (result) onChanged("Clip approved.");
                }}
              >
                Approve
              </Button>
              <Button
                size="sm"
                variant="danger"
                loading={reject.pending}
                onClick={async () => {
                  const result = await reject.run();
                  if (result) onChanged("Clip rejected.");
                }}
              >
                Reject
              </Button>
            </>
          )}
          <Link href={`/clips/${clip.id}`} className="ml-auto">
            <Button size="sm" variant="ghost">
              Edit
            </Button>
          </Link>
        </div>
        <div className="mt-2 text-[11px] text-ink-600">
          {formatRelative(clip.created_at)}
        </div>
      </div>
    </Card>
  );
}
