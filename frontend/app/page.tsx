"use client";

import Link from "next/link";

import { Shell } from "@/components/Shell";
import {
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  LoadingRows,
  PageHeader,
  ProgressBar,
  ScoreBadge,
  StatCard,
  StatusBadge,
} from "@/components/ui";
import { useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatDuration, formatRelative, formatScore, formatUsd } from "@/lib/format";

export default function DashboardPage() {
  return (
    <Shell>
      <DashboardContent />
    </Shell>
  );
}

function DashboardContent() {
  const dashboard = useQuery(() => api.analytics.dashboard(), [], { pollMs: 15000 });
  const jobs = useQuery(
    () => api.jobs.list({ limit: 6 }),
    [],
    { pollMs: 5000 },
  );
  const review = useQuery(
    () => api.clips.list({ status: "NEEDS_REVIEW", limit: 5, order_by: "viral_score" }),
    [],
  );

  const data = dashboard.data;

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="What the pipeline discovered, analysed and produced."
      />

      {dashboard.error && (
        <ErrorState error={dashboard.error} onRetry={dashboard.refetch} />
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Discovered today"
          value={data?.videos_discovered_today ?? "--"}
          hint="Metadata only"
        />
        <StatCard label="Videos analysed" value={data?.videos_analyzed ?? "--"} />
        <StatCard label="Clips generated" value={data?.clips_generated ?? "--"} />
        <StatCard
          label="Awaiting review"
          value={data?.clips_awaiting_review ?? "--"}
          tone={data && data.clips_awaiting_review > 0 ? "warn" : "neutral"}
        />
        <StatCard
          label="Clips approved"
          value={data?.clips_approved ?? "--"}
          tone="good"
        />
        <StatCard
          label="Average viral score"
          value={formatScore(data?.average_viral_score)}
        />
        <StatCard
          label="Active jobs"
          value={data?.active_jobs ?? "--"}
          tone={data && data.active_jobs > 0 ? "warn" : "neutral"}
        />
        <StatCard
          label="Model spend (24h)"
          value={formatUsd(data?.llm_cost_24h_usd)}
          hint={
            data && data.failed_jobs_24h > 0
              ? `${data.failed_jobs_24h} failed job(s)`
              : undefined
          }
          tone={data && data.failed_jobs_24h > 0 ? "bad" : "neutral"}
        />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Processing"
            description="Background work in flight."
            action={
              <Link
                href="/processing"
                className="text-xs text-accent-300 hover:text-accent-200"
              >
                View all
              </Link>
            }
          />
          <div className="p-4">
            {jobs.loading && !jobs.data ? (
              <LoadingRows rows={3} />
            ) : jobs.data && jobs.data.items.length > 0 ? (
              <ul className="space-y-2">
                {jobs.data.items.map((job) => (
                  <li
                    key={job.id}
                    className="rounded-lg border border-ink-700 bg-ink-850 px-3.5 py-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="truncate text-sm text-ink-100">
                        {job.job_type.replace(/_/g, " ").toLowerCase()}
                      </span>
                      <StatusBadge status={job.status} />
                    </div>
                    <div className="mt-2">
                      <ProgressBar
                        value={job.progress}
                        tone={
                          job.status === "FAILED"
                            ? "bad"
                            : job.status === "COMPLETED"
                              ? "good"
                              : "accent"
                        }
                      />
                    </div>
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-ink-500">
                      <span className="truncate">
                        {job.error_message ?? job.stage ?? "queued"}
                      </span>
                      <span>{formatRelative(job.created_at)}</span>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                title="Nothing processing"
                description="Upload a video and run an analysis to get started."
              />
            )}
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Waiting for you"
            description="Clips the pipeline will not publish without a human."
            action={
              <Link
                href="/clips?status=NEEDS_REVIEW"
                className="text-xs text-accent-300 hover:text-accent-200"
              >
                Review queue
              </Link>
            }
          />
          <div className="p-4">
            {review.loading && !review.data ? (
              <LoadingRows rows={3} />
            ) : review.data && review.data.items.length > 0 ? (
              <ul className="space-y-2">
                {review.data.items.map((clip) => (
                  <li key={clip.id}>
                    <Link
                      href={`/clips/${clip.id}`}
                      className="flex items-center gap-3 rounded-lg border border-ink-700 bg-ink-850 px-3.5 py-3 transition-colors hover:border-ink-600"
                    >
                      <div className="h-12 w-7 shrink-0 overflow-hidden rounded bg-ink-800">
                        {clip.thumbnail_url && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={clip.thumbnail_url}
                            alt=""
                            className="h-full w-full object-cover"
                          />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm text-ink-100">
                          {clip.title ?? clip.hook_text ?? "Untitled clip"}
                        </div>
                        <div className="mt-0.5 text-[11px] text-ink-500">
                          {formatDuration(clip.duration_seconds)} ·{" "}
                          {clip.video_title ?? "source"}
                        </div>
                      </div>
                      <ScoreBadge score={clip.viral_score} />
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                title="Review queue is clear"
                description="Nothing is waiting on a decision."
              />
            )}
          </div>
        </Card>
      </div>

      {data?.top_clip && (
        <Card className="mt-4 px-5 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-xs font-medium uppercase tracking-wide text-ink-400">
                Top performing clip
              </div>
              <Link
                href={`/clips/${data.top_clip.id}`}
                className="mt-1 block truncate text-sm font-medium text-ink-50 hover:text-accent-300"
              >
                {data.top_clip.title ?? "Untitled clip"}
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-ink-400">
                {formatDuration(data.top_clip.duration_seconds)}
              </span>
              <StatusBadge status={data.top_clip.status} />
              <ScoreBadge score={data.top_clip.viral_score} />
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
