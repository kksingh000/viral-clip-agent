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
  ProgressBar,
  Select,
  StatusBadge,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatCount, formatRelative, formatUsd, titleCase } from "@/lib/format";
import type { Job } from "@/types/api";

export default function ProcessingPage() {
  return (
    <Shell>
      <ProcessingContent />
    </Shell>
  );
}

function ProcessingContent() {
  const [status, setStatus] = useState("");
  const jobs = useQuery(
    () => api.jobs.list({ status: status || undefined, limit: 60 }),
    [status],
    { pollMs: 4000 },
  );

  return (
    <>
      <PageHeader
        title="Processing"
        description="Every unit of background work, with its cost and its failure reason."
      />

      <div className="mb-4 flex items-center gap-2">
        <Select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="max-w-[12rem]"
        >
          <option value="">All statuses</option>
          {["QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"].map(
            (value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ),
          )}
        </Select>
        <span className="text-xs text-ink-500">
          {jobs.data?.total ?? 0} job{jobs.data?.total === 1 ? "" : "s"}
        </span>
      </div>

      <ErrorState error={jobs.error} onRetry={jobs.refetch} />

      {jobs.loading && !jobs.data ? (
        <LoadingRows rows={6} />
      ) : jobs.data && jobs.data.items.length > 0 ? (
        <div className="space-y-2">
          {jobs.data.items.map((job) => (
            <JobRow key={job.id} job={job} onChanged={jobs.refetch} />
          ))}
        </div>
      ) : (
        <EmptyState title="No jobs" description="Nothing has been queued yet." />
      )}
    </>
  );
}

function JobRow({ job, onChanged }: { job: Job; onChanged: () => void }) {
  const cancel = useMutation(() => api.jobs.cancel(job.id));
  const cancellable = job.status === "QUEUED" || job.status === "PROCESSING";

  return (
    <Card className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <StatusBadge status={job.status} />
        <span className="text-sm text-ink-100">{titleCase(job.job_type)}</span>
        {job.retry_count > 0 && (
          <Badge tone="warn">
            retry {job.retry_count}/{job.max_retries}
          </Badge>
        )}
        {job.video_id && (
          <Link
            href={`/videos/${job.video_id}`}
            className="text-xs text-accent-300 hover:text-accent-200"
          >
            video
          </Link>
        )}
        {job.clip_id && (
          <Link
            href={`/clips/${job.clip_id}`}
            className="text-xs text-accent-300 hover:text-accent-200"
          >
            clip
          </Link>
        )}

        <div className="ml-auto flex items-center gap-3 text-[11px] text-ink-500">
          {job.llm_calls > 0 && (
            <span title="Model calls and tokens used by this job">
              {job.llm_calls} calls ·{" "}
              {formatCount(job.llm_input_tokens + job.llm_output_tokens)} tokens ·{" "}
              {formatUsd(job.estimated_cost_usd)}
            </span>
          )}
          {job.duration_seconds != null && (
            <span>{job.duration_seconds.toFixed(1)}s</span>
          )}
          <span>{formatRelative(job.created_at)}</span>
          {cancellable && (
            <Button
              size="sm"
              variant="ghost"
              loading={cancel.pending}
              onClick={async () => {
                await cancel.run();
                onChanged();
              }}
            >
              Cancel
            </Button>
          )}
        </div>
      </div>

      {job.status === "PROCESSING" && (
        <div className="mt-2">
          <ProgressBar value={job.progress} />
          <div className="mt-1 text-[11px] text-ink-500">
            {job.stage ?? "working"} · {Math.round(job.progress * 100)}%
          </div>
        </div>
      )}

      {job.error_message && (
        <div className="mt-2 rounded border border-bad-500/30 bg-bad-500/10 px-2.5 py-1.5 text-[11px] text-bad-500">
          <span className="font-mono">{job.error_code}</span> — {job.error_message}
        </div>
      )}

      {job.status === "COMPLETED" && Object.keys(job.result ?? {}).length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[11px] text-ink-500 hover:text-ink-300">
            Result
          </summary>
          <pre className="mt-1.5 overflow-x-auto rounded border border-ink-700 bg-ink-850 p-2.5 font-mono text-[10px] leading-relaxed text-ink-400">
            {JSON.stringify(job.result, null, 2)}
          </pre>
        </details>
      )}
    </Card>
  );
}
