"use client";

import Link from "next/link";
import { useState } from "react";

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
  Toast,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { ApiError, api } from "@/lib/api";
import { formatCount, formatDuration, formatRelative } from "@/lib/format";

const REGIONS = ["US", "GB", "IN", "CA", "AU", "DE", "BR", "JP"];

export default function TrendingPage() {
  return (
    <Shell>
      <TrendingContent />
    </Shell>
  );
}

function TrendingContent() {
  const [region, setRegion] = useState("US");
  const [query, setQuery] = useState("");
  const [toast, setToast] = useState<{ message: string; tone: "good" | "bad" } | null>(
    null,
  );

  const trending = useQuery(() => api.trending.list({ limit: 40 }), []);
  const topics = useQuery(() => api.trending.topics({ limit: 20 }), []);
  const discover = useMutation(() =>
    api.trending.discover({ region, query: query || undefined, max_results: 50 }),
  );

  return (
    <>
      <PageHeader
        title="Trending"
        description="Momentum, not totals. A 100k-view video gaining 50k this hour outranks a 10M-view video that has stopped moving."
      />

      <div className="mb-4 rounded-lg border border-ink-700 bg-ink-850 px-4 py-3 text-xs text-ink-400">
        Discovery reads <strong className="text-ink-200">public metadata only</strong>{" "}
        through the official YouTube Data API. It never downloads media. Every
        discovered video is recorded with{" "}
        <code className="rounded bg-ink-800 px-1 py-0.5 text-ink-300">
          authorization_status = UNKNOWN
        </code>{" "}
        and cannot be processed or published until you establish rights.
      </div>

      <Card className="mb-6 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Region">
            <Select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="w-28"
            >
              {REGIONS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Search (optional)">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Leave blank for most popular"
              className="w-64"
            />
          </Field>
          <Button
            variant="primary"
            loading={discover.pending}
            onClick={async () => {
              const result = await discover.run();
              if (result) {
                setToast({ message: "Discovery run queued.", tone: "good" });
                setTimeout(() => trending.refetch(), 4000);
              } else if (discover.error instanceof ApiError) {
                setToast({ message: discover.error.message, tone: "bad" });
              }
            }}
          >
            Run discovery
          </Button>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <div>
          <ErrorState error={trending.error} onRetry={trending.refetch} />
          {trending.loading && !trending.data ? (
            <LoadingRows rows={5} />
          ) : trending.data && trending.data.items.length > 0 ? (
            <div className="space-y-2">
              {trending.data.items.map((video) => (
                <Card key={video.id} className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <div className="h-12 w-20 shrink-0 overflow-hidden rounded bg-ink-800">
                      {video.thumbnail_url && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={video.thumbnail_url}
                          alt=""
                          className="h-full w-full object-cover"
                        />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <Link
                        href={`/videos/${video.id}`}
                        className="line-clamp-1 text-sm font-medium text-ink-50 hover:text-accent-300"
                      >
                        {video.title}
                      </Link>
                      <div className="mt-0.5 flex flex-wrap gap-x-3 text-[11px] text-ink-500">
                        <span>{video.creator_name}</span>
                        <span>{formatCount(video.view_count)} views</span>
                        {video.view_velocity != null && (
                          <span title="Views per hour">
                            {formatCount(Math.round(video.view_velocity))}/h
                          </span>
                        )}
                        {video.engagement_rate != null && (
                          <span title="(likes + comments) / views">
                            {(video.engagement_rate * 100).toFixed(1)}% engagement
                          </span>
                        )}
                        <span>{formatDuration(video.duration_seconds)}</span>
                        <span>{formatRelative(video.published_at)}</span>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {video.growth_acceleration != null &&
                        video.growth_acceleration > 0.15 && (
                          <Badge tone="good" title="Accelerating relative to its lifetime average">
                            accelerating
                          </Badge>
                        )}
                      <Badge tone="muted">rights unknown</Badge>
                      <ScoreBadge score={video.trend_score} />
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <EmptyState
              title="Nothing discovered yet"
              description="Run a discovery pass. A YOUTUBE_API_KEY must be configured."
            />
          )}
        </div>

        <Card className="h-fit">
          <CardHeader
            title="Emerging topics"
            description="Subjects repeating across the batch."
          />
          <div className="p-4">
            {topics.loading && !topics.data ? (
              <LoadingRows rows={3} />
            ) : topics.data && topics.data.items.length > 0 ? (
              <ul className="space-y-2">
                {topics.data.items.map((topic) => (
                  <li
                    key={topic.id}
                    className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm text-ink-100">
                        {topic.label}
                      </span>
                      <ScoreBadge score={topic.trend_score} />
                    </div>
                    <div className="mt-1 flex gap-3 text-[11px] text-ink-500">
                      <span>{topic.video_count} videos</span>
                      <span
                        className={
                          topic.growth_rate > 0 ? "text-good-500" : "text-ink-500"
                        }
                      >
                        {topic.growth_rate > 0 ? "+" : ""}
                        {(topic.growth_rate * 100).toFixed(0)}%
                      </span>
                      <span>{formatCount(topic.total_views)} views</span>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-ink-500">
                Topics appear after a discovery run.
              </p>
            )}
          </div>
        </Card>
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
