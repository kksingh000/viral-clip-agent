"use client";

import Link from "next/link";
import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  LoadingRows,
  PageHeader,
  StatCard,
  Toast,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatCount, formatPercent, formatScore } from "@/lib/format";
import type { AnalyticsPoint } from "@/types/api";

export default function AnalyticsPage() {
  return (
    <Shell>
      <AnalyticsContent />
    </Shell>
  );
}

function AnalyticsContent() {
  const [toast, setToast] = useState<string | null>(null);
  const analytics = useQuery(() => api.analytics.performance(), []);
  const calibrate = useMutation(() => api.analytics.calibrate());

  const data = analytics.data;

  return (
    <>
      <PageHeader
        title="Analytics"
        description="Predicted viral score against what actually happened, and the loop that closes the gap."
        actions={
          <Button
            variant="primary"
            loading={calibrate.pending}
            onClick={async () => {
              const result = await calibrate.run();
              if (result) {
                setToast(
                  result.status === "ok"
                    ? `Weights refitted on ${result.sample_size} clips (MAE ${Number(
                        result.baseline_mae,
                      ).toFixed(1)} → ${Number(result.calibrated_mae).toFixed(1)}).`
                    : `Not enough data yet: ${result.sample_size ?? 0} of ${
                        result.required ?? 25
                      } published clips.`,
                );
                analytics.refetch();
              }
            }}
          >
            Recalibrate scoring
          </Button>
        }
      />

      <ErrorState error={analytics.error} onRetry={analytics.refetch} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Clips published" value={data?.clips_published ?? "--"} />
        <StatCard label="Total views" value={formatCount(data?.total_views)} />
        <StatCard label="Total likes" value={formatCount(data?.total_likes)} />
        <StatCard label="Total comments" value={formatCount(data?.total_comments)} />
        <StatCard
          label="Avg completion"
          value={formatPercent(data?.average_completion_rate)}
        />
        <StatCard
          label="Score correlation"
          value={
            data?.score_correlation != null
              ? data.score_correlation.toFixed(2)
              : "--"
          }
          hint="Predicted score vs realised percentile"
          tone={
            data?.score_correlation == null
              ? "neutral"
              : data.score_correlation > 0.4
                ? "good"
                : data.score_correlation > 0.1
                  ? "warn"
                  : "bad"
          }
        />
        <StatCard
          label="Calibration sample"
          value={data?.calibration_sample_size ?? 0}
          hint="25 needed to refit"
        />
      </div>

      <Card className="mt-6">
        <CardHeader
          title="Prediction vs performance"
          description="Each clip's predicted viral score next to where it actually landed among your clips."
        />
        <div className="p-4">
          {analytics.loading && !data ? (
            <LoadingRows rows={4} />
          ) : data && data.points.length > 0 ? (
            <>
              <ScatterPlot points={data.points} />
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-ink-500">
                    <tr className="border-b border-ink-800">
                      <th className="py-2 font-medium">Clip</th>
                      <th className="py-2 font-medium">Predicted</th>
                      <th className="py-2 font-medium">Views</th>
                      <th className="py-2 font-medium">Percentile</th>
                      <th className="py-2 font-medium">Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.points.map((point) => (
                      <tr key={point.clip_id} className="border-b border-ink-850">
                        <td className="max-w-xs truncate py-2">
                          <Link
                            href={`/clips/${point.clip_id}`}
                            className="text-ink-200 hover:text-accent-300"
                          >
                            {point.title ?? "Untitled"}
                          </Link>
                        </td>
                        <td className="py-2 tabular-nums text-ink-300">
                          {formatScore(point.predicted_score)}
                        </td>
                        <td className="py-2 tabular-nums text-ink-300">
                          {formatCount(point.views)}
                        </td>
                        <td className="py-2 tabular-nums text-ink-300">
                          {formatScore(point.performance_percentile)}
                        </td>
                        <td
                          className={
                            point.prediction_error == null
                              ? "py-2 text-ink-500"
                              : Math.abs(point.prediction_error) > 25
                                ? "py-2 tabular-nums text-bad-500"
                                : "py-2 tabular-nums text-ink-300"
                          }
                        >
                          {point.prediction_error == null
                            ? "--"
                            : point.prediction_error.toFixed(0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <EmptyState
              title="No published clips measured yet"
              description="Connect a platform account and publish a clip; performance is synced automatically and fed back into scoring."
            />
          )}
        </div>
      </Card>

      {toast && <Toast message={toast} onDismiss={() => setToast(null)} />}
    </>
  );
}

/**
 * Predicted score (x) against realised percentile (y). Points near the
 * diagonal mean the model is calibrated; points far above it mean the score
 * was optimistic.
 */
function ScatterPlot({ points }: { points: AnalyticsPoint[] }) {
  const usable = points.filter(
    (p) => p.predicted_score != null && p.performance_percentile != null,
  );
  if (usable.length < 2) return null;

  const size = 260;
  const pad = 28;
  const scale = (value: number) => pad + (value / 100) * (size - pad * 2);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="h-64 w-full max-w-sm"
      role="img"
      aria-label="Predicted score against realised performance percentile"
    >
      <rect
        x={pad}
        y={pad}
        width={size - pad * 2}
        height={size - pad * 2}
        fill="none"
        stroke="var(--color-ink-700)"
      />
      <line
        x1={pad}
        y1={size - pad}
        x2={size - pad}
        y2={pad}
        stroke="var(--color-ink-600)"
        strokeDasharray="3 3"
      />
      {usable.map((point) => (
        <circle
          key={point.clip_id}
          cx={scale(point.predicted_score!)}
          cy={size - scale(point.performance_percentile!)}
          r={4}
          fill="var(--color-accent-400)"
          fillOpacity={0.8}
        >
          <title>
            {point.title ?? "Untitled"} — predicted{" "}
            {point.predicted_score!.toFixed(0)}, actual{" "}
            {point.performance_percentile!.toFixed(0)}
          </title>
        </circle>
      ))}
      <text x={size / 2} y={size - 6} textAnchor="middle" fontSize="9" fill="var(--color-ink-500)">
        predicted viral score
      </text>
      <text
        x={10}
        y={size / 2}
        textAnchor="middle"
        fontSize="9"
        fill="var(--color-ink-500)"
        transform={`rotate(-90 10 ${size / 2})`}
      >
        actual percentile
      </text>
    </svg>
  );
}
