"use client";

/** Primitive UI components shared across every page. */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

import { scoreTone } from "@/lib/format";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------- button
type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "success";
type ButtonSize = "sm" | "md";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent-600 text-white hover:bg-accent-500 disabled:bg-accent-600/40",
  secondary:
    "bg-ink-800 text-ink-100 border border-ink-600 hover:bg-ink-700 hover:border-ink-500",
  ghost: "text-ink-300 hover:text-ink-50 hover:bg-ink-800",
  danger: "bg-bad-500/15 text-bad-500 border border-bad-500/30 hover:bg-bad-500/25",
  success:
    "bg-good-500/15 text-good-500 border border-good-500/30 hover:bg-good-500/25",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-60",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        BUTTON_VARIANTS[variant],
        className,
      )}
    >
      {loading && <Spinner size={size === "sm" ? 12 : 14} />}
      {children}
    </button>
  );
}

export function Spinner({ size = 16 }: { size?: number }) {
  return (
    <span
      className="animate-spin-slow inline-block rounded-full border-2 border-current border-t-transparent"
      style={{ width: size, height: size }}
      aria-hidden
    />
  );
}

// ----------------------------------------------------------------------- badge
type BadgeTone =
  | "neutral"
  | "accent"
  | "good"
  | "warn"
  | "bad"
  | "muted";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "bg-ink-800 text-ink-200 border-ink-600",
  accent: "bg-accent-500/15 text-accent-300 border-accent-500/30",
  good: "bg-good-500/15 text-good-500 border-good-500/30",
  warn: "bg-warn-500/15 text-warn-500 border-warn-500/30",
  bad: "bg-bad-500/15 text-bad-500 border-bad-500/30",
  muted: "bg-ink-850 text-ink-400 border-ink-700",
};

export function Badge({
  tone = "neutral",
  children,
  className,
  title,
}: {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Status → tone mappings, kept next to the badge so they stay consistent. */
export const STATUS_TONES: Record<string, BadgeTone> = {
  // videos
  DISCOVERED: "muted",
  PENDING_MEDIA: "warn",
  UPLOADED: "accent",
  ANALYZING: "accent",
  ANALYZED: "good",
  FAILED: "bad",
  ARCHIVED: "muted",
  // clips
  DRAFT: "muted",
  QUEUED: "accent",
  RENDERING: "accent",
  RENDERED: "accent",
  NEEDS_REVIEW: "warn",
  APPROVED: "good",
  REJECTED: "bad",
  PUBLISHED: "good",
  // jobs
  PROCESSING: "accent",
  COMPLETED: "good",
  CANCELLED: "muted",
  // rights
  UNKNOWN: "muted",
  USER_OWNED: "good",
  LICENSED: "good",
  AUTHORIZED: "good",
  USER_UPLOADED: "good",
  NOT_AUTHORIZED: "bad",
  // safety
  SAFE: "good",
  REVIEW: "warn",
  BLOCK: "bad",
  PASS: "good",
  WARN: "warn",
  FAIL: "bad",
};

export function StatusBadge({ status }: { status: string | null | undefined }) {
  if (!status) return <Badge tone="muted">--</Badge>;
  return (
    <Badge tone={STATUS_TONES[status] ?? "neutral"}>
      {status.replace(/_/g, " ")}
    </Badge>
  );
}

export function ScoreBadge({ score }: { score: number | null | undefined }) {
  const tone = scoreTone(score);
  const map = { high: "good", mid: "warn", low: "bad", none: "muted" } as const;
  return (
    <Badge tone={map[tone]}>{score == null ? "--" : score.toFixed(0)}</Badge>
  );
}

// ------------------------------------------------------------------ containers
export function Card({
  children,
  className,
  raised = false,
}: {
  children: ReactNode;
  className?: string;
  raised?: boolean;
}) {
  return (
    <div className={cx(raised ? "surface-raised" : "surface", className)}>
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  description,
  action,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-ink-700 px-5 py-4">
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-ink-50">{title}</h2>
        {description && (
          <p className="mt-1 text-xs text-ink-400">{description}</p>
        )}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink-50">
          {title}
        </h1>
        {description && (
          <p className="mt-1 max-w-2xl text-sm text-ink-400">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const toneClass = {
    neutral: "text-ink-50",
    good: "text-good-500",
    warn: "text-warn-500",
    bad: "text-bad-500",
  }[tone];
  return (
    <Card className="px-4 py-3.5">
      <div className="text-xs font-medium uppercase tracking-wide text-ink-400">
        {label}
      </div>
      <div className={cx("mt-1.5 text-2xl font-semibold tabular-nums", toneClass)}>
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-ink-500">{hint}</div>}
    </Card>
  );
}

// ---------------------------------------------------------------------- states
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-700 px-6 py-14 text-center">
      <h3 className="text-sm font-medium text-ink-200">{title}</h3>
      {description && (
        <p className="mt-1.5 max-w-md text-sm text-ink-500">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
}: {
  error: Error | null;
  onRetry?: () => void;
}) {
  if (!error) return null;
  return (
    <div className="rounded-lg border border-bad-500/30 bg-bad-500/10 px-4 py-3 text-sm text-bad-500">
      <div className="font-medium">Something went wrong</div>
      <div className="mt-0.5 text-bad-500/80">{error.message}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 text-xs font-medium underline underline-offset-2"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function LoadingRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="animate-job h-14 rounded-lg border border-ink-700 bg-ink-850"
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------- inputs
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-ink-300">
        {label}
      </span>
      {children}
      {hint && <span className="mt-1 block text-xs text-ink-500">{hint}</span>}
    </label>
  );
}

const CONTROL =
  "w-full rounded-lg border border-ink-600 bg-ink-850 px-3 py-2 text-sm text-ink-100 " +
  "placeholder:text-ink-500 focus:border-accent-500 focus:outline-none disabled:opacity-60";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(CONTROL, props.className)} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cx(CONTROL, "pr-8", props.className)} />;
}

export function TextArea(
  props: React.TextareaHTMLAttributes<HTMLTextAreaElement>,
) {
  return <textarea {...props} className={cx(CONTROL, "resize-y", props.className)} />;
}

// ------------------------------------------------------------------------ misc
export function ProgressBar({
  value,
  tone = "accent",
}: {
  value: number;
  tone?: "accent" | "good" | "warn" | "bad";
}) {
  const bg = {
    accent: "bg-accent-500",
    good: "bg-good-500",
    warn: "bg-warn-500",
    bad: "bg-bad-500",
  }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
      <div
        className={cx("h-full rounded-full transition-all", bg)}
        style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

/** Horizontal 0-10 bar used for the viral-score dimension breakdown. */
export function DimensionBar({
  label,
  value,
  max = 10,
}: {
  label: string;
  value: number;
  max?: number;
}) {
  const ratio = Math.max(0, Math.min(1, value / max));
  const tone = ratio >= 0.85 ? "bg-good-500" : ratio >= 0.6 ? "bg-accent-500" : "bg-ink-500";
  return (
    <div className="flex items-center gap-3">
      <span className="w-40 shrink-0 text-xs text-ink-400">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
        <div className={cx("h-full rounded-full", tone)} style={{ width: `${ratio * 100}%` }} />
      </div>
      <span className="w-8 shrink-0 text-right text-xs tabular-nums text-ink-200">
        {value.toFixed(1)}
      </span>
    </div>
  );
}

export function Toast({
  message,
  tone = "neutral",
  onDismiss,
}: {
  message: string;
  tone?: "neutral" | "good" | "bad";
  onDismiss?: () => void;
}) {
  const toneClass = {
    neutral: "border-ink-600 bg-ink-800 text-ink-100",
    good: "border-good-500/30 bg-good-500/10 text-good-500",
    bad: "border-bad-500/30 bg-bad-500/10 text-bad-500",
  }[tone];
  return (
    <div
      role="status"
      className={cx(
        "fixed bottom-5 right-5 z-50 flex max-w-sm items-start gap-3 rounded-lg border px-4 py-3 text-sm shadow-lg",
        toneClass,
      )}
    >
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="text-xs opacity-70 hover:opacity-100">
          Dismiss
        </button>
      )}
    </div>
  );
}
