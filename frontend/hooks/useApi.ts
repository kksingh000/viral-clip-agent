"use client";

/**
 * Minimal data-fetching hooks.
 *
 * Deliberately not a data-fetching library: the dashboard needs "fetch, show a
 * spinner, poll while work is running, refetch on demand", and that is about
 * forty lines. Adding SWR or React Query here would be more configuration than
 * code.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";
import type { Job } from "@/types/api";

interface QueryState<T> {
  data: T | null;
  error: ApiError | Error | null;
  loading: boolean;
  refetch: () => void;
}

export function useQuery<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  options: { enabled?: boolean; pollMs?: number } = {},
): QueryState<T> {
  const { enabled = true, pollMs } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [nonce, setNonce] = useState(0);

  // Keep the latest fetcher without making it a dependency, so callers can
  // pass an inline closure without causing a fetch loop.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refetch = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);

    const run = async () => {
      try {
        const result = await fetcherRef.current();
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err as Error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void run();
    if (!pollMs) return () => {
      cancelled = true;
    };

    const timer = window.setInterval(run, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pollMs, nonce, ...deps]);

  return { data, error, loading, refetch };
}

interface MutationState<TArgs extends unknown[], TResult> {
  run: (...args: TArgs) => Promise<TResult | null>;
  pending: boolean;
  error: ApiError | Error | null;
  reset: () => void;
}

export function useMutation<TArgs extends unknown[], TResult>(
  action: (...args: TArgs) => Promise<TResult>,
): MutationState<TArgs, TResult> {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  const actionRef = useRef(action);
  actionRef.current = action;

  const run = useCallback(async (...args: TArgs) => {
    setPending(true);
    setError(null);
    try {
      return await actionRef.current(...args);
    } catch (err) {
      setError(err as Error);
      return null;
    } finally {
      setPending(false);
    }
  }, []);

  const reset = useCallback(() => setError(null), []);
  return { run, pending, error, reset };
}

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

/**
 * Follow a background job until it reaches a terminal state.
 *
 * Polling stops as soon as the job settles, so an idle dashboard makes no
 * requests.
 */
export function useJobWatcher(
  jobId: string | null,
  fetchJob: (id: string) => Promise<Job>,
  onSettled?: (job: Job) => void,
): { job: Job | null; active: boolean } {
  const [job, setJob] = useState<Job | null>(null);
  const settledRef = useRef(false);
  const callbackRef = useRef(onSettled);
  callbackRef.current = onSettled;

  useEffect(() => {
    setJob(null);
    settledRef.current = false;
    if (!jobId) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const next = await fetchJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (TERMINAL.has(next.status) && !settledRef.current) {
          settledRef.current = true;
          window.clearInterval(timer);
          callbackRef.current?.(next);
        }
      } catch {
        /* transient; the next tick retries */
      }
    };

    void tick();
    const timer = window.setInterval(tick, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  return { job, active: Boolean(job && !TERMINAL.has(job.status)) };
}
