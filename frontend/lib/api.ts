/**
 * Typed API client.
 *
 * One place that knows how to talk to the backend: base URL, auth header,
 * error shape. Every error surfaces as an ApiError carrying the backend's
 * `error_code`, so the UI can respond to specific conditions (an expired
 * session, a rights refusal) rather than showing a generic failure.
 */

import type {
  Analytics,
  Candidate,
  Clip,
  ClipDetail,
  Dashboard,
  Health,
  Job,
  JobAccepted,
  Page,
  PlatformAccount,
  Scene,
  Settings,
  Topic,
  TokenResponse,
  Transcript,
  TrendingVideo,
  User,
  Video,
} from "@/types/api";

declare global {
  interface Window {
    /** Injected per request by the root layout; see app/layout.tsx. */
    __API_URL__?: string;
  }
}

/**
 * Where the API lives.
 *
 * Runtime value first: `NEXT_PUBLIC_*` is inlined at build time, so an image
 * built once cannot be pointed at a different API later. The server injects
 * the current value into every page, which is what makes the same image
 * deployable to any host.
 */
function withScheme(value: string): string {
  const trimmed = value.trim().replace(/\/$/, "");
  if (!trimmed) return "";
  if (trimmed.includes("://")) return trimmed;
  // Hosting platforms expose a service address as a bare hostname. Fetching
  // "api.example.com/api/v1/..." would resolve against the current page
  // instead of the API, which fails in a thoroughly confusing way.
  const scheme = /^(localhost|127\.0\.0\.1)/.test(trimmed) ? "http" : "https";
  return `${scheme}://${trimmed}`;
}

function resolveApiBase(): string {
  if (typeof window !== "undefined" && window.__API_URL__) {
    return withScheme(window.__API_URL__);
  }
  return (
    withScheme(process.env.NEXT_PUBLIC_API_URL ?? "") || "http://localhost:8000"
  );
}

export const API_BASE = resolveApiBase();

const API = `${API_BASE}/api/v1`;
const TOKEN_KEY = "vca.access_token";
const REFRESH_KEY = "vca.refresh_token";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(
    status: number,
    code: string,
    message: string,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  get isAuth(): boolean {
    return this.status === 401 || this.code === "unauthenticated";
  }

  get isRights(): boolean {
    return this.code === "content_not_authorized";
  }
}

// --------------------------------------------------------------------- tokens
export const tokens = {
  get access(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(TOKEN_KEY);
  },
  get refresh(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(REFRESH_KEY);
  },
  set(value: TokenResponse) {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(TOKEN_KEY, value.access_token);
    window.localStorage.setItem(REFRESH_KEY, value.refresh_token);
  },
  clear() {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_KEY);
  },
};

// -------------------------------------------------------------------- request
interface RequestOptions {
  method?: string;
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
  /** Skip the automatic refresh-and-retry (used by the refresh call itself). */
  raw?: boolean;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const url = new URL(`${API}${path}`);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

let refreshInFlight: Promise<boolean> | null = null;

async function refreshSession(): Promise<boolean> {
  // Collapse concurrent 401s into a single refresh attempt.
  if (refreshInFlight) return refreshInFlight;
  const token = tokens.refresh;
  if (!token) return false;

  refreshInFlight = (async () => {
    try {
      const response = await fetch(`${API}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: token }),
      });
      if (!response.ok) return false;
      tokens.set((await response.json()) as TokenResponse);
      return true;
    } catch {
      return false;
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

async function toError(response: Response): Promise<ApiError> {
  let code = `http_${response.status}`;
  let message = response.statusText || "Request failed";
  let details: Record<string, unknown> = {};
  try {
    const body = await response.json();
    code = body.error_code ?? code;
    message = body.error_message ?? message;
    details = body.details ?? {};
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(response.status, code, message, details);
}

export async function request<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, query, signal, raw } = options;
  const headers: Record<string, string> = {};
  const token = tokens.access;
  if (token) headers.Authorization = `Bearer ${token}`;

  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    payload = body;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  const send = () =>
    fetch(buildUrl(path, query), { method, headers, body: payload, signal });

  let response = await send();

  if (response.status === 401 && !raw && tokens.refresh) {
    if (await refreshSession()) {
      const retryToken = tokens.access;
      if (retryToken) headers.Authorization = `Bearer ${retryToken}`;
      response = await send();
    }
  }

  if (!response.ok) throw await toError(response);
  if (response.status === 204) return undefined as T;

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return (await response.text()) as T;
  }
  return (await response.json()) as T;
}

// ----------------------------------------------------------------- endpoints
export const api = {
  health: () => fetch(`${API_BASE}/health`).then((r) => r.json() as Promise<Health>),

  auth: {
    register: (email: string, password: string, fullName?: string) =>
      request<TokenResponse>("/auth/register", {
        method: "POST",
        body: { email, password, full_name: fullName },
        raw: true,
      }),
    login: (email: string, password: string) =>
      request<TokenResponse>("/auth/login", {
        method: "POST",
        body: { email, password },
        raw: true,
      }),
    me: () => request<User>("/auth/me"),
  },

  videos: {
    list: (query?: Record<string, string | number | undefined>) =>
      request<Page<Video>>("/videos", { query }),
    get: (id: string) => request<Video>(`/videos/${id}`),
    create: (body: Record<string, unknown>) =>
      request<Video>("/videos", { method: "POST", body }),
    update: (id: string, body: Record<string, unknown>) =>
      request<Video>(`/videos/${id}`, { method: "PATCH", body }),
    remove: (id: string) =>
      request<{ message: string }>(`/videos/${id}`, { method: "DELETE" }),
    setAuthorization: (id: string, body: Record<string, unknown>) =>
      request<Video>(`/videos/${id}/authorization`, { method: "PUT", body }),
    upload: (id: string, file: File) => {
      const form = new FormData();
      form.append("file", file);
      return request<Video>(`/videos/${id}/upload`, { method: "POST", body: form });
    },
    analyze: (id: string, force = false, autoGenerate = false) =>
      request<JobAccepted>(`/videos/${id}/analyze`, {
        method: "POST",
        body: { force },
        query: { auto_generate: autoGenerate },
      }),
    transcript: (id: string, includeWords = false) =>
      request<Transcript>(`/videos/${id}/transcript`, {
        query: { include_words: includeWords },
      }),
    scenes: (id: string) => request<Scene[]>(`/videos/${id}/scenes`),
    candidates: (id: string, includeDiscarded = false) =>
      request<Candidate[]>(`/videos/${id}/candidates`, {
        query: { include_discarded: includeDiscarded },
      }),
  },

  clips: {
    list: (query?: Record<string, string | number | undefined>) =>
      request<Page<Clip>>("/clips", { query }),
    get: (id: string) => request<ClipDetail>(`/clips/${id}`),
    generate: (body: Record<string, unknown>) =>
      request<JobAccepted>("/clips/generate", { method: "POST", body }),
    manual: (body: Record<string, unknown>) =>
      request<JobAccepted>("/clips/manual", { method: "POST", body }),
    update: (id: string, body: Record<string, unknown>) =>
      request<Clip>(`/clips/${id}`, { method: "PATCH", body }),
    regenerate: (id: string, body: Record<string, unknown>) =>
      request<JobAccepted>(`/clips/${id}/regenerate`, { method: "POST", body }),
    variants: (id: string) =>
      request<JobAccepted>(`/clips/${id}/variants`, { method: "POST" }),
    approve: (id: string, note?: string) =>
      request<Clip>(`/clips/${id}/approve`, { method: "POST", body: { note } }),
    reject: (id: string, note?: string) =>
      request<Clip>(`/clips/${id}/reject`, { method: "POST", body: { note } }),
    remove: (id: string) =>
      request<{ message: string }>(`/clips/${id}`, { method: "DELETE" }),
    downloadUrl: (id: string) => `${API}/clips/${id}/download`,
  },

  jobs: {
    list: (query?: Record<string, string | number | undefined>) =>
      request<Page<Job>>("/jobs", { query }),
    get: (id: string) => request<Job>(`/jobs/${id}`),
    cancel: (id: string) =>
      request<{ message: string }>(`/jobs/${id}/cancel`, { method: "POST" }),
  },

  trending: {
    list: (query?: Record<string, string | number | undefined>) =>
      request<Page<TrendingVideo>>("/trending", { query }),
    discover: (body: Record<string, unknown>) =>
      request<JobAccepted>("/trending/discover", { method: "POST", body }),
    topics: (query?: Record<string, string | number | undefined>) =>
      request<Page<Topic>>("/topics", { query }),
  },

  settings: {
    get: () => request<Settings>("/settings"),
    update: (body: Record<string, unknown>) =>
      request<Settings>("/settings", { method: "PUT", body }),
    defaults: () =>
      request<{
        scoring_weights: Record<string, number>;
        trend_weights: Record<string, number>;
      }>("/settings/defaults"),
  },

  analytics: {
    dashboard: () => request<Dashboard>("/dashboard"),
    performance: () => request<Analytics>("/analytics"),
    calibrate: () =>
      request<Record<string, unknown>>("/analytics/calibrate", { method: "POST" }),
  },

  integrations: {
    accounts: () => request<PlatformAccount[]>("/integrations/accounts"),
    connect: (body: Record<string, unknown>) =>
      request<PlatformAccount>("/integrations/accounts", { method: "POST", body }),
    disconnect: (id: string) =>
      request<{ message: string }>(`/integrations/accounts/${id}`, {
        method: "DELETE",
      }),
    publish: (clipId: string, body: Record<string, unknown>) =>
      request<JobAccepted>(`/integrations/publish/${clipId}`, {
        method: "POST",
        body,
      }),
  },
};
