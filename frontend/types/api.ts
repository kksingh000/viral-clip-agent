/**
 * Types mirroring the backend's OpenAPI schema.
 *
 * Hand-written rather than generated so the surface stays small and readable;
 * regenerate from /openapi.json if the API grows significantly.
 */

export type AuthorizationStatus =
  | "UNKNOWN"
  | "USER_OWNED"
  | "LICENSED"
  | "AUTHORIZED"
  | "USER_UPLOADED"
  | "NOT_AUTHORIZED";

export type VideoStatus =
  | "DISCOVERED"
  | "PENDING_MEDIA"
  | "UPLOADED"
  | "ANALYZING"
  | "ANALYZED"
  | "FAILED"
  | "ARCHIVED";

export type VideoSource =
  | "UPLOAD"
  | "LOCAL_PATH"
  | "CLOUD_STORAGE"
  | "YOUTUBE_OWNED"
  | "DISCOVERY"
  | "API_INTEGRATION";

export type ClipStatus =
  | "DRAFT"
  | "QUEUED"
  | "RENDERING"
  | "RENDERED"
  | "NEEDS_REVIEW"
  | "APPROVED"
  | "REJECTED"
  | "PUBLISHED"
  | "FAILED";

export type JobStatus =
  | "QUEUED"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export type CropMode = "SMART" | "CENTER" | "BLUR_PAD" | "SPLIT_SPEAKERS";

export type CaptionStyle =
  | "NONE"
  | "NORMAL"
  | "BOLD"
  | "KARAOKE"
  | "WORD_HIGHLIGHT"
  | "MINIMAL"
  | "PODCAST";

export type CaptionPosition =
  | "TOP"
  | "UPPER_THIRD"
  | "CENTER"
  | "LOWER_THIRD"
  | "BOTTOM";

export type SafetyVerdict = "SAFE" | "REVIEW" | "BLOCK";
export type QualityStatus = "PASS" | "WARN" | "FAIL";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface Video {
  id: string;
  title: string;
  description: string | null;
  source: VideoSource;
  source_url: string | null;
  external_id: string | null;
  creator_name: string | null;
  category: string | null;
  language: string | null;
  tags: string[];
  thumbnail_url: string | null;
  published_at: string | null;
  authorization_status: AuthorizationStatus;
  authorization_note: string | null;
  authorized_at: string | null;
  status: VideoStatus;
  status_detail: string | null;
  analyzed_at: string | null;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  audio_available: boolean;
  file_size_bytes: number | null;
  view_count: number | null;
  like_count: number | null;
  comment_count: number | null;
  trend_score: number | null;
  trend_breakdown: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  has_media: boolean;
  candidate_count: number;
  clip_count: number;
  media_url: string | null;
}

export interface TrendingVideo extends Video {
  view_velocity: number | null;
  engagement_rate: number | null;
  growth_acceleration: number | null;
}

export interface ScoreBreakdown {
  final_score: number;
  base_score: number;
  dimensions: Record<string, number>;
  contributions: Record<string, number>;
  penalties: Record<string, number>;
  adjustments?: string[];
}

export interface Candidate {
  id: string;
  video_id: string;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  context_lead_in: number;
  hook: string | null;
  summary: string | null;
  reason: string | null;
  transcript_excerpt: string | null;
  viral_score: number;
  score_breakdown: ScoreBreakdown;
  status: string;
  rank: number | null;
  model_name: string | null;
  created_at: string;
}

export interface TranscriptSegment {
  id: string;
  index: number;
  start_time: number;
  end_time: number;
  text: string;
  speaker: string | null;
  prefilter_score: number | null;
  words: { word: string; start: number; end: number }[];
  signals: Record<string, unknown>;
}

export interface Transcript {
  id: string;
  video_id: string;
  provider: string;
  model: string | null;
  language: string | null;
  duration_seconds: number | null;
  word_count: number;
  text: string;
  is_synthetic: boolean;
  segments: TranscriptSegment[];
}

export interface Scene {
  id: string;
  index: number;
  start_time: number;
  end_time: number;
  change_score: number | null;
}

export interface HookAlternative {
  text: string;
  rationale?: string;
  confidence?: number;
}

export interface Clip {
  id: string;
  video_id: string;
  candidate_id: string | null;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  crop_mode: CropMode;
  caption_style: CaptionStyle;
  caption_position: CaptionPosition;
  title: string | null;
  hook_text: string | null;
  hook_alternatives: HookAlternative[];
  description: string | null;
  instagram_caption: string | null;
  hashtags: string[];
  keywords: string[];
  transcript_text: string | null;
  status: ClipStatus;
  status_detail: string | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  file_size_bytes: number | null;
  viral_score: number | null;
  quality_score: number | null;
  safety_verdict: string | null;
  regeneration_count: number;
  auto_approved: boolean;
  approved_at: string | null;
  rejected_at: string | null;
  review_note: string | null;
  publish_status: string;
  created_at: string;
  updated_at: string;
  media_url: string | null;
  thumbnail_url: string | null;
  subtitle_url: string | null;
  video_title: string | null;
}

export interface QualityIssue {
  check: string;
  severity: "INFO" | "MINOR" | "MAJOR" | "CRITICAL";
  message: string;
  auto_fixable: boolean;
  fix_action: string | null;
}

export interface QualityCheck {
  id: string;
  attempt: number;
  status: QualityStatus;
  score: number;
  threshold: number;
  issues: QualityIssue[];
  recommendations: string[];
  technical: Record<string, unknown>;
  created_at: string;
}

export interface SafetyCheck {
  id: string;
  verdict: SafetyVerdict;
  category_scores: Record<string, number>;
  flagged_categories: string[];
  rationale: string | null;
  created_at: string;
}

export interface ClipVariant {
  id: string;
  label: string;
  description: string | null;
  start_time: number;
  end_time: number;
  crop_mode: CropMode;
  caption_style: CaptionStyle;
  status: ClipStatus;
  is_primary: boolean;
  media_url: string | null;
}

export interface ClipDetail extends Clip {
  captions: { start: number; end: number; text: string }[];
  crop_keyframes: { t: number; x: number; y: number }[];
  render_config: Record<string, unknown>;
  variants: ClipVariant[];
  quality_checks: QualityCheck[];
  safety_checks: SafetyCheck[];
}

export interface Job {
  id: string;
  job_type: string;
  status: JobStatus;
  progress: number;
  stage: string | null;
  video_id: string | null;
  clip_id: string | null;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  error_code: string | null;
  error_message: string | null;
  retry_count: number;
  max_retries: number;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  llm_calls: number;
  llm_input_tokens: number;
  llm_output_tokens: number;
  estimated_cost_usd: number;
  created_at: string;
}

export interface JobAccepted {
  job_id: string;
  status: JobStatus;
  message: string;
}

export interface Topic {
  id: string;
  slug: string;
  label: string;
  description: string | null;
  kind: string;
  keywords: string[];
  trend_score: number;
  growth_rate: number;
  video_count: number;
  total_views: number;
  region: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
}

export interface Dashboard {
  videos_discovered_today: number;
  videos_analyzed: number;
  videos_pending: number;
  clips_generated: number;
  clips_approved: number;
  clips_awaiting_review: number;
  average_viral_score: number | null;
  top_clip: {
    id: string;
    title: string | null;
    viral_score: number | null;
    duration_seconds: number;
    status: string;
  } | null;
  active_jobs: number;
  failed_jobs_24h: number;
  llm_cost_24h_usd: number;
}

export interface Settings {
  min_viral_score: number;
  auto_approve_score: number;
  reject_below_score: number;
  clips_per_video: number;
  min_clip_seconds: number;
  max_clip_seconds: number;
  crop_mode: CropMode;
  caption_style: CaptionStyle;
  caption_position: CaptionPosition;
  caption_overrides: Record<string, unknown>;
  output_width: number;
  output_height: number;
  output_fps: number | null;
  scoring_weights: Record<string, number>;
  trend_weights: Record<string, number>;
  preferred_categories: string[];
  preferred_languages: string[];
  trend_regions: string[];
  quality_threshold: number;
  max_regeneration_attempts: number;
  block_on_safety_review: boolean;
}

export interface AnalyticsPoint {
  clip_id: string;
  title: string | null;
  predicted_score: number | null;
  views: number | null;
  performance_percentile: number | null;
  prediction_error: number | null;
}

export interface Analytics {
  clips_published: number;
  total_views: number;
  total_likes: number;
  total_comments: number;
  average_completion_rate: number | null;
  score_correlation: number | null;
  calibration_sample_size: number;
  points: AnalyticsPoint[];
}

export interface PlatformAccount {
  id: string;
  platform: string;
  external_account_id: string;
  display_name: string | null;
  scopes: string[];
  is_active: boolean;
  connected_at: string | null;
  token_expires_at: string | null;
  last_error: string | null;
}

export interface HealthComponent {
  name: string;
  status: string;
  detail: string | null;
  latency_ms: number | null;
}

export interface Health {
  status: string;
  version: string;
  environment: string;
  components: HealthComponent[];
}
