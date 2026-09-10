# Viral Clip Agent

An AI content-repurposing platform: discover what is gaining momentum, find the
smallest **self-contained** moments inside authorized long-form video, and turn
them into publish-ready 9:16 shorts — with speaker-following framing, animated
captions, generated packaging, automated QC, a safety review and a human
approval step.

It is not a video cutter with an LLM bolted on. Deterministic code does
everything measurable (ffmpeg, geometry, timing, scoring arithmetic, quality
probing); a model is consulted only where meaning is genuinely required, and
every model call is schema-constrained, validated, repaired, retried, and
backed by a deterministic fallback.

---

## Contents

- [Rights model — read this first](#rights-model--read-this-first)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [How a clip gets made](#how-a-clip-gets-made)
- [API](#api)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Deploying](#deploying)
- [Operations](#operations)
- [Current status](#current-status)

---

## Rights model — read this first

**Discovering a URL is not permission to download or republish it.** The
system keeps the three concerns separate and enforces the separation in code,
not just in the UI:

| Concern | What it does | What it never does |
|---|---|---|
| **Trend discovery** | Reads public metadata through the official YouTube Data API | Download media, imply republishing rights |
| **Content authorization** | Records a rights position asserted by the account holder | Assert rights on the user's behalf |
| **Video processing** | Transcribes, analyses, cuts and renders | Touch anything not authorized |

Every video carries an `authorization_status`:

```
UNKNOWN          metadata only — cannot be processed or published
USER_OWNED       the account holder recorded or produced it
LICENSED         a licence covers republishing (note/evidence required)
AUTHORIZED       written permission from the rights holder (note/evidence required)
USER_UPLOADED    set automatically when the account holder uploads the file
NOT_AUTHORIZED   explicitly refused
```

Discovery writes `UNKNOWN` and stores **metadata only, never media**.
`assert_processable()` gates every media operation and
`assert_publishable()` gates every upload; both are re-checked inside the
worker immediately before the action. Publishing additionally requires the clip
to be **approved** and not **BLOCK**ed by the safety review.

---

## What it does

**Trend discovery.** Ranks by momentum rather than totals — view velocity,
engagement rate, comment rate, recency decay and *growth acceleration* (is it
moving faster now than its own lifetime average?), combined with configurable
weights. A 100k-view video gaining 50k in the last hour outranks a 10M-view
video that has stopped.

**Topic intelligence.** Clusters discovered metadata into named topics
("AI coding agents", not three separate keyword rows), tracks video counts,
growth rate and reach over time, and feeds topic heat back into each video's
trend score.

**Semantic segmentation.** Transcripts are split where the *content* breaks —
sentence endings, pauses, speaker changes, shot changes — never every 30
seconds. Boundary strength matters: a sentence into a long pause splits a
segment even when it is short, which is what stops a channel intro being glued
onto the actual hook.

**Viral moment detection.** `ViralMomentAgent` reads the transcript and
proposes spans that stand alone, scored across twelve dimensions (hook
strength, curiosity, standalone context, payoff, shareability, comment
potential, …). Selection is then repaired deterministically: cuts are snapped
to word and sentence boundaries, extended backwards when the opening depends on
a pronoun with no antecedent, fitted to the duration window, and de-duplicated
by overlap and transcript similarity. Growth stops at *extension limits* —
reaching the minimum duration never justifies swallowing a channel intro or a
rambling sign-off.

**Smart vertical framing.** Faces are detected on sampled frames, associated
into tracks, and the *talking* face is chosen using mouth-region motion between
adjacent frames — then the camera path is median-filtered, dead-banded,
rate-limited and cut (not panned) at shot changes. The result drives ffmpeg's
`crop` filter through `sendcmd` in a single encode pass. Four layouts are
available: `SMART` (speaker-following), `CENTER`, `BLUR_PAD` (whole frame over
a blurred fill) and `SPLIT_SPEAKERS`, which stacks two framed speakers for
interviews and two-host podcasts and falls back to single-subject framing when
only one person is present.

**Captions.** Word-level timings become styled caption states rendered as RGBA
overlays — six styles including karaoke and per-word highlight — with
`.srt` and `.ass` sidecars written alongside. Fonts resolve per script, so
Devanagari (Hindi/Hinglish) renders correctly.

**Quality control.** Everything measurable is measured in code: resolution,
aspect ratio, duration, loudness (EBU R128), true peak, silence ratio, black
frames, caption timing, caption readability and safe-area position. Failures
that can be corrected automatically are — trimming dead air, enlarging
captions, extending context — bounded by a configurable attempt count, before
anything is put in front of a human.

**Safety review.** Nine categories scored 0–10 with quoted evidence, resolving
to `SAFE` / `REVIEW` / `BLOCK`. The arithmetic decision lives in code: a model
that scores hate at 9 but declares `SAFE` is overruled.

**Feedback loop.** Published clips are measured, ranked into percentiles within
the account, and the scoring weights are refit by ridge regression against
realised performance — with bounded drift, so one unusual fortnight cannot
rewrite the model.

---

## Architecture

```
                    Next.js dashboard
                            |
                            v
                       FastAPI  (thin: validate, authorize, record, enqueue)
                            |
          +-----------------+------------------+
          v                 v                  v
      PostgreSQL          Redis           Object storage
      (+pgvector)        (broker)          (local | S3)
                            |
                            v
                     Celery workers
                            |
      +---------+-----------+-----------+----------+
      v         v           v           v          v
   Trend    Analysis    Moment      Clip        Publish
   agent    pipeline    detection   generation  + analytics
                                        |
                                     FFmpeg
```

**Agents.** `TrendAgent → VideoAnalyzerAgent → ViralMomentAgent → HookAgent →
VideoQualityAgent → ContentSafetyAgent → PublishingAgent → AnalyticsAgent`.
Each declares a Pydantic output model, a prompt stating role / objective /
input / output / criteria / constraints / failure behaviour, and a
deterministic fallback.

**Provider abstraction.** `LLMProvider`, `TranscriptionProvider`,
`StorageProvider`, `VisionProvider`, `PublishingProvider` — each with multiple
implementations plus a mock, selected by configuration in
`app/providers/registry.py`.

Design decisions and their rationale: [`docs/architecture.md`](docs/architecture.md).

---

## Quick start

### Docker (recommended)

```bash
cp .env.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))" >> .env
docker compose up --build
```

- Dashboard: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Migrations run automatically (the `migrate` service must complete before the
API and workers start).

### Local development (no Docker)

```bash
# Backend
cd backend
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt

cp ../.env.example ../.env        # then set DATABASE_URL to the SQLite line
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

```bash
# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

`ffmpeg` is resolved from `FFMPEG_PATH`, then `PATH`, then the bundled
`imageio-ffmpeg` wheel — so the pipeline runs on a clean machine with no system
ffmpeg. `ffprobe` is not bundled; without it, probing falls back to parsing
`ffmpeg -i` output (fine for development, and `/health` says so). The Docker
images install both.

Without Redis, tasks run on a small in-process thread pool so the dashboard
behaves identically; `/health` reports `task_queue: degraded`. That is a
development convenience, not a production queue.

### Try the whole pipeline in two commands

```bash
python scripts/make_sample_video.py            # synthetic 90s source, no third-party content
python scripts/seed_demo.py                    # register → upload → analyse → render
```

Then open the dashboard: the video, its transcript, the ranked moments with
their reasoning, and a rendered 1080×1920 clip awaiting review.

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the
annotated list. The settings that change behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `anthropic` / `openai` / `mock`. **`mock` calls no model** — every agent uses its deterministic fallback and says so in the job warnings. |
| `LLM_MODEL` | `claude-sonnet-5` | Main model. `LLM_MODEL_CHEAP` handles packaging, QC and safety. |
| `TRANSCRIPTION_PROVIDER` | `mock` | `faster_whisper` (on-box) / `openai` (hosted) / `mock`. |
| `YOUTUBE_API_KEY` | unset | Enables trend discovery. Metadata only. |
| `STORAGE_PROVIDER` | `local` | `local` (HMAC-signed URLs) or `s3` (presigned). |
| `SECRET_KEY` | unset | **Required** in staging/production; startup fails without it. |

### Running with no credentials at all

The system is designed to be honest about degradation rather than to fake it:

- **No LLM** → agents run deterministic fallbacks. Candidates come from the
  signal model, hooks are quoted verbatim from the transcript, and the safety
  agent returns `REVIEW` — **never `SAFE`**, because nothing was actually
  assessed. `MockLLMProvider` *raises* rather than inventing text, so a bug that
  skips the fallback fails loudly instead of shipping fabricated output.
- **No transcription provider** → a `<media>.transcript.json` sidecar is used if
  present; otherwise real segment timings are emitted with placeholder words,
  flagged `synthetic`, and moment detection refuses to run rather than pretend
  to read them.
- **No broker** → in-process execution, reported as degraded.

`GET /health` reports each of these per component, and the dashboard shows a
banner.

---

## How a clip gets made

```
authorized video
      │
      ├─ probe (ffprobe → ffmpeg-stderr fallback)
      ├─ extract audio (16 kHz mono PCM, cached)
      ├─ transcribe → word-level timestamps
      ├─ detect scenes (ffmpeg scene predicate)
      ├─ segment transcript semantically
      │
      ├─ deterministic signal prefilter        ← cheap; decides what the model sees
      ├─ ViralMomentAgent (windowed, one call per chunk)
      ├─ snap to word/sentence boundaries
      ├─ repair dangling openings
      ├─ score (12 dimensions × configurable weights − structural penalties)
      ├─ de-duplicate (overlap + transcript similarity)
      │
      ├─ sample frames → face tracks → active speaker → smoothed crop path
      ├─ render captions → RGBA overlay states
      ├─ HookAgent → hooks, titles, caption, hashtags
      ├─ ONE ffmpeg pass: cut + dynamic crop + captions + hook + loudnorm
      │
      ├─ technical QC → auto-correct → re-render (bounded)
      ├─ ContentSafetyAgent
      ├─ route: auto-approve / needs review / rejected
      └─ human decision → export or publish
```

**Cost control.** The whole transcript is never sent to an expensive model.
Segments are pre-filtered by the deterministic signal score, packed into
character-bounded windows (overlapping by one segment so a moment straddling a
boundary is still seen whole), and each window is one call. Transcripts, audio,
scenes and analyses are cached — re-analysing reuses them unless `force` is
set. Token usage and estimated cost are recorded per job and shown in the UI.

---

## API

OpenAPI at `/docs`. Principal routes:

```
POST   /api/v1/auth/register | login | refresh
GET    /api/v1/auth/me
POST   /api/v1/auth/api-keys

POST   /api/v1/videos                          register (metadata; fetches nothing)
GET    /api/v1/videos
GET    /api/v1/videos/{id}
PUT    /api/v1/videos/{id}/authorization       record the rights position
POST   /api/v1/videos/{id}/upload              attach media (validated, size-capped)
POST   /api/v1/videos/{id}/analyze             → 202 + job id
GET    /api/v1/videos/{id}/transcript | scenes | candidates | segments

POST   /api/v1/clips/generate                  → 202 + job id
POST   /api/v1/clips/manual                    cut an arbitrary span
GET    /api/v1/clips | /api/v1/clips/{id}
PATCH  /api/v1/clips/{id}                      text-only edits
POST   /api/v1/clips/{id}/regenerate | variants
POST   /api/v1/clips/{id}/approve | reject
GET    /api/v1/clips/{id}/download

GET    /api/v1/jobs | /api/v1/jobs/{id}
POST   /api/v1/jobs/{id}/cancel

GET    /api/v1/trending | /api/v1/topics
POST   /api/v1/trending/discover

GET    /api/v1/dashboard | /api/v1/analytics
POST   /api/v1/analytics/calibrate

GET    /api/v1/settings   PUT /api/v1/settings
GET    /api/v1/integrations/accounts   POST /api/v1/integrations/publish/{clip_id}

GET    /health
```

Errors are uniform and carry a stable code:

```json
{
  "error_code": "content_not_authorized",
  "error_message": "Video ... has authorization_status=UNKNOWN, ...",
  "details": { "authorization_status": "UNKNOWN" },
  "request_id": "60fe87e9ffc24ea5"
}
```

---

## Testing

```bash
cd backend
pytest                        # everything (303 tests)
pytest tests/unit -q          # fast: no ffmpeg
pytest -m "not slow"          # skip the ffmpeg end-to-end
ruff check app tests
```

Coverage is concentrated where the behaviour is hard to get right:

- **Signals & segmentation** — boundary strength, boilerplate, filler,
  Hinglish, dangling openings.
- **Selection** — never cutting mid-word, running on to finish a sentence,
  duration fitting, context repair, de-duplication.
- **Scoring** — weight normalisation, penalties, clamping, partial overrides.
- **Agents** — schema validation, the repair turn, retry, fallback, plus the
  guarantees that hooks cannot invent numbers and safety never returns `SAFE`
  when nothing was assessed. A `ScriptedLLMProvider` exercises the real model
  code path without a network call.
- **Framing** — 9:16 geometry, track association, active-speaker choice,
  hysteresis, velocity limiting, jitter rejection, scene cuts, split-speaker
  window geometry and its single-speaker fallback.
- **Trend maths** — velocity, acceleration, recency decay, weight
  normalisation, topic growth, and the assertion that momentum outranks totals.
- **Renderer** — real ffmpeg renders of every filter-graph branch (dynamic
  crop, blurred padding, split screen, captions + hook, captions off).
- **Captions** — cue construction, per-word states, timeline coverage, safe area.
- **Security** — password hashing, token type confusion, path traversal, upload
  limits, signed-URL forgery and expiry, and the rights gate itself.
- **Platform APIs** — YouTube and Instagram request construction verified
  through `httpx.MockTransport`: URLs, params, headers, the resumable-upload
  chunking and 308 resume logic, Instagram's container-poll-publish sequence,
  and every error path. The real client code runs unchanged; only the socket is
  replaced.
- **Configuration** — every documented `.env` format parses, and the startup
  guards (production `SECRET_KEY`, S3 bucket, provider credentials) actually
  refuse to boot.
- **Failure isolation** — a failing clip mid-batch leaves the others committed,
  cancelled jobs never execute, retries back off then stop, and the rate
  limiter releases expired keys.
- **API contract** — tenant isolation asserted per endpoint (one account cannot
  see, read or mutate another's videos, jobs or settings, and gets 404 rather
  than 403 so existence is not disclosed), auth requirements, API-key
  revocation, pagination, filtering and the error envelope.
- **Integration** — the whole pipeline against a real ffmpeg render, plus the
  refusals (unauthorized video, unsupported upload type).

The end-to-end test plants a transcript sidecar so it exercises real moment
detection without needing a transcription model, and asserts real properties of
the output (1080×1920, checks recorded, safety never `SAFE`, boilerplate not
ranked first).

---

## Project layout

```
viral-agent/
├── backend/
│   ├── app/
│   │   ├── api/v1/           routers (auth, videos, clips, jobs, trends, …)
│   │   ├── agents/           BaseAgent + the six agents
│   │   ├── ai/prompts/       prompt text, versioned as code
│   │   ├── core/             config, logging, errors, security
│   │   ├── database/         engines, sessions, portable column types
│   │   ├── models/           SQLAlchemy models (22 tables)
│   │   ├── providers/        llm | transcription | storage | vision | publishing
│   │   ├── schemas/          Pydantic request/response models
│   │   ├── services/         pipeline, clips, selection, scoring, quality, …
│   │   ├── video/            ffmpeg, audio, scenes, tracking, captions, renderer
│   │   ├── workers/          Celery app + tasks
│   │   └── main.py
│   ├── alembic/              migrations
│   └── tests/                unit + integration + fixtures
├── frontend/                 Next.js 15 + TypeScript + Tailwind 4
├── worker/                   worker image (adds the ML wheels)
├── scripts/                  sample generator, demo seeder
├── docs/                     architecture notes
├── docker-compose.yml
└── .env.example
```

---

## Deploying

### Before the first deploy

1. **Generate a secret.** `python -c "import secrets; print(secrets.token_urlsafe(48))"`
   into `SECRET_KEY`. The API refuses to boot in `staging`/`production`
   without one — that is deliberate, not an obstacle to work around.
2. **Set `CORS_ORIGINS`** to the exact origins the browser uses, scheme
   included. `localhost` and `127.0.0.1` are different origins.
3. **Set `API_URL`** to the URL the *browser* reaches the API on, not the
   internal service name. It is read per request, so the same image works on
   any host — no rebuild to repoint it. A bare hostname is promoted to
   `https://` automatically. (`NEXT_PUBLIC_API_URL` still works as a
   build-time fallback.)
4. **Point storage somewhere durable.** `STORAGE_PROVIDER=s3` for more than one
   replica: the `local` backend writes to a volume only one node can see.
5. **Choose providers.** With the defaults (`mock`) the system runs and is
   honest about it, but no model is consulted. Set `LLM_PROVIDER`,
   `TRANSCRIPTION_PROVIDER` and their keys for real output.

### Render

A Render Blueprint is checked in as `render.yaml`. Two things differ from
Compose and both matter: **object storage is mandatory** (a Render disk
attaches to one service, so the API and workers cannot share a volume), and
**background workers are not on the free tier**. Full walkthrough, costs and
troubleshooting: [`docs/deploy-render.md`](docs/deploy-render.md).

### Docker Compose

```bash
docker compose up --build -d
docker compose logs -f migrate      # must exit 0 before api/worker start
curl -fsS https://your-host/health  # expect "status": "ok"
```

`migrate` runs `alembic upgrade head` to completion and the API and workers
wait on it, so a schema change never races a rollout.

### Verify

`GET /health` reports each component independently. A green deployment shows
`status: ok` with **no** component `degraded`. Common causes of `degraded`:

| Component | Meaning |
|---|---|
| `task_queue` | No broker reachable — work is running in-process. Fix `REDIS_URL`. |
| `llm` | `LLM_PROVIDER=mock`; agents are using deterministic fallbacks. |
| `transcription` | `TRANSCRIPTION_PROVIDER=mock`; transcripts are structural only. |
| `ffmpeg` | ffprobe missing (the container ships it; this means a custom image). |

### Scaling

- **Workers** scale horizontally: `WORKER_REPLICAS=4`. Keep per-worker
  `--concurrency` low (2–4); ffmpeg is already multi-threaded.
- **Split the queues** when render load dominates — run a fleet on
  `-Q render` and a smaller one on `-Q analysis,discovery,publishing`.
- **The API** is stateless and scales freely, but rate limiting is only shared
  across replicas when Redis is reachable; otherwise each replica counts
  separately.
- **`beat` must be a single replica.** More than one produces duplicate
  scheduled work.

### Backups

`postgres-data` holds everything that cannot be regenerated: rights records,
approvals, analytics and calibration history. `media-data` holds source
uploads and rendered clips — expensive to lose, but reproducible from sources.
Scratch space under `WORKDIR` is disposable.

---

## Operations

**Observability.** Structured JSON logs with a request id on every line and a
job id inside every task. Jobs record stage, progress, duration, retry count,
`error_code`, `error_message`, token usage and estimated cost. `/health`
reports database, queue, ffmpeg and each provider separately.

**Failure isolation.** One bad video never stops a batch: the task wrapper
catches everything, records the failure, and either schedules an exponential
backoff retry or leaves the job `FAILED` while the rest of the batch continues.
A `reap_stale_jobs` beat task fails jobs whose worker died, so nothing sits in
`PROCESSING` forever.

**Scaling.** Queues are separated (`analysis`, `render`, `discovery`,
`publishing`) so a long render cannot starve short jobs. Scale the render queue
horizontally; keep per-worker concurrency low because ffmpeg is already
multi-threaded.

**Security.** JWT access/refresh plus hashed API keys; PBKDF2 password hashing;
per-endpoint rate limiting; upload type/size validation; path-traversal
rejection at the storage layer; HMAC-signed media URLs with expiry; secrets
only from the environment, with startup refusing to run in production without
`SECRET_KEY`.

---

## Current status

**Working and verified end to end** — ingestion, authorization gating,
transcription (incl. sidecar), scene detection, segmentation, moment detection,
scoring, de-duplication, framing, captions, rendering, QC with auto-correction,
safety review, approval, export, the dashboard, and the job system. Verified by
303 passing tests including real ffmpeg renders of every framing branch, and
by running the full pipeline against a live API.

**Verified up to the network boundary, not against a live account** — YouTube
Shorts upload, Instagram Reels publishing, and YouTube trend discovery. Every
request these build is asserted against the documented API shape (URLs,
parameters, headers, multi-step protocol, chunked upload with 308 resume, and
each error path) using an intercepting transport, so the code is exercised in
full. What no test here can prove is how the real services behave: quota
responses, token refresh, processing latency and policy rejections still need
one pass with real credentials before you rely on them.

**Deliberately minimal** — the editor trims, reframes, restyles captions and
rewrites packaging; it is not a general NLE. Embeddings/pgvector columns exist
and are wired into the schema, but similarity search currently uses transcript
overlap and perceptual hashing rather than vector search.

**Recently closed.** Every one of these has a regression test.

*Deployment blockers:*

- `CORS_ORIGINS` as a comma-separated string — the format `.env.example`
  documents and Docker Compose passes — **crashed the API on boot**.
  pydantic-settings JSON-decodes list-typed fields inside the environment
  source, before any validator runs. Fixed with `NoDecode` on every
  env-readable collection field.
- Enum columns were declared as bare `String`, so SQLAlchemy returned plain
  `str` and every `loaded.status is SomeEnum.MEMBER` comparison was silently
  always false. That **disabled job cancellation entirely** and meant
  **Instagram publishing could never receive its `video_url`**. Fixed at the
  cause with an `EnumString` column type that round-trips through the enum and
  validates on write — no migration, identical DDL.

*Correctness:*

- The clip batch loop called `session.rollback()` on failure, which discarded
  every clip already produced in that transaction — the opposite of the
  documented isolation. Now a SAVEPOINT per clip.
- A validator raising `ValueError` returned **500 instead of 422**, because
  Pydantic leaves the exception object in `ctx` and it is not JSON
  serialisable. That hit the rights-note and clip-range rules.
- The "licensed content needs a recorded basis" rule was enforced on *update*
  but not on *create*.
- `SPLIT_SPEAKERS` was a valid enum value with no implementation, silently
  rendering a centre crop. Now a real stacked two-speaker layout.
- The fallback selector absorbed rambling sign-offs to reach the minimum
  duration; extension limits now stop growth at unusable material.

*Robustness:* clip assets were looked up by `video_id` (they are keyed by
`clip_id`, so the query never matched); the in-process rate limiter never
released keys, leaking memory on path-varying traffic; gzip was applied to
video downloads; and a storage failure during variant rendering aborted the
whole batch instead of failing one variant.
