# Architecture notes

Why the system is shaped the way it is. The README covers *what* it does; this
covers the decisions and the trade-offs behind them.

---

## 1. The product constraint that drives everything

The goal is not "find an interesting 30 seconds". It is:

> Find the **smallest self-contained** piece of content that maximises the
> probability of retention, completion, sharing, commenting and rewatching.

Three consequences run through the whole design:

1. **Boundaries matter more than length.** A clip that opens mid-sentence or
   ends before the payoff fails regardless of how good the middle is. So
   boundary handling is deterministic, tested, and applied *after* the model
   proposes a span — the model chooses *what*, code guarantees *where*.
2. **Standalone context is a first-class scoring dimension**, and a structural
   penalty on top. "And that is why it works" is a bad opening no matter how
   strong the sentence is, because the antecedent is outside the clip.
3. **Not finding anything is a valid answer.** The prompt says so explicitly.
   An agent that always returns five candidates is useless.

---

## 2. Where the model is, and where it is not

| Decision | Who makes it |
|---|---|
| ffmpeg invocation, geometry, encoding | code |
| Duration, resolution, loudness, black frames, caption timing | code |
| Weighted score arithmetic, penalties, thresholds | code |
| Boundary snapping, de-duplication, context repair | code |
| Trend velocity, acceleration, topic growth | code |
| *Which span is a good short* | model |
| *Whether copy is faithful and compelling* | model |
| *Whether a clip stands alone / ends abruptly* | model |
| *Whether content is unsafe* | model |

The rule: if the answer is computable, compute it. This keeps the system cheap,
reproducible, debuggable, and functional without any model at all.

There are two places where code deliberately **overrules** the model:

- **Safety.** A response scoring hate at 9 but declaring `SAFE` is corrected to
  `BLOCK` in `postprocess`. The arithmetic mapping from scores to verdicts is
  policy, not a judgement call.
- **Copy faithfulness.** `HookAgent.postprocess` drops any hook asserting a
  number that does not appear in the clip transcript. A prompt instruction not
  to invent facts is necessary but not sufficient.

---

## 3. Degradation must be honest

The system runs with zero credentials. The hard requirement is that a degraded
mode never *looks* like a working one.

- `MockLLMProvider` raises on every call and sets `is_mock = True`. Agents check
  that flag and route to their fallback. If an agent ever forgets, it gets an
  exception rather than silently fabricated output.
- The safety fallback returns `REVIEW`, never `SAFE`. Nothing was assessed, so
  claiming safety would be a false assurance. Every clip in this mode lands in
  the human queue — which is the correct behaviour, not a limitation.
- The transcription fallback emits **real timings with placeholder words** and
  flags `synthetic: true`. `ViralMomentAgent` refuses to run on a synthetic
  transcript rather than pretending to read placeholder text, and the API and
  UI both surface the flag.
- `/health` reports every provider separately, and the dashboard shows a banner.

### Why agent outputs forbid extra keys

Pydantic ignores unknown fields by default. A model returning `{"nope": 1}`
would validate into `ViralMomentOutput(candidates=[])` — indistinguishable from
the legitimate "no good moments here" answer. `AgentOutput` sets
`extra="forbid"` so malformed responses become validation errors, which the
repair-and-retry loop can act on. (Found by a test that expected a fallback and
got a silent empty result.)

---

## 4. The scoring model

Split deliberately in two:

- **Dimensions** (12, each 0–10) are *judgements* — from the model, or from
  `estimate_dimensions()` when there isn't one.
- **Combination** is *arithmetic* — a weighted sum plus explicit structural
  penalties, producing 0–100.

Keeping the arithmetic out of the model makes the final number auditable
(the UI shows every dimension bar and every penalty), reproducible, and
adjustable by the feedback loop without touching a prompt.

Weights are renormalised on every override so a partial change cannot silently
rescale the score.

### The deterministic signal model

`app/services/signals.py` is a small, explainable lexicon-and-counts prior. It
does three jobs:

1. **Cheap prefilter** — decides which segments are worth model tokens.
2. **Fallback scoring** — gives the no-LLM path something real to rank with.
3. **Reference** — logged next to the model's own scores so the two can be
   compared.

It is a prior, not a classifier, and it will never judge whether a joke lands.
Two calibration notes learned the hard way:

- Scores are centred on `PREFILTER_BASELINE = 0.28`, not zero. An earlier
  version floored most spans at 0.0, which made every candidate tie and
  destroyed the ranking the fallback depends on.
- Channel boilerplate ("welcome back", "subscribe") is the strongest *negative*
  signal there is. It is fluent, confident speech that every positive lexicon
  happily rewards, so it must be subtracted explicitly.

### Extension limits

"Unusable" material — boilerplate, or speech that is mostly filler — is defined
once and honoured at **three** stages, because fixing only one of them moves the
bug rather than removing it:

1. **Seeding.** An unusable segment cannot seed a candidate. Growing outwards
   from "please subscribe" opens on the ask; growing outwards from a sign-off
   ends on it.
2. **Segment growth.** Neighbouring unusable segments are never absorbed.
3. **Boundary snapping.** `snap_boundaries` extends word-by-word to reach the
   minimum duration, and used to walk straight past both of the above. It now
   takes `ExtensionLimits` marking the nearest unusable segment on each side and
   stops there, recording *why* it stopped short.

Reaching the target length is never worth making the clip worse.

---

## 5. Segmentation: boundary strength

Fixed slices cut through sentences and jokes. But a naive minimum segment
length is just as bad: enforcing it across a real content boundary glues the
channel intro onto the hook that follows it.

So boundaries carry a **strength**:

| Boundary | Strength |
|---|---|
| Speaker change | 1.00 |
| Gap ≥ 1.1 s | 0.95 |
| Sentence end into a pause ≥ 0.5 s | 0.90 |
| Sentence end | 0.70 |
| Gap ≥ 0.5 s | 0.60 |
| Shot change | 0.50 |

A segment closes at any boundary once it is long enough, **or** at a strong
boundary (≥ 0.85) once it is at least 2 s. That single rule is what separates
"please subscribe." from "Most people completely misunderstand…".

---

## 6. Framing: how the active speaker is chosen

True active-speaker detection needs audio-visual sync or diarization. The
approximation used here is cheap and works on real footage:

At each sample point the vision provider decodes the frame **and the next
frame** and measures mean absolute difference over the mouth region of each
detected face (lower third of the box, middle half horizontally). A talking
face moves its mouth; a listening one does not. Whole-frame motion would be
dominated by camera movement, which is why the measurement is local to the
face — and why it uses adjacent frames rather than adjacent *samples* (half a
second apart, mouth movement is noise).

The subject is then chosen by a weighted score (mouth activity 0.50, face size
0.25, persistence 0.15, centrality 0.10) with **hysteresis**: a challenger must
beat the incumbent by a margin *and* the incumbent must have held the frame for
a minimum time. Without that, two speakers of similar prominence make the crop
ping-pong every frame.

The raw target series is then median-filtered (kills single-frame detection
glitches), dead-banded (ignores sub-1.5 % movements), exponentially smoothed and
velocity-limited — and **cut, not panned, at shot changes**, because panning
across an edit looks broken.

### Split-speaker framing

For interviews and two-host podcasts, `SPLIT_SPEAKERS` frames the two most
prominent faces into stacked halves. The windows are **static** by design: a
split screen that also pans is disorienting, and the point of the layout is
that both speakers stay visible.

Two guards stop it producing something worse than a single crop:

- Fewer than two tracks → return `None` and fall back to speaker-following,
  with a note on the clip. A split screen of one person is not a split screen.
- Two tracks whose median centres are within 12 % of the frame width → also
  `None`. That is one face whose detection broke and restarted, and rendering
  it twice looks like a bug.

Each window is cut at exactly the aspect of half the output frame, so the scale
into `vstack` never distorts.

This mode previously existed only as an enum value: requesting it produced a
static centre crop and said nothing about it. An enum member that silently does
something else is worse than one that does not exist, which is why the
implementation now either delivers the layout or explains the fallback.

The result is a minimal keyframe list driving ffmpeg's `crop` filter through
`sendcmd`, so the whole thing happens in the single encode pass.

**Performance note.** Frame sampling seeks once, then walks sequentially using
`grab()` to skip without decoding, and runs Haar detection on a 640px-wide copy.
Seeking per sample instead was ~4.5× slower. Timestamps come from
`CAP_PROP_POS_FRAMES`, not `CAP_PROP_POS_MSEC`, which reports a stale value on
the first read after a seek.

---

## 7. Captions: why PNG overlays rather than burned-in ASS

Rendering caption states to RGBA PNGs and compositing them as one overlay was
chosen over ffmpeg's `subtitles` filter because:

- It does not depend on the ffmpeg build carrying **libass**. The bundled
  `imageio-ffmpeg` binary happens to have it; many minimal builds do not.
- Per-word highlight boxes, karaoke fills and rounded backgrounds are just
  drawing operations, with no subtitle-format gymnastics.
- Identical states are cached, and the whole sequence enters ffmpeg as a single
  `concat` input — one `overlay` filter, not hundreds of `enable=between(...)`
  evaluations.

`.srt` and `.ass` sidecars are still written for every clip, so the captions
remain editable and portable.

Fonts resolve **per script**: text containing U+0900–U+097F gets a Devanagari
face. Without that, Hindi and Hinglish captions render as boxes. The container
installs DejaVu and Noto (including Devanagari); a missing font degrades to
Pillow's bitmap font with a logged warning rather than failing the render.

---

## 8. Rendering: one pass, with a fallback

Cut, dynamic crop, scale, caption overlay, hook card and loudness
normalisation all happen in a single ffmpeg invocation. Multiple passes would
mean generation loss and roughly linear extra time.

The caption overlay is the most fragile input (many PNGs through the concat
demuxer), so a render failure with captions present retries **once without
them** and records the reason on the clip, rather than losing the clip entirely.

Crop windows are computed to be exactly 9:16 before scaling, so the final
`scale` never distorts. Sources narrower than 9:16 crop vertically instead.

---

## 9. Portability: Postgres in production, SQLite everywhere else

The production target is PostgreSQL with pgvector. But the full pipeline needs
to be runnable — and testable in CI — without a database server, so the models
use portable column types:

- `GUID` — native `uuid` on Postgres, `CHAR(32)` elsewhere.
- `JSONB` — `JSONB` on Postgres, generic `JSON` elsewhere.
- `Vector` — pgvector when present, JSON array otherwise.
- `UTCDateTime` — always stores and returns timezone-aware UTC.

`UTCDateTime` exists because SQLite has no timezone support and returns naive
datetimes, which then cannot be subtracted from the aware values the
application creates. That surfaced as a `TypeError` while computing job
durations; normalising at the column boundary fixes it everywhere at once
rather than at each arithmetic site.

Alembic runs in batch mode on SQLite so the same migrations apply to both.

---

## 10. Async API, sync workers

The API is async (FastAPI + asyncpg). Workers are synchronous (Celery +
psycopg), because ffmpeg, OpenCV and CTranslate2 are all blocking and gain
nothing from an event loop.

The API only ever *creates* jobs — `create_job_async` is the single async entry
point; every subsequent transition belongs to the worker. The few API routes
that need synchronous services (download resolution, publish scheduling,
calibration) run them via `anyio.to_thread.run_sync` rather than duplicating
the service layer.

Blocking storage writes during upload also go through `to_thread`, so a large
upload never stalls the event loop.

---

## 11. Failure isolation

Every task body runs inside `run_job()`, which:

1. loads the job row and refuses to run a cancelled one,
2. marks it `PROCESSING`,
3. runs the body in a transactional scope,
4. on success records the result,
5. on failure records `error_code`, `error_message`, `retry_count` and timing,
   then either schedules an exponential backoff retry (for errors marked
   `retryable`) or leaves it `FAILED`.

Nothing re-raises past that wrapper. Within a batch, a clip that fails to render
is rolled back and recorded in `failures[]` while the remaining clips continue —
one bad video cannot stop the pipeline.

A `reap_stale_jobs` beat task fails jobs stuck in `PROCESSING` past a cutoff, so
a killed worker does not leave phantom work on the dashboard forever.

---

## 12. Cost control

1. **Cheap filtering first.** Deterministic signals rank segments; only the
   survivors reach a model.
2. **Windowing.** Transcripts are packed into character-bounded windows that
   overlap by one segment, so a moment straddling a boundary is still seen whole
   by one call.
3. **Model tiering.** Packaging, QC and safety use `LLM_MODEL_CHEAP`; only
   moment detection uses the main model.
4. **Caching.** Extracted audio, transcripts, scenes and analyses are reused on
   re-analysis unless `force` is set.
5. **Accounting.** Tokens and estimated cost are accumulated per job and
   surfaced in the UI, so the expensive step is visible rather than inferred.

---

## 13. What is deliberately not built

- **A general NLE.** The editor trims, reframes, restyles captions and rewrites
  packaging. Anything beyond that belongs in a real editor.
- **Vector similarity search.** The pgvector columns exist and the schema is
  ready, but duplicate detection currently uses timestamp overlap, transcript
  Jaccard similarity and perceptual hashing — which is sufficient at this scale
  and has no embedding cost.
- **Self-serve OAuth flows.** The deployment performs the OAuth exchange; the
  API records the resulting tokens. Putting client secrets and redirect handling
  inside the app would tie it to one hosting arrangement.
- **Automatic publishing by default.** Every clip needs a decision unless it
  clears the auto-approve threshold *and* the safety review, and the source
  rights permit republishing.
