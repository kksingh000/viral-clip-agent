"use client";

import { useEffect, useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Button,
  Card,
  CardHeader,
  ErrorState,
  Field,
  Input,
  LoadingRows,
  PageHeader,
  Select,
  Toast,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { titleCase } from "@/lib/format";
import type { CaptionPosition, CaptionStyle, CropMode, Settings } from "@/types/api";

export default function SettingsPage() {
  return (
    <Shell>
      <SettingsContent />
    </Shell>
  );
}

function SettingsContent() {
  const settings = useQuery(() => api.settings.get(), []);
  const defaults = useQuery(() => api.settings.defaults(), []);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [toast, setToast] = useState<{ message: string; tone: "good" | "bad" } | null>(
    null,
  );

  useEffect(() => {
    if (settings.data) setDraft(settings.data);
  }, [settings.data]);

  const save = useMutation((body: Record<string, unknown>) =>
    api.settings.update(body),
  );

  if (settings.loading && !draft) return <LoadingRows rows={5} />;
  if (settings.error)
    return <ErrorState error={settings.error} onRetry={settings.refetch} />;
  if (!draft) return null;

  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    setDraft({ ...draft, [key]: value });

  const submit = async () => {
    const result = await save.run({
      min_viral_score: draft.min_viral_score,
      auto_approve_score: draft.auto_approve_score,
      reject_below_score: draft.reject_below_score,
      clips_per_video: draft.clips_per_video,
      min_clip_seconds: draft.min_clip_seconds,
      max_clip_seconds: draft.max_clip_seconds,
      crop_mode: draft.crop_mode,
      caption_style: draft.caption_style,
      caption_position: draft.caption_position,
      output_width: draft.output_width,
      output_height: draft.output_height,
      output_fps: draft.output_fps,
      quality_threshold: draft.quality_threshold,
      max_regeneration_attempts: draft.max_regeneration_attempts,
      block_on_safety_review: draft.block_on_safety_review,
      scoring_weights: draft.scoring_weights,
      preferred_languages: draft.preferred_languages,
      trend_regions: draft.trend_regions,
    });
    setToast(
      result
        ? { message: "Settings saved.", tone: "good" }
        : { message: save.error?.message ?? "Could not save.", tone: "bad" },
    );
    if (result) settings.refetch();
  };

  return (
    <>
      <PageHeader
        title="Settings"
        description="Thresholds, clip shape, render defaults and scoring weights."
        actions={
          <Button variant="primary" loading={save.pending} onClick={submit}>
            Save changes
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Approval thresholds"
            description="Auto-approve above the first number; require review between; flag below the second."
          />
          <div className="grid gap-4 p-5 sm:grid-cols-2">
            <Field label="Auto-approve at or above">
              <Input
                type="number"
                min={0}
                max={100}
                value={draft.auto_approve_score}
                onChange={(e) => set("auto_approve_score", Number(e.target.value))}
              />
            </Field>
            <Field label="Flag below">
              <Input
                type="number"
                min={0}
                max={100}
                value={draft.reject_below_score}
                onChange={(e) => set("reject_below_score", Number(e.target.value))}
              />
            </Field>
            <Field
              label="Minimum score to render"
              hint="Candidates below this are never turned into clips."
            >
              <Input
                type="number"
                min={0}
                max={100}
                value={draft.min_viral_score}
                onChange={(e) => set("min_viral_score", Number(e.target.value))}
              />
            </Field>
            <Field label="Clips per video">
              <Input
                type="number"
                min={1}
                max={20}
                value={draft.clips_per_video}
                onChange={(e) => set("clips_per_video", Number(e.target.value))}
              />
            </Field>
            <div className="sm:col-span-2">
              <label className="flex items-center gap-2 text-xs text-ink-300">
                <input
                  type="checkbox"
                  checked={draft.block_on_safety_review}
                  onChange={(e) =>
                    set("block_on_safety_review", e.target.checked)
                  }
                  className="h-3.5 w-3.5 accent-indigo-500"
                />
                Never auto-approve a clip the safety agent flagged for review
              </label>
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Clip shape"
            description="The agent picks a length inside this window rather than a fixed number."
          />
          <div className="grid gap-4 p-5 sm:grid-cols-2">
            <Field label="Minimum seconds">
              <Input
                type="number"
                min={5}
                max={180}
                value={draft.min_clip_seconds}
                onChange={(e) => set("min_clip_seconds", Number(e.target.value))}
              />
            </Field>
            <Field label="Maximum seconds">
              <Input
                type="number"
                min={5}
                max={180}
                value={draft.max_clip_seconds}
                onChange={(e) => set("max_clip_seconds", Number(e.target.value))}
              />
            </Field>
            <Field label="Output width">
              <Input
                type="number"
                value={draft.output_width}
                onChange={(e) => set("output_width", Number(e.target.value))}
              />
            </Field>
            <Field label="Output height">
              <Input
                type="number"
                value={draft.output_height}
                onChange={(e) => set("output_height", Number(e.target.value))}
              />
            </Field>
            <Field label="Frame rate" hint="Blank follows the source.">
              <Input
                type="number"
                min={24}
                max={60}
                value={draft.output_fps ?? ""}
                onChange={(e) =>
                  set(
                    "output_fps",
                    e.target.value === "" ? null : Number(e.target.value),
                  )
                }
              />
            </Field>
          </div>
        </Card>

        <Card>
          <CardHeader title="Render defaults" />
          <div className="grid gap-4 p-5 sm:grid-cols-2">
            <Field label="Framing">
              <Select
                value={draft.crop_mode}
                onChange={(e) => set("crop_mode", e.target.value as CropMode)}
              >
                {["SMART", "CENTER", "BLUR_PAD", "SPLIT_SPEAKERS"].map((value) => (
                  <option key={value} value={value}>
                    {titleCase(value)}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Caption style">
              <Select
                value={draft.caption_style}
                onChange={(e) =>
                  set("caption_style", e.target.value as CaptionStyle)
                }
              >
                {[
                  "WORD_HIGHLIGHT",
                  "KARAOKE",
                  "BOLD",
                  "NORMAL",
                  "MINIMAL",
                  "PODCAST",
                  "NONE",
                ].map((value) => (
                  <option key={value} value={value}>
                    {titleCase(value)}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Caption position">
              <Select
                value={draft.caption_position}
                onChange={(e) =>
                  set("caption_position", e.target.value as CaptionPosition)
                }
              >
                {["LOWER_THIRD", "CENTER", "UPPER_THIRD", "BOTTOM", "TOP"].map(
                  (value) => (
                    <option key={value} value={value}>
                      {titleCase(value)}
                    </option>
                  ),
                )}
              </Select>
            </Field>
            <Field
              label="Quality threshold"
              hint="Below this the clip is re-rendered, then sent for review."
            >
              <Input
                type="number"
                min={0}
                max={100}
                value={draft.quality_threshold}
                onChange={(e) => set("quality_threshold", Number(e.target.value))}
              />
            </Field>
            <Field label="Auto-correction attempts">
              <Input
                type="number"
                min={0}
                max={5}
                value={draft.max_regeneration_attempts}
                onChange={(e) =>
                  set("max_regeneration_attempts", Number(e.target.value))
                }
              />
            </Field>
            <Field label="Languages" hint="Comma separated. 'auto' detects.">
              <Input
                value={draft.preferred_languages.join(", ")}
                onChange={(e) =>
                  set(
                    "preferred_languages",
                    e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
                  )
                }
              />
            </Field>
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Scoring weights"
            description="How much each dimension contributes. Values are renormalised on save."
            action={
              defaults.data ? (
                <button
                  onClick={() =>
                    set("scoring_weights", { ...defaults.data!.scoring_weights })
                  }
                  className="text-xs text-accent-300 hover:text-accent-200"
                >
                  Reset to defaults
                </button>
              ) : undefined
            }
          />
          <div className="space-y-2.5 p-5">
            {Object.entries(draft.scoring_weights).map(([name, value]) => (
              <div key={name} className="flex items-center gap-3">
                <span className="w-40 shrink-0 text-xs text-ink-400">
                  {titleCase(name)}
                </span>
                <input
                  type="range"
                  min={0}
                  max={0.35}
                  step={0.005}
                  value={value}
                  onChange={(e) =>
                    set("scoring_weights", {
                      ...draft.scoring_weights,
                      [name]: Number(e.target.value),
                    })
                  }
                  className="flex-1 accent-indigo-500"
                />
                <span className="w-12 shrink-0 text-right text-xs tabular-nums text-ink-200">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
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
