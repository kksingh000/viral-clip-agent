"use client";

import Link from "next/link";
import { useRef, useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Field,
  Input,
  LoadingRows,
  PageHeader,
  Select,
  StatusBadge,
  TextArea,
  Toast,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatBytes, formatDuration, formatRelative } from "@/lib/format";
import type { AuthorizationStatus } from "@/types/api";

const RIGHTS_OPTIONS: { value: AuthorizationStatus; label: string; note: string }[] = [
  {
    value: "USER_OWNED",
    label: "I own this content",
    note: "You recorded or produced it.",
  },
  {
    value: "LICENSED",
    label: "I hold a licence",
    note: "A licence covers republishing. Record the basis below.",
  },
  {
    value: "AUTHORIZED",
    label: "The owner authorized me",
    note: "Written permission from the rights holder. Record it below.",
  },
  {
    value: "UNKNOWN",
    label: "Not established yet",
    note: "Stored as metadata only. Cannot be analysed or published.",
  },
];

export default function VideosPage() {
  return (
    <Shell>
      <VideosContent />
    </Shell>
  );
}

function VideosContent() {
  const [statusFilter, setStatusFilter] = useState("");
  const [search, setSearch] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [toast, setToast] = useState<{ message: string; tone: "good" | "bad" } | null>(
    null,
  );

  const videos = useQuery(
    () =>
      api.videos.list({
        status: statusFilter || undefined,
        search: search || undefined,
        limit: 50,
      }),
    [statusFilter, search],
  );

  return (
    <>
      <PageHeader
        title="Videos"
        description="Source material. Only content with a recorded rights position can be analysed."
        actions={
          <Button variant="primary" onClick={() => setShowAdd((v) => !v)}>
            {showAdd ? "Cancel" : "Add video"}
          </Button>
        }
      />

      {showAdd && (
        <AddVideoForm
          onDone={(message, tone) => {
            setToast({ message, tone });
            setShowAdd(false);
            videos.refetch();
          }}
        />
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Input
          placeholder="Search titles..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs"
        />
        <Select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="max-w-[12rem]"
        >
          <option value="">All statuses</option>
          {["PENDING_MEDIA", "UPLOADED", "ANALYZING", "ANALYZED", "FAILED", "DISCOVERED"].map(
            (value) => (
              <option key={value} value={value}>
                {value.replace(/_/g, " ")}
              </option>
            ),
          )}
        </Select>
      </div>

      <ErrorState error={videos.error} onRetry={videos.refetch} />

      {videos.loading && !videos.data ? (
        <LoadingRows rows={5} />
      ) : videos.data && videos.data.items.length > 0 ? (
        <div className="space-y-2">
          {videos.data.items.map((video) => (
            <Card key={video.id} className="px-4 py-3.5">
              <div className="flex flex-wrap items-center gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Link
                      href={`/videos/${video.id}`}
                      className="truncate text-sm font-medium text-ink-50 hover:text-accent-300"
                    >
                      {video.title}
                    </Link>
                    <StatusBadge status={video.status} />
                    <StatusBadge status={video.authorization_status} />
                    {!video.has_media && (
                      <Badge tone="muted">no media</Badge>
                    )}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-ink-500">
                    <span>{video.creator_name ?? "unknown creator"}</span>
                    <span>{formatDuration(video.duration_seconds)}</span>
                    {video.width && (
                      <span>
                        {video.width}×{video.height}
                      </span>
                    )}
                    <span>{formatBytes(video.file_size_bytes)}</span>
                    <span>{formatRelative(video.created_at)}</span>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs text-ink-400">
                  <span title="Detected candidate moments">
                    {video.candidate_count} candidates
                  </span>
                  <span title="Rendered clips">{video.clip_count} clips</span>
                  <VideoActions video={video} onChanged={videos.refetch} />
                </div>
              </div>
              {video.status_detail && (
                <p className="mt-2 rounded border border-ink-700 bg-ink-850 px-2.5 py-1.5 text-[11px] text-ink-400">
                  {video.status_detail}
                </p>
              )}
            </Card>
          ))}
        </div>
      ) : (
        <EmptyState
          title="No videos yet"
          description="Register a video and upload the file you own or have licensed."
          action={
            <Button variant="primary" onClick={() => setShowAdd(true)}>
              Add video
            </Button>
          }
        />
      )}

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

function VideoActions({
  video,
  onChanged,
}: {
  video: { id: string; has_media: boolean; authorization_status: string; status: string };
  onChanged: () => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const upload = useMutation((file: File) => api.videos.upload(video.id, file));
  const analyze = useMutation(() => api.videos.analyze(video.id, false, true));

  const processable = [
    "USER_OWNED",
    "LICENSED",
    "AUTHORIZED",
    "USER_UPLOADED",
  ].includes(video.authorization_status);

  return (
    <div className="flex items-center gap-1.5">
      <input
        ref={fileInput}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          if (file) {
            await upload.run(file);
            onChanged();
          }
          event.target.value = "";
        }}
      />
      {!video.has_media && (
        <Button
          size="sm"
          loading={upload.pending}
          onClick={() => fileInput.current?.click()}
          title={
            processable
              ? "Upload the source file"
              : "Uploading records this as your own content"
          }
        >
          Upload
        </Button>
      )}
      {video.has_media && (
        <Button
          size="sm"
          variant="primary"
          loading={analyze.pending}
          disabled={!processable || video.status === "ANALYZING"}
          title={
            processable
              ? "Analyse and generate clips"
              : "Record the rights position before analysing"
          }
          onClick={async () => {
            await analyze.run();
            onChanged();
          }}
        >
          Analyse
        </Button>
      )}
      <Link href={`/videos/${video.id}`}>
        <Button size="sm" variant="ghost">
          Open
        </Button>
      </Link>
    </div>
  );
}

function AddVideoForm({
  onDone,
}: {
  onDone: (message: string, tone: "good" | "bad") => void;
}) {
  const [title, setTitle] = useState("");
  const [creator, setCreator] = useState("");
  const [rights, setRights] = useState<AuthorizationStatus>("USER_OWNED");
  const [note, setNote] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const create = useMutation(async () => {
    const video = await api.videos.create({
      title,
      source: "UPLOAD",
      creator_name: creator || undefined,
      authorization_status: rights,
      authorization_note: note || undefined,
    });
    if (file) await api.videos.upload(video.id, file);
    return video;
  });

  const selected = RIGHTS_OPTIONS.find((o) => o.value === rights);
  const needsNote = rights === "LICENSED" || rights === "AUTHORIZED";

  return (
    <Card className="mb-4 p-5">
      <form
        className="grid gap-4 lg:grid-cols-2"
        onSubmit={async (event) => {
          event.preventDefault();
          const result = await create.run();
          if (result) onDone(`Added "${result.title}".`, "good");
        }}
      >
        <Field label="Title">
          <Input
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Interview with ..."
          />
        </Field>
        <Field label="Creator">
          <Input
            value={creator}
            onChange={(e) => setCreator(e.target.value)}
            placeholder="Channel or speaker"
          />
        </Field>

        <Field label="Rights" hint={selected?.note}>
          <Select
            value={rights}
            onChange={(e) => setRights(e.target.value as AuthorizationStatus)}
          >
            {RIGHTS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Video file" hint="Optional now; you can upload later.">
          <Input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </Field>

        {needsNote && (
          <div className="lg:col-span-2">
            <Field
              label="Basis of the rights"
              hint="Required for licensed or authorized content. Stored as the record of the assertion."
            >
              <TextArea
                required
                rows={2}
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Licence reference, written permission, or contract."
              />
            </Field>
          </div>
        )}

        {create.error && (
          <div className="lg:col-span-2">
            <ErrorState error={create.error} />
          </div>
        )}

        <div className="flex items-center gap-2 lg:col-span-2">
          <Button type="submit" variant="primary" loading={create.pending}>
            Add video
          </Button>
          <span className="text-xs text-ink-500">
            Registering a video never fetches media from a third party.
          </span>
        </div>
      </form>
    </Card>
  );
}
