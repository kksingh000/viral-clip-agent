"use client";

import { useState } from "react";

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
  Toast,
} from "@/components/ui";
import { useMutation, useQuery } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";

const PLATFORMS = [
  { value: "YOUTUBE_SHORTS", label: "YouTube Shorts" },
  { value: "INSTAGRAM_REELS", label: "Instagram Reels" },
];

export default function IntegrationsPage() {
  return (
    <Shell>
      <IntegrationsContent />
    </Shell>
  );
}

function IntegrationsContent() {
  const accounts = useQuery(() => api.integrations.accounts(), []);
  const [toast, setToast] = useState<{ message: string; tone: "good" | "bad" } | null>(
    null,
  );
  const [showConnect, setShowConnect] = useState(false);

  return (
    <>
      <PageHeader
        title="Integrations"
        description="Connected publishing accounts. Only official platform APIs are used."
        actions={
          <Button variant="primary" onClick={() => setShowConnect((v) => !v)}>
            {showConnect ? "Cancel" : "Connect account"}
          </Button>
        }
      />

      <div className="mb-4 rounded-lg border border-ink-700 bg-ink-850 px-4 py-3 text-xs leading-relaxed text-ink-400">
        Publishing is gated twice: the <strong className="text-ink-200">source</strong>{" "}
        must carry a rights position that permits republishing, and the{" "}
        <strong className="text-ink-200">clip</strong> must be approved and not
        blocked by the safety review. Both checks run again inside the worker
        immediately before upload.
      </div>

      {showConnect && (
        <ConnectForm
          onDone={(message, tone) => {
            setToast({ message, tone });
            setShowConnect(false);
            accounts.refetch();
          }}
        />
      )}

      <ErrorState error={accounts.error} onRetry={accounts.refetch} />

      {accounts.loading && !accounts.data ? (
        <LoadingRows rows={2} />
      ) : accounts.data && accounts.data.length > 0 ? (
        <div className="space-y-2">
          {accounts.data.map((account) => (
            <AccountRow
              key={account.id}
              account={account}
              onChanged={() => accounts.refetch()}
            />
          ))}
        </div>
      ) : (
        <EmptyState
          title="No accounts connected"
          description="Until an account is connected, approved clips are exported for manual upload."
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

function AccountRow({
  account,
  onChanged,
}: {
  account: {
    id: string;
    platform: string;
    display_name: string | null;
    external_account_id: string;
    is_active: boolean;
    connected_at: string | null;
    token_expires_at: string | null;
    last_error: string | null;
  };
  onChanged: () => void;
}) {
  const disconnect = useMutation(() => api.integrations.disconnect(account.id));
  const expired =
    account.token_expires_at != null &&
    new Date(account.token_expires_at).getTime() < Date.now();

  return (
    <Card className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <Badge tone="accent">{account.platform.replace(/_/g, " ")}</Badge>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm text-ink-100">
            {account.display_name ?? account.external_account_id}
          </div>
          <div className="text-[11px] text-ink-500">
            Connected {formatRelative(account.connected_at)}
            {expired && " · token expired"}
          </div>
        </div>
        {expired && <Badge tone="warn">reconnect</Badge>}
        {!account.is_active && <Badge tone="muted">inactive</Badge>}
        <Button
          size="sm"
          variant="danger"
          loading={disconnect.pending}
          onClick={async () => {
            await disconnect.run();
            onChanged();
          }}
        >
          Disconnect
        </Button>
      </div>
      {account.last_error && (
        <p className="mt-2 rounded border border-bad-500/30 bg-bad-500/10 px-2.5 py-1.5 text-[11px] text-bad-500">
          {account.last_error}
        </p>
      )}
    </Card>
  );
}

function ConnectForm({
  onDone,
}: {
  onDone: (message: string, tone: "good" | "bad") => void;
}) {
  const [platform, setPlatform] = useState(PLATFORMS[0]!.value);
  const [accountId, setAccountId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [accessToken, setAccessToken] = useState("");
  const [refreshToken, setRefreshToken] = useState("");

  const connect = useMutation(() =>
    api.integrations.connect({
      platform,
      external_account_id: accountId,
      display_name: displayName || undefined,
      access_token: accessToken,
      refresh_token: refreshToken || undefined,
    }),
  );

  return (
    <Card className="mb-4 p-5">
      <p className="mb-4 text-xs leading-relaxed text-ink-400">
        Your deployment performs the OAuth exchange (the client secret and
        redirect URI live outside this app). Paste the resulting tokens here to
        record the account. Tokens are stored as opaque references and never
        appear in logs.
      </p>
      <form
        className="grid gap-4 lg:grid-cols-2"
        onSubmit={async (event) => {
          event.preventDefault();
          const result = await connect.run();
          if (result) onDone("Account connected.", "good");
          else onDone(connect.error?.message ?? "Could not connect.", "bad");
        }}
      >
        <Field label="Platform">
          <Select value={platform} onChange={(e) => setPlatform(e.target.value)}>
            {PLATFORMS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </Field>
        <Field
          label="Account id"
          hint={
            platform === "INSTAGRAM_REELS"
              ? "Instagram Business account id"
              : "YouTube channel id"
          }
        >
          <Input
            required
            value={accountId}
            onChange={(e) => setAccountId(e.target.value)}
          />
        </Field>
        <Field label="Display name">
          <Input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Optional label"
          />
        </Field>
        <Field label="Access token">
          <Input
            required
            type="password"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
            autoComplete="off"
          />
        </Field>
        <Field label="Refresh token" hint="Optional but recommended.">
          <Input
            type="password"
            value={refreshToken}
            onChange={(e) => setRefreshToken(e.target.value)}
            autoComplete="off"
          />
        </Field>
        <div className="lg:col-span-2">
          <ErrorState error={connect.error} />
          <Button type="submit" variant="primary" loading={connect.pending}>
            Connect
          </Button>
        </div>
      </form>
    </Card>
  );
}
