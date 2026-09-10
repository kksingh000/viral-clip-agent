"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button, Card, Field, Input } from "@/components/ui";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { user, loading, error, login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!loading && user) router.replace("/");
  }, [loading, user, router]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setPending(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password, name || undefined);
    } catch {
      /* the error is surfaced from context */
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-600 font-bold text-white">
            V
          </div>
          <div>
            <h1 className="text-lg font-semibold text-ink-50">Viral Clip Agent</h1>
            <p className="text-xs text-ink-400">
              Turn authorized long-form video into shorts.
            </p>
          </div>
        </div>

        <Card className="p-6">
          <form onSubmit={submit} className="space-y-4">
            {mode === "register" && (
              <Field label="Name">
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  autoComplete="name"
                />
              </Field>
            )}
            <Field label="Email">
              <Input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
              />
            </Field>
            <Field
              label="Password"
              hint={mode === "register" ? "At least 8 characters." : undefined}
            >
              <Input
                type="password"
                required
                minLength={mode === "register" ? 8 : undefined}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete={
                  mode === "register" ? "new-password" : "current-password"
                }
              />
            </Field>

            {error && (
              <p className="rounded-lg border border-bad-500/30 bg-bad-500/10 px-3 py-2 text-xs text-bad-500">
                {error}
              </p>
            )}

            <Button
              type="submit"
              variant="primary"
              loading={pending}
              className="w-full"
            >
              {mode === "login" ? "Sign in" : "Create account"}
            </Button>
          </form>

          <button
            onClick={() => setMode(mode === "login" ? "register" : "login")}
            className="mt-4 w-full text-center text-xs text-ink-400 hover:text-ink-200"
          >
            {mode === "login"
              ? "No account? Create one"
              : "Already have an account? Sign in"}
          </button>
        </Card>

        <p className="mt-6 text-center text-[11px] leading-relaxed text-ink-500">
          This platform processes only content you own, have licensed, or are
          otherwise authorized to use. Trend discovery reads public metadata and
          never downloads media.
        </p>
      </div>
    </div>
  );
}
