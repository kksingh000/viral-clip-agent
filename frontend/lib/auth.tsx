"use client";

/** Session context: who is signed in, and the guard that redirects when nobody is. */

import { useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { ApiError, api, tokens } from "@/lib/api";
import type { User } from "@/types/api";

interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!tokens.access) {
        setLoading(false);
        return;
      }
      try {
        const me = await api.auth.me();
        if (!cancelled) setUser(me);
      } catch (err) {
        if (err instanceof ApiError && err.isAuth) tokens.clear();
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setError(null);
      try {
        tokens.set(await api.auth.login(email, password));
        setUser(await api.auth.me());
        router.push("/");
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : "Could not sign in.";
        setError(message);
        throw err;
      }
    },
    [router],
  );

  const register = useCallback(
    async (email: string, password: string, name?: string) => {
      setError(null);
      try {
        tokens.set(await api.auth.register(email, password, name));
        setUser(await api.auth.me());
        router.push("/");
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : "Could not create the account.";
        setError(message);
        throw err;
      }
    },
    [router],
  );

  const logout = useCallback(() => {
    tokens.clear();
    setUser(null);
    router.push("/login");
  }, [router]);

  const value = useMemo(
    () => ({ user, loading, error, login, register, logout }),
    [user, loading, error, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside an AuthProvider.");
  return context;
}

/** Redirects to /login once the session check has finished and found nobody. */
export function useRequireAuth(): { user: User | null; loading: boolean } {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  return { user, loading };
}
