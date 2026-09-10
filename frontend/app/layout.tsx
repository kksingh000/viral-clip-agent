import type { Metadata, Viewport } from "next";

import { AuthProvider } from "@/lib/auth";

import "./globals.css";

export const metadata: Metadata = {
  title: "Viral Clip Agent",
  description:
    "Discover momentum, find the self-contained moments inside authorized long-form video, and turn them into vertical shorts.",
  robots: { index: false, follow: false },
};

/**
 * Render every page per request.
 *
 * Without this, Next statically prerenders the pages and bakes the value
 * below into the HTML at build time -- which is exactly the problem the
 * runtime config exists to solve. Every page here is a client component
 * behind authentication, so static prerendering buys nothing anyway.
 */
export const dynamic = "force-dynamic";

export const viewport: Viewport = {
  themeColor: "#08090c",
  width: "device-width",
  initialScale: 1,
};

/**
 * The API URL is resolved per request, not baked into the bundle.
 *
 * `NEXT_PUBLIC_*` values are inlined at build time, which means an image can
 * only ever talk to the host it was built for. On any platform where the
 * API's URL is not known until the service exists (Render, Fly, most PaaS)
 * that is a guaranteed misconfiguration. This layout is a server component,
 * so it reads the value on every request and hands it to the browser; the
 * client falls back to the build-time value, then to localhost.
 */
function runtimeConfigScript(): string {
  const apiUrl = (
    process.env.API_URL ??
    process.env.NEXT_PUBLIC_API_URL ??
    "http://localhost:8000"
  ).replace(/[/]$/, "");

  // JSON.stringify quotes the value but leaves `<` alone, so a stray
  // `</script>` in the environment would break out of the tag. The value is
  // operator-supplied rather than user-supplied, but escaping costs nothing
  // and removes the question. U+2028/2029 are included because they are valid
  // JSON yet terminate a line in JavaScript.
  const encoded = JSON.stringify(apiUrl).replace(
    /[<>\u2028\u2029]/g,
    (char) => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"),
  );
  return `window.__API_URL__=${encoded};`;
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <script
          id="runtime-config"
          dangerouslySetInnerHTML={{ __html: runtimeConfigScript() }}
        />
      </head>
      <body className="antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
