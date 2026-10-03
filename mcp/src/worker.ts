import { createMcpHandler } from "@modelcontextprotocol/server";
import { createServer, VERSION } from "./server.ts";
import { DataService } from "./data.ts";
import { LIMITS, readBounded } from "./bounds.ts";
import { catalog } from "./catalog.ts";
export interface Env {
  PUBLIC_ORIGIN: string;
  ASSETS?: { fetch(request: Request): Promise<Response> };
}
export function createWorker(service = new DataService()) {
  const handler = createMcpHandler(() => createServer(service), {
    responseMode: "json",
    maxRequestBodySize: LIMITS.requestBytes,
    maxSubscriptions: 0,
    keepAliveMs: 0,
  }); // Default legacy stateless compatibility remains enabled.
  let active = 0;
  return {
    async fetch(request: Request, env: Env): Promise<Response> {
      const url = new URL(request.url);
      let allowed: URL;
      try {
        allowed = new URL(env.PUBLIC_ORIGIN);
      } catch {
        return new Response("PUBLIC_ORIGIN must be configured.", {
          status: 503,
        });
      }
      if (
        url.origin !== allowed.origin ||
        (request.headers.has("host") &&
          request.headers.get("host") !== allowed.host)
      )
        return new Response("Invalid host.", { status: 403 });
      const origin = request.headers.get("origin");
      if (origin !== null && origin !== allowed.origin)
        return new Response("Origin not allowed.", { status: 403 });
      if (url.pathname === "/mcp/health" && request.method === "GET")
        return Response.json({
          status: "ok",
          version: VERSION,
          sdk: "2.3.0",
          protocol: "2026-07-28",
          datasets: catalog.length,
          limits: LIMITS,
        });
      if (url.pathname !== "/mcp")
        return env.ASSETS
          ? env.ASSETS.fetch(request)
          : new Response("Not found", { status: 404 });
      if (request.method === "OPTIONS")
        return new Response(null, {
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": allowed.origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers":
              "Content-Type, MCP-Protocol-Version",
            Vary: "Origin",
          },
        });
      if (request.method !== "POST")
        return new Response("Stateless MCP uses POST.", {
          status: 405,
          headers: { Allow: "POST, OPTIONS" },
        });
      if (active >= LIMITS.concurrency)
        return new Response("Busy; retry later.", {
          status: 429,
          headers: { "Retry-After": "2" },
        });
      if (request.headers.has("content-encoding"))
        return new Response("Encoded requests unsupported.", { status: 415 });
      if (Number(request.headers.get("content-length")) > LIMITS.requestBytes)
        return new Response("Request too large.", { status: 413 });
      active++;
      try {
        const signal = AbortSignal.any([
          request.signal,
          AbortSignal.timeout(LIMITS.timeoutMs + 1000),
        ]);
        const text = await readBounded(
          request.body,
          LIMITS.requestBytes,
          signal,
        );
        // One request, one operation: the SDK's legacy batch fallback otherwise
        // multiplies CPU/subrequest work behind a single admission slot.
        if (text.trimStart().startsWith("[")) {
          return Response.json(
            {
              jsonrpc: "2.0",
              id: null,
              error: {
                code: -32600,
                message:
                  "Batch requests are unsupported; send one MCP operation per HTTP request.",
              },
            },
            { status: 400 },
          );
        }
        // The SDK parses once; no user JSON is passed as parsedBody, bypassing SDK bounds.
        const headers = new Headers(request.headers);
        headers.delete("content-length");
        const response = await handler.fetch(
          new Request(request.url, {
            method: "POST",
            headers,
            body: text,
            signal,
          }),
        );
        const output = await readBounded(
          response.body,
          LIMITS.responseBytes,
          signal,
        );
        const responseHeaders = new Headers(response.headers);
        responseHeaders.set("Cache-Control", "no-store");
        responseHeaders.set("X-Content-Type-Options", "nosniff");
        if (origin) {
          responseHeaders.set("Access-Control-Allow-Origin", allowed.origin);
          responseHeaders.set("Vary", "Origin");
        }
        return new Response(output || null, {
          status: response.status,
          headers: responseHeaders,
        });
      } catch (e) {
        const oversized =
          e instanceof Error && e.message.includes("Byte limit");
        return new Response(
          oversized ? "Byte limit exceeded." : "Request timed out or failed.",
          { status: oversized ? 413 : 503 },
        );
      } finally {
        active--;
      }
    },
  };
}
export default createWorker();
