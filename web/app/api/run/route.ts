import { BACKEND_URL, backendDown } from "@/lib/backend";
import { allowRun, clientKey } from "@/lib/rateLimit";
import { firstIssue, runRequestSchema } from "@/lib/validation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * POST /api/run  ->  validates, then proxies to FastAPI POST /api/run/stream
 * and pipes the Server-Sent Events straight back to the browser.
 */
export async function POST(request: Request): Promise<Response> {
  const json = await request.json().catch(() => null);
  const parsed = runRequestSchema.safeParse(json);
  if (!parsed.success) {
    return Response.json({ error: firstIssue(parsed.error) }, { status: 400 });
  }

  const limit = allowRun(clientKey(request));
  if (!limit.ok) {
    return Response.json(
      { error: `Too many runs. Try again in ${limit.retryAfter}s.` },
      { status: 429, headers: { "Retry-After": String(limit.retryAfter) } },
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/run/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: parsed.data.task, run_anyway: parsed.data.runAnyway }),
      cache: "no-store",
    });
  } catch {
    return backendDown();
  }

  if (!upstream.ok || !upstream.body) {
    const body = await upstream.json().catch(() => ({}));
    const detail = (body as { detail?: unknown }).detail;
    return Response.json(
      { error: typeof detail === "string" ? detail : `Backend error (${upstream.status}).` },
      { status: upstream.status || 502 },
    );
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
