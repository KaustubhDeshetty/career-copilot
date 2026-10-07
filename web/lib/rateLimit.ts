// Tiny in-memory rate limiter for the run endpoint. Good enough for a single
// local/server instance; a multi-instance deployment would use Redis instead.

const WINDOW_MS = 60_000;
const MAX_RUNS_PER_WINDOW = 5;
const hits = new Map<string, number[]>();

export function allowRun(key: string): { ok: true } | { ok: false; retryAfter: number } {
  const now = Date.now();
  const recent = (hits.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= MAX_RUNS_PER_WINDOW) {
    const retryAfter = Math.ceil((WINDOW_MS - (now - recent[0])) / 1000);
    hits.set(key, recent);
    return { ok: false, retryAfter };
  }
  recent.push(now);
  hits.set(key, recent);
  return { ok: true };
}

export function clientKey(request: Request): string {
  return request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "local";
}
