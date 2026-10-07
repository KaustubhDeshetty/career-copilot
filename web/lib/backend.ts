// Server-only helpers for talking to the Python FastAPI backend.
// The browser never calls Python directly: every request goes through the
// Next.js route handlers in app/api/, which validate input first.

export const BACKEND_URL = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export function backendDown(): Response {
  return Response.json(
    {
      error:
        "Can't reach the Python backend. Start it with `uvicorn api:app --port 8000` " +
        `in the Career Copilot folder (expected at ${BACKEND_URL}).`,
    },
    { status: 502 },
  );
}

/** Forward a JSON response from the backend, keeping its status code. */
export async function relayJson(upstream: Response): Promise<Response> {
  const body = await upstream.json().catch(() => ({}));
  // FastAPI puts error messages in `detail`; normalise to `error` for the UI.
  if (!upstream.ok && body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    return Response.json(
      { error: typeof detail === "string" ? detail : "Request rejected by the backend." },
      { status: upstream.status },
    );
  }
  return Response.json(body, { status: upstream.status });
}
