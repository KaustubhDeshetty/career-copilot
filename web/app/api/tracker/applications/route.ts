import { BACKEND_URL, backendDown, relayJson } from "@/lib/backend";
import { firstIssue, saveApplicationSchema } from "@/lib/validation";

export async function POST(request: Request): Promise<Response> {
  const parsed = saveApplicationSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return Response.json({ error: firstIssue(parsed.error) }, { status: 400 });
  }
  try {
    return await relayJson(
      await fetch(`${BACKEND_URL}/api/tracker/applications`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed.data),
      }),
    );
  } catch {
    return backendDown();
  }
}
