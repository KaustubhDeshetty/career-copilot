import { BACKEND_URL, backendDown, relayJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  try {
    return await relayJson(await fetch(`${BACKEND_URL}/api/tracker`, { cache: "no-store" }));
  } catch {
    return backendDown();
  }
}
