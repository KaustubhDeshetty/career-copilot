import type { StreamEvent } from "./types";

/**
 * Minimal Server-Sent Events reader for a fetch() response.
 * EventSource can't send a POST body, so the stream is parsed by hand:
 * events are separated by a blank line, payload lines start with "data:",
 * and lines starting with ":" are keep-alive comments.
 */
export async function* readEvents(response: Response): AsyncGenerator<StreamEvent> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = raw
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      try {
        yield JSON.parse(data) as StreamEvent;
      } catch {
        // ignore a malformed frame rather than killing the whole run
      }
    }
  }
}
