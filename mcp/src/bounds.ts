export const LIMITS = Object.freeze({
  requestBytes: 16_384,
  upstreamBytes: 131_072,
  resultBytes: 98_304,
  responseBytes: 196_608,
  rows: 100,
  offset: 10_000,
  filters: 8,
  columns: 20,
  timeoutMs: 8_000,
  concurrency: 4,
});
// Enforce decoded byte limits on the stream BEFORE JSON.parse, even without Content-Length.
export async function readBounded(
  body: ReadableStream<Uint8Array> | null,
  max: number,
  signal?: AbortSignal,
): Promise<string> {
  if (!body) return "";
  const reader = body.getReader();
  const abort = () => {
    void reader.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", abort, { once: true });
  let size = 0;
  const chunks: Uint8Array[] = [];
  try {
    while (true) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      signal?.throwIfAborted();
      if (done) break;
      size += value.byteLength;
      if (size > max)
        throw new Error(
          "Byte limit exceeded; select fewer columns or a smaller page.",
        );
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } finally {
    signal?.removeEventListener("abort", abort);
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
export function boundedResult(value: Record<string, unknown>) {
  if (
    new TextEncoder().encode(JSON.stringify(value)).byteLength >
    LIMITS.resultBytes
  )
    throw new Error("Result limit exceeded; request a smaller page.");
  return {
    content: [
      {
        type: "text" as const,
        text: "Result includes source attribution and limits in structuredContent.",
      },
    ],
    structuredContent: value,
  };
}
