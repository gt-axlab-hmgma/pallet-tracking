import type { Box, DemoState, MockObject, MutationResult, Zone, ZoneEvent } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** FastAPI sends `detail` as a string for our errors and as a list for request-shape errors. */
function detailMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : null))
      .filter(Boolean);
    if (msgs.length) return msgs.join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new ApiError("Cannot reach the demo API. Is `uv run pv-demo` running?", 0);
  }
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json()).detail;
    } catch {}
    throw new ApiError(detailMessage(detail, `Request failed (${res.status})`), res.status);
  }
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) });

export const api = {
  getState: () => request<DemoState>("/api/state"),

  saveZones: (zones: Zone[]) =>
    request<DemoState & MutationResult>("/api/zones", { method: "PUT", ...json({ zones }) }),

  addObject: (object_type: string, box: Box) =>
    request<{ object: MockObject } & MutationResult>("/api/objects", {
      method: "POST",
      ...json({ object_type, box }),
    }),

  updateObject: (id: number, patch: { box?: Box; object_type?: string }) =>
    request<{ object: MockObject } & MutationResult>(`/api/objects/${id}`, {
      method: "PUT",
      ...json(patch),
    }),

  deleteObject: (id: number) => request<MutationResult>(`/api/objects/${id}`, { method: "DELETE" }),

  recentEvents: (limit = 200) =>
    request<{ events: ZoneEvent[] }>(`/api/events?limit=${limit}`).then((r) => r.events),
};
