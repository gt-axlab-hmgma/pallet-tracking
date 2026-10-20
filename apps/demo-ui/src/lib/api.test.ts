import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("api", () => {
  it("sends zones as a JSON PUT", async () => {
    const fn = mockFetch(200, { zones: [], events: [], sink_error: null });
    await api.saveZones([
      {
        id: "a",
        polygon: [
          [0, 0],
          [1, 0],
          [1, 1],
        ],
      },
    ]);
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/zones");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({
      zones: [
        {
          id: "a",
          polygon: [
            [0, 0],
            [1, 0],
            [1, 1],
          ],
        },
      ],
    });
  });

  it("surfaces the backend's message on 422", async () => {
    mockFetch(422, { detail: "polygon edges self-intersect" });
    await expect(api.saveZones([])).rejects.toMatchObject({
      status: 422,
      message: "polygon edges self-intersect",
    });
  });

  it("joins FastAPI validation messages", async () => {
    mockFetch(422, { detail: [{ msg: "Field required" }, { msg: "bad box" }] });
    await expect(api.connectCamera("x")).rejects.toThrow("Field required; bad box");
  });

  it("explains when the API is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("fetch failed"))),
    );
    const err = await api.getState().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toContain("pv-demo");
  });

  it("sends a video as the raw body with its name in the query", async () => {
    const fn = mockFetch(200, { source: null });
    const file = new File(["frames"], "dock 3 & 4.mp4", { type: "video/mp4" });
    await api.uploadVideo(file);
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/source/video?name=dock%203%20%26%204.mp4");
    expect(init.method).toBe("PUT");
    expect(init.body).toBe(file);
    expect(init.headers).toEqual({ "Content-Type": "application/octet-stream" });
  });

  it("connects a camera with a JSON body and stops a source with DELETE", async () => {
    const fn = mockFetch(200, { source: null });
    await api.connectCamera("rtsp://cam.local/stream");
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/source/camera");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ device: "rtsp://cam.local/stream" });
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    await api.stopSource();
    const [stopUrl, stopInit] = fn.mock.calls[1] as unknown as [string, RequestInit];
    expect(stopUrl).toBe("/api/source");
    expect(stopInit.method).toBe("DELETE");
  });

  it("unwraps recent events", async () => {
    mockFetch(200, { events: [{ event_id: "e" }] });
    expect(await api.recentEvents(5)).toEqual([{ event_id: "e" }]);
  });
});
