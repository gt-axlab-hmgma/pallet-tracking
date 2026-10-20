import { describe, expect, it } from "vitest";
import { mergeEvents, mergeLive } from "./useDemo";
import type { DemoState, ZoneEvent } from "@/lib/types";

const ev = (id: string): ZoneEvent => ({
  event_id: id,
  device_id: "d",
  camera_id: "c",
  seq: 0,
  event_type: "enter",
  zone_id: "z",
  object_type: "pallet_type_a",
  track_id: 1,
  source: "mock",
  device_ts: "2026-10-20T14:00:00.000Z",
  message: "m",
});

describe("mergeEvents", () => {
  it("shows the newest event first", () => {
    const merged = mergeEvents([], [ev("a"), ev("b"), ev("c")]); // backend sends oldest first
    expect(merged.map((e) => e.event_id)).toEqual(["c", "b", "a"]);
  });

  it("ignores events already shown (stream plus backfill overlap)", () => {
    const first = mergeEvents([], [ev("a"), ev("b")]);
    const again = mergeEvents(first, [ev("a"), ev("b"), ev("c")]);
    expect(again.map((e) => e.event_id)).toEqual(["c", "b", "a"]);
    expect(mergeEvents(again, [ev("c")])).toBe(again);
  });

  it("caps the list", () => {
    const many = Array.from({ length: 400 }, (_, i) => ev(`e${i}`));
    expect(mergeEvents([], many)).toHaveLength(300);
  });
});

const demoState = (over: Partial<DemoState>): DemoState => ({
  camera_id: "cam01",
  device_id: "d",
  image_size: [640, 480],
  frame_version: 1,
  can_play: true,
  source: null,
  version: 1,
  zones: [],
  objects: [],
  sink: "jsonl",
  warnings: [],
  ...over,
});

describe("mergeLive", () => {
  it("takes the tracked pallets, source status and warnings but keeps the zones being edited", () => {
    const mine = demoState({ zones: [{ id: "z1", polygon: [] }], version: 3 });
    const source = { kind: "camera", name: "camera 0", state: "error", message: "lost" } as const;
    const fresh = demoState({
      zones: [],
      version: 2,
      objects: [{ id: 5, box: [0, 0, 1, 1], zones: [] }],
      source,
      warnings: ["careful"],
    });
    const merged = mergeLive(mine, fresh);
    expect(merged.objects.map((o) => o.id)).toEqual([5]);
    expect(merged.source).toEqual(source);
    expect(merged.warnings).toEqual(["careful"]);
    expect(merged.zones).toBe(mine.zones);
    expect(merged.version).toBe(3);
  });

  it("takes everything when the frame changed, for example after another source was loaded", () => {
    const mine = demoState({ frame_version: 1, image_size: [640, 480] });
    const fresh = demoState({ frame_version: 2, image_size: [1920, 1080] });
    expect(mergeLive(mine, fresh)).toBe(fresh);
  });
});
