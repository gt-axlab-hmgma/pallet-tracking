import { describe, expect, it } from "vitest";
import { addProposed, contrastColor, proposalFor } from "./autozone";
import type { Point, Zone } from "./types";

const bay = (x: number): Point[] => [
  [x, 100],
  [x + 80, 100],
  [x + 80, 20],
  [x, 20],
];

describe("addProposed", () => {
  it("keeps the zones that exist and adds the proposed ones after them", () => {
    const existing: Zone[] = [{ id: "dock", polygon: bay(0), min_dwell_s: 2 }];
    const out = addProposed(existing, [bay(200), bay(400)]);
    expect(out[0]).toEqual(existing[0]);
    expect(out.map((z) => z.polygon)).toEqual([bay(0), bay(200), bay(400)]);
  });

  it("names them like drawn zones, filling the first free numbers", () => {
    const existing: Zone[] = [
      { id: "zone_1", polygon: bay(0) },
      { id: "zone_3", polygon: bay(100) },
    ];
    const ids = addProposed(existing, [bay(200), bay(400), bay(600)]).map((z) => z.id);
    expect(ids).toEqual(["zone_1", "zone_3", "zone_2", "zone_4", "zone_5"]);
  });

  it("gives every ID once, and a dwell time of zero", () => {
    const out = addProposed([], [bay(0), bay(100), bay(200)]);
    expect(new Set(out.map((z) => z.id)).size).toBe(3);
    expect(out.every((z) => z.min_dwell_s === 0)).toBe(true);
  });

  it("changes nothing when nothing is proposed, and does not modify its input", () => {
    const existing: Zone[] = [{ id: "zone_1", polygon: bay(0) }];
    expect(addProposed(existing, [])).toEqual(existing);
    addProposed(existing, [bay(200)]);
    expect(existing).toHaveLength(1);
  });
});

describe("proposalFor", () => {
  const proposal = { frame_version: 4, polygons: [bay(0)], hue: 27 };

  it("shows the proposal while the frame it was made for is on screen", () => {
    expect(proposalFor(proposal, 4)).toEqual([bay(0)]);
  });

  it("shows nothing once the frame has changed", () => {
    expect(proposalFor(proposal, 5)).toEqual([]);
  });

  it("shows nothing when there is no proposal", () => {
    expect(proposalFor(null, 4)).toEqual([]);
  });
});

describe("contrastColor", () => {
  it("is the opposite hue from yellow paint, which is blue", () => {
    expect(contrastColor(27)).toBe("hsl(234 85% 58%)"); // OpenCV hue 27 is 54 degrees
  });

  it("wraps around the colour wheel", () => {
    expect(contrastColor(120)).toBe("hsl(60 85% 58%)"); // blue paint gives a yellow outline
    expect(contrastColor(179)).toBe("hsl(178 85% 58%)");
  });

  it("is exactly half a turn from the paint for every hue, so it never matches the marking", () => {
    for (let hue = 0; hue < 180; hue++) {
      const degrees = Number(/hsl\((\d+)/.exec(contrastColor(hue))?.[1]);
      const apart = Math.abs(degrees - hue * 2) % 360;
      expect(Math.min(apart, 360 - apart)).toBe(180);
    }
  });

  it("falls back to the accent colour when no paint colour is known", () => {
    expect(contrastColor(null)).toBe("var(--accent)");
  });
});
