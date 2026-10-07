import { describe, expect, it } from "vitest";
import {
  centroid,
  defaultBox,
  distance,
  groundPoint,
  moveBox,
  nextZoneId,
  resizeBox,
  zoneColor,
} from "./geometry";

describe("centroid", () => {
  it("is the centre of a square", () => {
    expect(centroid([[0, 0], [10, 0], [10, 10], [0, 10]])).toEqual([5, 5]);
  });

  it("stays inside an L shape, unlike the bounding-box centre", () => {
    const [x, y] = centroid([[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]]);
    expect(x).toBeLessThan(50);
    expect(y).toBeLessThan(50);
  });

  it("is independent of winding order", () => {
    const sq: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]];
    expect(centroid([...sq].reverse())).toEqual(centroid(sq));
  });

  it("falls back to the mean for a degenerate polygon", () => {
    expect(centroid([[0, 0], [5, 5], [10, 10]])).toEqual([5, 5]);
  });
});

describe("nextZoneId", () => {
  it("starts at zone_1 and fills gaps", () => {
    expect(nextZoneId([])).toBe("zone_1");
    expect(nextZoneId([{ id: "zone_1" }, { id: "zone_3" }])).toBe("zone_2");
    expect(nextZoneId([{ id: "loading" }])).toBe("zone_1");
  });
});

describe("boxes", () => {
  it("ground point is bottom-centre", () => {
    expect(groundPoint([100, 50, 200, 150])).toEqual([150, 150]);
  });

  it("default boxes scale with the frame and stay inside it", () => {
    for (const size of [[1280, 720], [640, 480], [1920, 1080]] as [number, number][]) {
      for (let i = 0; i < 12; i++) {
        const [x1, y1, x2, y2] = defaultBox(size, i);
        expect(x1).toBeGreaterThanOrEqual(0);
        expect(y1).toBeGreaterThanOrEqual(0);
        expect(x2).toBeLessThanOrEqual(size[0]);
        expect(y2).toBeLessThanOrEqual(size[1]);
      }
    }
    expect(defaultBox([1280, 720], 0)).not.toEqual(defaultBox([1280, 720], 1));
  });

  it("moves and resizes with a minimum size", () => {
    expect(moveBox([0, 0, 10, 10], 5, -3)).toEqual([5, -3, 15, 7]);
    expect(resizeBox([10, 10, 50, 50], [80, 90], 20)).toEqual([10, 10, 80, 90]);
    expect(resizeBox([10, 10, 50, 50], [11, 12], 20)).toEqual([10, 10, 30, 30]);
  });

  it("measures distance", () => {
    expect(distance([0, 0], [3, 4])).toBe(5);
  });
});

describe("zoneColor", () => {
  it("gives neighbouring zones different hues and repeats after six", () => {
    expect(zoneColor(0).stroke).not.toBe(zoneColor(1).stroke);
    expect(zoneColor(0).stroke).toBe(zoneColor(6).stroke);
  });
});
