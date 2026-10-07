import type { Box, Point, Zone } from "./types";

/** Area-weighted centroid, so the label sits inside slanted and L-shaped zones. */
export function centroid(points: Point[]): Point {
  let a = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < points.length; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[(i + 1) % points.length];
    const cross = x0 * y1 - x1 * y0;
    a += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  if (Math.abs(a) < 1e-9) {
    const n = points.length || 1;
    return [points.reduce((s, p) => s + p[0], 0) / n, points.reduce((s, p) => s + p[1], 0) / n];
  }
  return [cx / (3 * a), cy / (3 * a)];
}

/** Smallest unused `zone_N`, so deleting zone_2 and drawing again reuses the name. */
export function nextZoneId(zones: Pick<Zone, "id">[]): string {
  const used = new Set(zones.map((z) => z.id));
  let n = 1;
  while (used.has(`zone_${n}`)) n++;
  return `zone_${n}`;
}

export const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Bottom-centre of the box: where the object touches the floor. Mirrors the backend rule. */
export const groundPoint = (box: Box): Point => [(box[0] + box[2]) / 2, box[3]];

/**
 * Hues for up to six zones before they repeat. They skip the amber range (55-95) that the UI
 * uses for selection, so a zone colour is never confused with "selected".
 */
const ZONE_HUES = [245, 160, 335, 205, 120, 290];

export function zoneColor(index: number) {
  const hue = ZONE_HUES[index % ZONE_HUES.length];
  return {
    stroke: `oklch(0.76 0.14 ${hue})`,
    fill: `oklch(0.7 0.14 ${hue} / 0.2)`,
    fillActive: `oklch(0.7 0.14 ${hue} / 0.34)`,
  };
}

/** A box sized relative to the frame (140x100 on a 1280-wide frame), fanned out so adds do not stack. */
export function defaultBox(imageSize: [number, number], index: number): Box {
  const [w, h] = imageSize;
  const bw = (140 * w) / 1280;
  const bh = (100 * w) / 1280;
  const x = w * 0.5 - bw / 2 + ((index % 5) - 2) * bw * 0.6;
  const y = h * 0.82 - bh + (Math.floor(index / 5) % 2) * bh * 0.4;
  return [x, y, x + bw, y + bh];
}

/** Resize from the bottom-right handle, keeping the top-left corner fixed and a minimum size. */
export function resizeBox(box: Box, pointer: Point, min: number): Box {
  return [box[0], box[1], Math.max(box[0] + min, pointer[0]), Math.max(box[1] + min, pointer[1])];
}

export function moveBox(box: Box, dx: number, dy: number): Box {
  return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy];
}
