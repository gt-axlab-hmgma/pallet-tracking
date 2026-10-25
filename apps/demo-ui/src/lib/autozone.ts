import { nextZoneId } from "./geometry";
import type { Point, Zone, ZoneProposal } from "./types";

/** The zones as they would be after accepting: the existing ones, then the proposed ones. */
export function addProposed(zones: Zone[], polygons: Point[][]): Zone[] {
  const out = [...zones];
  for (const polygon of polygons) {
    out.push({ id: nextZoneId(out), polygon, min_dwell_s: 0 });
  }
  return out;
}

/**
 * A colour for drawing the proposed zones that cannot be mistaken for the floor marking: the
 * opposite hue on the colour wheel from the paint. Markings come in many colours, so the outline
 * follows the one found. Without a known paint colour it falls back to the app's accent.
 * `hue` is the OpenCV hue, 0-179, which is half of the usual 0-359 degrees.
 */
export function contrastColor(hue: number | null): string {
  if (hue === null) return "var(--accent)";
  const degrees = (((hue * 2 + 180) % 360) + 360) % 360;
  return `hsl(${degrees} 85% 58%)`;
}

/**
 * The proposal to show, or none. It was made for one frame, so once the source changes it is
 * about a picture that is no longer on screen and is not shown.
 */
export function proposalFor(proposal: ZoneProposal | null, frameVersion: number): Point[][] {
  return proposal && proposal.frame_version === frameVersion ? proposal.polygons : [];
}
