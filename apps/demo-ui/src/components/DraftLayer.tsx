import type { Point } from "@/lib/types";

interface Props {
  points: Point[];
  cursor: Point | null;
  px: number;
  /** True when the cursor is close enough to the first vertex that a click will close the shape. */
  closing: boolean;
}

/** The polygon being drawn: placed vertices, a rubber band to the cursor and a close target. */
export function DraftLayer({ points, cursor, px, closing }: Props) {
  if (points.length === 0) return null;
  const line = points.map((p) => p.join(",")).join(" ");
  const last = points[points.length - 1];
  return (
    <g style={{ pointerEvents: "none" }}>
      {points.length >= 3 && (
        <polygon points={line} fill="var(--accent)" fillOpacity={0.14} stroke="none" />
      )}
      <polyline
        points={line}
        fill="none"
        stroke="black"
        strokeOpacity={0.5}
        strokeWidth={4.5 * px}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <polyline
        points={line}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2 * px}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {cursor && (
        <line
          x1={last[0]}
          y1={last[1]}
          x2={closing ? points[0][0] : cursor[0]}
          y2={closing ? points[0][1] : cursor[1]}
          stroke="var(--accent)"
          strokeWidth={1.5 * px}
          strokeDasharray={`${6 * px} ${5 * px}`}
        />
      )}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={p[1]}
          r={(i === 0 ? (closing ? 8 : 6) : 3.5) * px}
          fill={i === 0 && closing ? "var(--accent)" : "var(--surface)"}
          stroke="var(--accent)"
          strokeWidth={2 * px}
        />
      ))}
    </g>
  );
}
