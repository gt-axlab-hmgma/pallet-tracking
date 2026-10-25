import type { Point } from "@/lib/types";

interface Props {
  polygons: Point[][];
  px: number;
  /** Outline colour: one that contrasts with the paint the bays were found by. */
  color: string;
}

/** Proposed zones: dashed outlines that cannot be clicked, until the user adds or discards them. */
export function ProposalLayer({ polygons, px, color }: Props) {
  if (polygons.length === 0) return null;
  return (
    <g style={{ pointerEvents: "none" }} aria-hidden>
      {polygons.map((polygon, i) => {
        const points = polygon.map((p) => p.join(",")).join(" ");
        return (
          <g key={i}>
            <polygon
              points={points}
              fill={color}
              fillOpacity={0.1}
              stroke="black"
              strokeOpacity={0.5}
              strokeWidth={4.5 * px}
              strokeLinejoin="round"
            />
            <polygon
              points={points}
              fill="none"
              stroke={color}
              strokeWidth={2 * px}
              strokeDasharray={`${8 * px} ${6 * px}`}
              strokeLinejoin="round"
            />
            {polygon.map((p, j) => (
              <circle
                key={j}
                cx={p[0]}
                cy={p[1]}
                r={3.5 * px}
                fill="var(--surface)"
                stroke={color}
                strokeWidth={2 * px}
              />
            ))}
          </g>
        );
      })}
    </g>
  );
}
