import type { Point } from "@/lib/types";

interface Props {
  polygon: Point[];
  px: number;
  onStart: (index: number, e: React.PointerEvent) => void;
  onMove: (e: React.PointerEvent) => void;
  onEnd: (e: React.PointerEvent) => void;
}

/** Drag handles on the selected zone's corners. */
export function VertexHandles({ polygon, px, onStart, onMove, onEnd }: Props) {
  return (
    <g>
      {polygon.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={6 * px}
          fill="var(--surface)"
          stroke="var(--accent)"
          strokeWidth={2.5 * px}
          style={{ cursor: "grab" }}
          onPointerDown={(e) => onStart(i, e)}
          onPointerMove={onMove}
          onPointerUp={onEnd}
          onPointerCancel={onEnd}
        />
      ))}
    </g>
  );
}
