import { groundPoint } from "@/lib/geometry";
import type { MockObject } from "@/lib/types";

interface Props {
  objects: MockObject[];
  selectedId: number | null;
  px: number;
  interactive: boolean;
  onStart: (id: number, kind: "move" | "resize", e: React.PointerEvent) => void;
  onMove: (e: React.PointerEvent) => void;
  onEnd: (e: React.PointerEvent) => void;
}

/**
 * Mock detections. Each box is drawn twice (dark under, light over) so it reads on any frame.
 * The marker at the bottom centre is the ground-contact point that decides zone membership; it
 * turns green while the backend says the object is inside a zone.
 */
export function ObjectLayer({
  objects,
  selectedId,
  px,
  interactive,
  onStart,
  onMove,
  onEnd,
}: Props) {
  return (
    <g style={{ pointerEvents: interactive ? "auto" : "none" }}>
      {objects.map((o) => {
        const [x1, y1, x2, y2] = o.box;
        const [gx, gy] = groundPoint(o.box);
        const selected = o.id === selectedId;
        const inside = o.zones.length > 0;
        const stroke = selected ? "var(--accent)" : "white";
        const label = `#${o.id} ${o.object_type}`;
        return (
          <g key={o.id} data-object={o.id}>
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="white"
              fillOpacity={0.06}
              stroke="black"
              strokeOpacity={0.55}
              strokeWidth={5 * px}
            />
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="none"
              stroke={stroke}
              strokeWidth={2 * px}
              style={{ cursor: "grab" }}
              onPointerDown={(e) => onStart(o.id, "move", e)}
              onPointerMove={onMove}
              onPointerUp={onEnd}
              onPointerCancel={onEnd}
            />
            {/* a transparent fill target so the whole box is draggable, not only its outline */}
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="transparent"
              style={{ cursor: "grab" }}
              onPointerDown={(e) => onStart(o.id, "move", e)}
              onPointerMove={onMove}
              onPointerUp={onEnd}
              onPointerCancel={onEnd}
            />
            <text
              x={x1}
              y={y1 - 6 * px}
              fontSize={11 * px}
              fontWeight={600}
              fill={selected ? "var(--accent)" : "white"}
              stroke="black"
              strokeOpacity={0.6}
              strokeWidth={3 * px}
              paintOrder="stroke"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {label}
            </text>
            <g style={{ pointerEvents: "none" }}>
              <circle
                cx={gx}
                cy={gy}
                r={8 * px}
                fill="none"
                stroke="black"
                strokeOpacity={0.55}
                strokeWidth={4 * px}
              />
              <circle
                cx={gx}
                cy={gy}
                r={8 * px}
                fill={inside ? "var(--join)" : "transparent"}
                fillOpacity={0.35}
                stroke={inside ? "var(--join)" : "white"}
                strokeWidth={2 * px}
              />
              <circle cx={gx} cy={gy} r={2.5 * px} fill={inside ? "var(--join)" : "white"} />
            </g>
            {selected && (
              <rect
                x={x2 - 5 * px}
                y={y2 - 5 * px}
                width={10 * px}
                height={10 * px}
                fill="var(--surface)"
                stroke="var(--accent)"
                strokeWidth={2 * px}
                style={{ cursor: "nwse-resize" }}
                onPointerDown={(e) => onStart(o.id, "resize", e)}
                onPointerMove={onMove}
                onPointerUp={onEnd}
                onPointerCancel={onEnd}
              />
            )}
          </g>
        );
      })}
    </g>
  );
}
