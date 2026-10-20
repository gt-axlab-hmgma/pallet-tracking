import { groundPoint } from "@/lib/geometry";
import type { TrackedBox } from "@/lib/types";

interface Props {
  objects: TrackedBox[];
  px: number;
}

/**
 * The pallets the tracker is following. Each box is drawn twice (dark under, light over) so it
 * reads on any frame. The marker at the bottom centre is the ground-contact point that decides
 * zone membership; it turns green while the backend says the pallet is inside a zone.
 */
export function ObjectLayer({ objects, px }: Props) {
  return (
    <g style={{ pointerEvents: "none" }}>
      {objects.map((o) => {
        const [x1, y1, x2, y2] = o.box;
        const [gx, gy] = groundPoint(o.box);
        const inside = o.zones.length > 0;
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
              stroke="white"
              strokeWidth={2 * px}
            />
            <text
              x={x1}
              y={y1 - 6 * px}
              fontSize={11 * px}
              fontWeight={600}
              fill="white"
              stroke="black"
              strokeOpacity={0.6}
              strokeWidth={3 * px}
              paintOrder="stroke"
              style={{ userSelect: "none" }}
            >
              #{o.id}
            </text>
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
        );
      })}
    </g>
  );
}
