"use client";

import { memo } from "react";
import { centroid, zoneColor } from "@/lib/geometry";
import type { Zone } from "@/lib/types";

interface Props {
  zones: Zone[];
  selectedId: string | null;
  /** Image units per CSS pixel, so strokes and labels keep a constant on-screen size. */
  px: number;
  interactive: boolean;
  onSelect: (id: string) => void;
}

/**
 * Zones live on their own layer and are memoised: dragging an object re-renders the stage on
 * every pointer move, but never this layer.
 */
export const ZoneLayer = memo(function ZoneLayer({
  zones,
  selectedId,
  px,
  interactive,
  onSelect,
}: Props) {
  return (
    <g style={{ pointerEvents: interactive ? "auto" : "none" }}>
      {zones.map((zone, i) => {
        const color = zoneColor(i);
        const active = zone.id === selectedId;
        const points = zone.polygon.map((p) => p.join(",")).join(" ");
        const [lx, ly] = centroid(zone.polygon);
        return (
          <g key={zone.id}>
            <polygon
              points={points}
              fill={active ? color.fillActive : color.fill}
              stroke={color.stroke}
              strokeWidth={(active ? 2.5 : 1.75) * px}
              strokeLinejoin="round"
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelect(zone.id);
              }}
              style={{ cursor: "pointer" }}
            />
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={12 * px}
              fontWeight={600}
              fill="white"
              stroke="black"
              strokeOpacity={0.55}
              strokeWidth={3 * px}
              paintOrder="stroke"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {zone.id}
            </text>
          </g>
        );
      })}
    </g>
  );
});
