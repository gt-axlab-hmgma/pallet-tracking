"use client";

import { useEffect, useRef, useState } from "react";
import { ZoneLayer } from "./ZoneLayer";
import type { DemoState } from "@/lib/types";

interface Props {
  state: DemoState;
  selectedZone: string | null;
  onSelectZone: (id: string | null) => void;
}

/** The camera frame with zones on top. SVG user units are image pixels, so coordinates are 1:1. */
export function Stage({ state, selectedZone, onSelectZone }: Props) {
  const [w, h] = state.image_size;
  const svgRef = useRef<SVGSVGElement>(null);
  const [px, setPx] = useState(1);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const measure = () => setPx(w / Math.max(el.clientWidth, 1));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [w]);

  return (
    <div className="grid h-full place-items-center p-4">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${w} ${h}`}
        className="max-h-full max-w-full touch-none select-none bg-black shadow-[0_0_0_1px_var(--line)]"
        style={{ aspectRatio: `${w} / ${h}`, width: "100%", height: "auto" }}
        role="img"
        aria-label={`Camera ${state.camera_id} reference frame with ${state.zones.length} zones`}
        onPointerDown={() => onSelectZone(null)}
      >
        <image href="/api/frame" width={w} height={h} />
        <ZoneLayer
          zones={state.zones}
          selectedId={selectedZone}
          px={px}
          interactive
          onSelect={onSelectZone}
        />
      </svg>
    </div>
  );
}
