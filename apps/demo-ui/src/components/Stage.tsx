"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { distance } from "@/lib/geometry";
import type { DemoState, Point } from "@/lib/types";
import { DraftLayer } from "./DraftLayer";
import type { Mode } from "./Toolbar";
import { ZoneLayer } from "./ZoneLayer";

interface Props {
  state: DemoState;
  mode: Mode;
  selectedZone: string | null;
  onSelectZone: (id: string | null) => void;
  onCreateZone: (polygon: Point[]) => void;
}

const CLOSE_RADIUS_PX = 12; // on-screen distance within which a click closes the polygon

function isTyping(target: EventTarget | null) {
  return target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName);
}

/** The camera frame with zones on top. SVG user units are image pixels, so coordinates are 1:1. */
export function Stage({ state, mode, selectedZone, onSelectZone, onCreateZone }: Props) {
  const [w, h] = state.image_size;
  const svgRef = useRef<SVGSVGElement>(null);
  const [px, setPx] = useState(1);
  const [draft, setDraft] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const drawing = mode === "draw";

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const measure = () => setPx(w / Math.max(el.clientWidth, 1));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [w]);

  /** Client pixels to image pixels, clamped to the frame. */
  const toImage = useCallback(
    (e: { clientX: number; clientY: number }): Point => {
      const svg = svgRef.current!;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
      return [Math.min(Math.max(p.x, 0), w), Math.min(Math.max(p.y, 0), h)];
    },
    [w, h],
  );

  const closeDraft = useCallback(
    (points: Point[]) => {
      if (points.length < 3) return;
      onCreateZone(points.map(([x, y]) => [Math.round(x), Math.round(y)] as Point));
      setDraft([]);
      setCursor(null);
    },
    [onCreateZone],
  );

  // Leaving draw mode abandons the unfinished polygon.
  useEffect(() => {
    if (!drawing) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDraft([]);
      setCursor(null);
    }
  }, [drawing]);

  useEffect(() => {
    if (!drawing) return;
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      if (e.key === "Escape") setDraft([]);
      else if (e.key === "Backspace") setDraft((d) => d.slice(0, -1));
      else if (e.key === "Enter") closeDraft(draft);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawing, draft, closeDraft]);

  const nearFirst = (p: Point) =>
    draft.length >= 3 && distance(p, draft[0]) <= CLOSE_RADIUS_PX * px;

  const onPointerDown = (e: React.PointerEvent) => {
    if (!drawing) {
      onSelectZone(null);
      return;
    }
    const p = toImage(e);
    if (nearFirst(p)) closeDraft(draft);
    else setDraft((d) => [...d, p]);
  };

  const hint = !drawing
    ? "Click a zone to select it. Press Z to draw a new one."
    : draft.length === 0
      ? "Click to place the first corner. Add as many corners as the floor area needs."
      : draft.length < 3
        ? "Keep clicking to add corners. Backspace removes the last one, Esc cancels."
        : "Click the first point or press Enter to close. Backspace undoes, Esc cancels.";

  return (
    <div className="relative grid h-full grid-rows-[1fr_auto]">
      <div className="grid min-h-0 place-items-center p-4">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${w} ${h}`}
          className="max-h-full max-w-full touch-none select-none bg-black shadow-[0_0_0_1px_var(--line)]"
          style={{
            aspectRatio: `${w} / ${h}`,
            width: "100%",
            height: "auto",
            cursor: drawing ? "crosshair" : "default",
          }}
          role="img"
          aria-label={`Camera ${state.camera_id} reference frame with ${state.zones.length} zones`}
          onPointerDown={onPointerDown}
          onPointerMove={(e) => drawing && setCursor(toImage(e))}
          onPointerLeave={() => setCursor(null)}
        >
          <image href="/api/frame" width={w} height={h} />
          <ZoneLayer
            zones={state.zones}
            selectedId={selectedZone}
            px={px}
            interactive={!drawing}
            onSelect={onSelectZone}
          />
          <DraftLayer
            points={draft}
            cursor={cursor}
            px={px}
            closing={cursor !== null && nearFirst(cursor)}
          />
        </svg>
      </div>
      <p className="border-t border-line bg-surface px-4 py-2 text-dim" aria-live="polite">
        {hint}
      </p>
    </div>
  );
}
