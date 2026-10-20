"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { distance, moveBox, resizeBox } from "@/lib/geometry";
import type { Box, DemoState, Point } from "@/lib/types";
import { DraftLayer } from "./DraftLayer";
import type { Mode } from "./Toolbar";
import { ObjectLayer } from "./ObjectLayer";
import { VertexHandles } from "./VertexHandles";
import { ZoneLayer } from "./ZoneLayer";

interface Props {
  state: DemoState;
  mode: Mode;
  selectedZone: string | null;
  onSelectZone: (id: string | null) => void;
  onCreateZone: (polygon: Point[]) => void;
  onEditZone: (id: string, polygon: Point[]) => void;
  selectedObject: number | null;
  onSelectObject: (id: number | null) => void;
  liveDrag: boolean;
  onDragObject: (id: number, box: Box) => void;
  onDropObject: (id: number, box: Box) => void;
}

const CLOSE_RADIUS_PX = 12; // on-screen distance within which a click closes the polygon

function isTyping(target: EventTarget | null) {
  return target instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName);
}

/** The camera frame with zones on top. SVG user units are image pixels, so coordinates are 1:1. */
export function Stage({
  state,
  mode,
  selectedZone,
  onSelectZone,
  onCreateZone,
  onEditZone,
  selectedObject,
  onSelectObject,
  liveDrag,
  onDragObject,
  onDropObject,
}: Props) {
  const [w, h] = state.image_size;
  const svgRef = useRef<SVGSVGElement>(null);
  const [px, setPx] = useState(1);
  const [draft, setDraft] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [vertexDrag, setVertexDrag] = useState<{
    zoneId: string;
    index: number;
    point: Point;
  } | null>(null);
  const objDrag = useRef<{
    id: number;
    kind: "move" | "resize";
    startPointer: Point;
    startBox: Box;
  } | null>(null);
  const [liveBox, setLiveBox] = useState<{ id: number; box: Box } | null>(null);
  const drawing = mode === "draw";

  // While a corner is being dragged the zone is shown with its live shape.
  const zones = useMemo(
    () =>
      state.zones.map((z) =>
        vertexDrag && z.id === vertexDrag.zoneId
          ? {
              ...z,
              polygon: z.polygon.map((p, i) => (i === vertexDrag.index ? vertexDrag.point : p)),
            }
          : z,
      ),
    [state.zones, vertexDrag],
  );
  // The dragged object is shown at its live position without touching the zone layer.
  const objects = useMemo(
    () =>
      state.objects.map((o) => (liveBox && o.id === liveBox.id ? { ...o, box: liveBox.box } : o)),
    [state.objects, liveBox],
  );
  const selected = zones.find((z) => z.id === selectedZone) ?? null;

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
      onSelectObject(null);
      return;
    }
    const p = toImage(e);
    if (nearFirst(p)) closeDraft(draft);
    else setDraft((d) => [...d, p]);
  };

  const endVertexDrag = (e: React.PointerEvent) => {
    if (!vertexDrag) return;
    const zone = state.zones.find((z) => z.id === vertexDrag.zoneId);
    setVertexDrag(null);
    if (!zone) return;
    const [x, y] = toImage(e);
    onEditZone(
      zone.id,
      zone.polygon.map((p, i) =>
        i === vertexDrag.index ? ([Math.round(x), Math.round(y)] as Point) : p,
      ),
    );
  };

  const startObjectDrag = (id: number, kind: "move" | "resize", e: React.PointerEvent) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const obj = state.objects.find((o) => o.id === id);
    if (!obj) return;
    onSelectObject(id);
    onSelectZone(null);
    objDrag.current = { id, kind, startPointer: toImage(e), startBox: obj.box };
  };

  const moveObjectDrag = (e: React.PointerEvent) => {
    const d = objDrag.current;
    if (!d) return;
    const p = toImage(e);
    let box: Box;
    if (d.kind === "resize") {
      box = resizeBox(d.startBox, p, 24 * px);
    } else {
      // keep the whole box inside the frame
      const dx = Math.min(Math.max(p[0] - d.startPointer[0], -d.startBox[0]), w - d.startBox[2]);
      const dy = Math.min(Math.max(p[1] - d.startPointer[1], -d.startBox[1]), h - d.startBox[3]);
      box = moveBox(d.startBox, dx, dy);
    }
    setLiveBox({ id: d.id, box });
    if (liveDrag) onDragObject(d.id, box);
  };

  const endObjectDrag = () => {
    const d = objDrag.current;
    objDrag.current = null;
    if (!d || !liveBox || liveBox.id !== d.id) return setLiveBox(null);
    setLiveBox(null);
    onDropObject(d.id, liveBox.box);
  };

  const hint = !drawing
    ? state.live
      ? "Live: the boxes are tracks from the video. Click a zone to select it, or press Z to draw one."
      : selectedObject !== null
        ? "Drag the box to move it. The dot at its bottom centre is what zones test; the square resizes."
        : selected
          ? "Drag a corner to reshape the zone. Delete removes it, double-click its name to rename."
          : "Click a zone to select it, or drag a mock object across one. Press Z to draw."
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
          {state.live ? (
            // a motion JPEG only plays in an <img>, and SVG <image> would show one frame
            <foreignObject width={w} height={h} style={{ pointerEvents: "none" }}>
              {/* next/image cannot play a motion JPEG */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/api/live/stream"
                alt=""
                style={{ display: "block", width: "100%", height: "100%" }}
              />
            </foreignObject>
          ) : (
            <image href="/api/frame" width={w} height={h} />
          )}
          <ZoneLayer
            zones={zones}
            selectedId={selectedZone}
            px={px}
            interactive={!drawing}
            onSelect={onSelectZone}
          />
          {!drawing && selected && (
            <VertexHandles
              polygon={selected.polygon}
              px={px}
              onStart={(index, e) => {
                e.stopPropagation();
                e.currentTarget.setPointerCapture(e.pointerId);
                setVertexDrag({
                  zoneId: selected.id,
                  index,
                  point: selected.polygon[index],
                });
              }}
              onMove={(e) => {
                if (vertexDrag)
                  setVertexDrag({
                    ...vertexDrag,
                    point: toImage(e),
                  });
              }}
              onEnd={endVertexDrag}
            />
          )}
          <ObjectLayer
            objects={objects}
            selectedId={selectedObject}
            px={px}
            interactive={!drawing && !state.live}
            onStart={startObjectDrag}
            onMove={moveObjectDrag}
            onEnd={endObjectDrag}
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
