"use client";

import { useCallback, useEffect, useState } from "react";
import { useDemo } from "@/hooks/useDemo";
import { useEventStream } from "@/hooks/useEventStream";
import { nextZoneId } from "@/lib/geometry";
import type { Point } from "@/lib/types";
import { EventFeed } from "./EventFeed";
import { NoticeBar } from "./NoticeBar";
import { Stage } from "./Stage";
import { type Mode, Toolbar } from "./Toolbar";
import { SourcePanel } from "./SourcePanel";
import { ZonePanel } from "./ZonePanel";
import { TopBar } from "./TopBar";
import { WarningsBar } from "./WarningsBar";

export function Workspace() {
  const demo = useDemo();
  const status = useEventStream(demo.pushEvent, demo.backfillEvents);
  const [mode, setMode] = useState<Mode>("select");
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const { state, saveZones } = demo;

  const createZone = useCallback(
    (polygon: Point[]) => {
      if (!state) return;
      const id = nextZoneId(state.zones);
      saveZones([...state.zones, { id, polygon, min_dwell_s: 0 }]);
      setSelectedZone(id);
      setMode("select");
    },
    [state, saveZones],
  );

  const editZone = useCallback(
    (id: string, polygon: Point[]) => {
      if (!state) return;
      saveZones(state.zones.map((z) => (z.id === id ? { ...z, polygon } : z)));
    },
    [state, saveZones],
  );

  const deleteZone = useCallback(
    (id: string) => {
      if (!state) return;
      saveZones(state.zones.filter((z) => z.id !== id));
      setSelectedZone((cur) => (cur === id ? null : cur));
    },
    [state, saveZones],
  );

  const renameZone = useCallback(
    (id: string, next: string): string | null => {
      if (!state) return null;
      if (!next) return "A zone needs a name.";
      if (state.zones.some((z) => z.id === next)) return `"${next}" is already used.`;
      saveZones(state.zones.map((z) => (z.id === id ? { ...z, id: next } : z)));
      setSelectedZone((cur) => (cur === id ? next : cur));
      return null;
    },
    [state, saveZones],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "v" || e.key === "V") setMode("select");
      else if (e.key === "z" || e.key === "Z") setMode("draw");
      else if ((e.key === "Delete" || e.key === "Backspace") && mode === "select" && selectedZone) {
        e.preventDefault();
        deleteZone(selectedZone);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, selectedZone, deleteZone]);

  return (
    <div className="grid h-full grid-rows-[40px_1fr]">
      <TopBar state={state} status={status} />
      <div className="grid min-h-0 grid-cols-[280px_1fr_320px]">
        <aside
          className="flex flex-col gap-4 border-r border-line bg-surface p-3"
          aria-label="Tools"
        >
          {state && (
            <SourcePanel
              source={state.source}
              canPlay={state.can_play}
              busy={demo.sourceBusy}
              onVideo={demo.loadVideo}
              onCamera={demo.connectCamera}
              onStop={demo.stopSource}
            />
          )}
          <Toolbar mode={mode} onChange={setMode} />
          {state && (
            <ZonePanel
              zones={state.zones}
              objects={state.objects}
              selectedId={selectedZone}
              onSelect={(id) => {
                setMode("select");
                setSelectedZone(id);
              }}
              onRename={renameZone}
              onDelete={deleteZone}
            />
          )}
        </aside>
        <main className="relative flex min-h-0 min-w-0 flex-col bg-stage" aria-label="Camera view">
          <NoticeBar notice={demo.notice} onDismiss={demo.dismissNotice} />
          <WarningsBar warnings={state?.warnings ?? []} />
          <div className="min-h-0 flex-1">
            {state ? (
              <Stage
                state={state}
                mode={mode}
                selectedZone={selectedZone}
                onSelectZone={setSelectedZone}
                onCreateZone={createZone}
                onEditZone={editZone}
              />
            ) : (
              <p className="p-6 text-dim">{demo.notice?.text ?? "Loading…"}</p>
            )}
          </div>
        </main>
        <aside className="min-h-0 border-l border-line bg-surface">
          <EventFeed
            events={demo.events}
            sink={state?.sink}
            sinkError={demo.sinkError}
            configError={demo.configError}
          />
        </aside>
      </div>
    </div>
  );
}
