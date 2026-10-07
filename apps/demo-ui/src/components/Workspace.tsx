"use client";

import { useState } from "react";
import { useDemo } from "@/hooks/useDemo";
import { useEventStream } from "@/hooks/useEventStream";
import { Stage } from "./Stage";
import { TopBar } from "./TopBar";

export function Workspace() {
  const demo = useDemo();
  const status = useEventStream(demo.pushEvent, demo.backfillEvents);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const { state } = demo;

  return (
    <div className="grid h-full grid-rows-[40px_1fr]">
      <TopBar state={state} status={status} />
      <div className="grid min-h-0 grid-cols-[280px_1fr_320px]">
        <aside className="border-r border-line bg-surface" aria-label="Zones and objects" />
        <main className="min-h-0 min-w-0 bg-stage" aria-label="Camera view">
          {state ? (
            <Stage state={state} selectedZone={selectedZone} onSelectZone={setSelectedZone} />
          ) : (
            <p className="p-6 text-dim">{demo.notice?.text ?? "Loading…"}</p>
          )}
        </main>
        <aside className="border-l border-line bg-surface" aria-label="Event feed" />
      </div>
    </div>
  );
}
