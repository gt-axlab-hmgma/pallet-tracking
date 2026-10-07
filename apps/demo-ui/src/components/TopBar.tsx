import type { StreamStatus } from "@/hooks/useEventStream";
import type { DemoState } from "@/lib/types";
import { StatusPill } from "./StatusPill";
import { ThemeToggle } from "./ThemeToggle";

export function TopBar({ state, status }: { state: DemoState | null; status: StreamStatus }) {
  return (
    <header className="flex items-center gap-5 border-b border-line bg-surface px-4">
      <h1 className="font-display text-[length:var(--text-lead)] font-semibold tracking-tight">
        Pallet zones
      </h1>
      {state && (
        <dl className="flex items-center gap-4 text-dim">
          <div className="flex gap-1.5">
            <dt className="text-faint">Camera</dt>
            <dd className="font-medium text-text">{state.camera_id}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-faint">Device</dt>
            <dd className="font-medium text-text">{state.device_id}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-faint">Events to</dt>
            <dd className="font-medium text-text">{state.sink}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-faint">Zones v</dt>
            <dd className="font-medium text-text">{state.version}</dd>
          </div>
        </dl>
      )}
      <div className="ml-auto flex items-center gap-4">
        <StatusPill status={status} />
        <ThemeToggle />
      </div>
    </header>
  );
}
