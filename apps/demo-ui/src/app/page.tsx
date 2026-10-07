import { ThemeToggle } from "@/components/ThemeToggle";

export default function Home() {
  return (
    <div className="grid h-full grid-rows-[40px_1fr]">
      <header className="flex items-center gap-4 border-b border-line bg-surface px-4">
        <h1 className="font-display text-[length:var(--text-lead)] font-semibold tracking-tight">
          Pallet zones
        </h1>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>
      <div className="grid min-h-0 grid-cols-[280px_1fr_320px]">
        <aside className="border-r border-line bg-surface" aria-label="Zones and objects" />
        <main className="min-w-0 bg-stage" aria-label="Camera view" />
        <aside className="border-l border-line bg-surface" aria-label="Event feed" />
      </div>
    </div>
  );
}
