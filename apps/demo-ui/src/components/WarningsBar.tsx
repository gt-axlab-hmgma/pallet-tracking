/** Startup conditions the user should know about, such as zones not being saved. */
export function WarningsBar({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <ul className="border-b border-line bg-accent-soft px-4 py-2 text-text" aria-label="Warnings">
      {warnings.map((w) => (
        <li key={w}>{w}</li>
      ))}
    </ul>
  );
}
