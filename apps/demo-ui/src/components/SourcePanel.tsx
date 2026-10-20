"use client";

import { useRef, useState } from "react";
import type { SourceStatus } from "@/lib/types";

interface Props {
  source: SourceStatus | null;
  canPlay: boolean;
  /** A label while a change is under way ("Sending clip.mp4…"), else null. */
  busy: string | null;
  onVideo: (file: File) => void;
  onCamera: (device: string) => void;
  onStop: () => void;
}

type Kind = "video" | "camera";

const field =
  "h-8 min-w-0 rounded border border-line bg-surface px-2 text-text placeholder:text-faint hover:bg-surface-2";
const primary =
  "h-8 rounded bg-accent px-3 font-medium text-accent-ink hover:brightness-110 disabled:opacity-50";

const DOT = {
  playing: "bg-join",
  ended: "bg-faint",
  error: "bg-leave",
} as const;

/** Pick what to watch: a video clip from this computer, or a camera attached to the API machine. */
export function SourcePanel({ source, canPlay, busy, onVideo, onCamera, onStop }: Props) {
  const [kind, setKind] = useState<Kind>("video");
  const [device, setDevice] = useState("0");
  const fileInput = useRef<HTMLInputElement>(null);
  const disabled = !canPlay || busy !== null;

  return (
    <section aria-labelledby="source-h" className="flex flex-col gap-2">
      <h2
        id="source-h"
        className="font-display text-[length:var(--text-micro)] font-semibold uppercase tracking-[0.08em] text-faint"
      >
        Source
      </h2>

      {!canPlay && (
        <p className="text-dim">
          No model is loaded. Start <code>pv-demo</code> with <code>--weights model.onnx</code> to
          track a video or camera.
        </p>
      )}

      <div
        role="radiogroup"
        aria-label="Source type"
        className="grid grid-cols-2 gap-px rounded border border-line bg-line"
      >
        {(
          [
            ["video", "Video clip"],
            ["camera", "Camera"],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            type="button"
            role="radio"
            aria-checked={kind === k}
            onClick={() => setKind(k)}
            className={`h-8 font-medium first:rounded-l last:rounded-r ${
              kind === k
                ? "bg-accent text-accent-ink"
                : "bg-surface text-dim hover:bg-surface-2 hover:text-text"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {kind === "video" ? (
        <>
          <input
            ref={fileInput}
            type="file"
            accept="video/*,.mkv"
            hidden
            data-testid="video-file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = ""; // choosing the same file again must still fire
              if (file) onVideo(file);
            }}
          />
          <button
            type="button"
            disabled={disabled}
            onClick={() => fileInput.current?.click()}
            className={primary}
          >
            Choose a video…
          </button>
          <p className="text-faint">Stays on this computer. mp4, mov, avi, mkv or webm.</p>
        </>
      ) : (
        <>
          <form
            className="flex gap-1.5"
            onSubmit={(e) => {
              e.preventDefault();
              if (device.trim()) onCamera(device.trim());
            }}
          >
            <input
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              aria-label="Camera index or stream URL"
              placeholder="0 or rtsp://…"
              autoComplete="off"
              spellCheck={false}
              className={`${field} flex-1`}
            />
            <button type="submit" disabled={disabled || !device.trim()} className={primary}>
              Connect
            </button>
          </form>
          <p className="text-faint">
            0 is the first camera on the machine running the API. IP cameras: an rtsp:// URL.
          </p>
        </>
      )}

      {busy && (
        <p className="text-dim" role="status">
          {busy}
        </p>
      )}

      {source && !busy && (
        <div className="flex items-center gap-2 rounded border border-line bg-surface-2 px-2 py-1.5">
          <span aria-hidden className={`size-2 shrink-0 rounded-full ${DOT[source.state]}`} />
          <div className="min-w-0 flex-1">
            <p className="truncate font-medium" title={source.name}>
              {source.name}
            </p>
            <p className={source.state === "error" ? "text-danger" : "text-dim"} role="status">
              {source.state === "playing" && "Tracking"}
              {source.state === "ended" && "Finished"}
              {source.state === "error" && (source.message ?? "Stopped")}
            </p>
          </div>
          <button
            type="button"
            onClick={onStop}
            className="h-7 shrink-0 rounded border border-line px-2 text-dim hover:bg-surface hover:text-text"
          >
            Stop
          </button>
        </div>
      )}
    </section>
  );
}
