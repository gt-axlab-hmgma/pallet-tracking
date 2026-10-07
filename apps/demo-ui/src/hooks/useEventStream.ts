"use client";

import { useEffect, useRef, useState } from "react";
import type { ZoneEvent } from "@/lib/types";

export type StreamStatus = "connecting" | "live" | "reconnecting";

/**
 * Subscribes to the backend's server-sent events. EventSource reconnects by itself; after every
 * (re)connect `onOpen` is called so the caller can backfill anything missed in the gap.
 */
export function useEventStream(onEvent: (e: ZoneEvent) => void, onOpen: () => void) {
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const handlers = useRef({ onEvent, onOpen });

  useEffect(() => {
    handlers.current = { onEvent, onOpen };
  });

  useEffect(() => {
    const source = new EventSource("/api/events/stream");
    source.onopen = () => {
      setStatus("live");
      handlers.current.onOpen();
    };
    source.onerror = () => setStatus("reconnecting");
    source.addEventListener("zone_event", (msg) => {
      handlers.current.onEvent(JSON.parse((msg as MessageEvent<string>).data) as ZoneEvent);
    });
    return () => source.close();
  }, []);

  return status;
}
