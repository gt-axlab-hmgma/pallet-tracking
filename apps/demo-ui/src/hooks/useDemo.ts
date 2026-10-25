"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { MAX_VIDEO_BYTES, formatSize } from "@/lib/limits";
import type { DemoState, MutationResult, Point, Zone, ZoneEvent, ZoneProposal } from "@/lib/types";

const MAX_EVENTS = 300;
const POLL_MS = 200; // how often the tracked pallets and the source status are refreshed

export interface Notice {
  kind: "error" | "warn";
  text: string;
}

const messageOf = (e: unknown) => (e instanceof ApiError ? e.message : "Something went wrong.");

/** Merge by event_id, newest first. Safe to call with overlapping batches (stream plus backfill). */
export function mergeEvents(current: ZoneEvent[], incoming: ZoneEvent[]): ZoneEvent[] {
  const seen = new Set(current.map((e) => e.event_id));
  const fresh = incoming.filter((e) => !seen.has(e.event_id));
  if (fresh.length === 0) return current;
  return [...fresh.reverse(), ...current].slice(0, MAX_EVENTS);
}

/**
 * Fold a polled state into the current one. It takes the tracked pallets, the source status and
 * the warnings, and keeps the zones: the user may be editing them, and a slow poll must not undo
 * that. If the frame itself changed (another source), the whole fresh state is taken.
 */
export function mergeLive(current: DemoState, fresh: DemoState): DemoState {
  if (fresh.frame_version !== current.frame_version) return fresh;
  return { ...current, objects: fresh.objects, source: fresh.source, warnings: fresh.warnings };
}

export function useDemo() {
  const [state, setState] = useState<DemoState | null>(null);
  const [events, setEvents] = useState<ZoneEvent[]>([]);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [sinkError, setSinkError] = useState<string | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [sourceBusy, setSourceBusy] = useState<string | null>(null);
  const [proposal, setProposal] = useState<ZoneProposal | null>(null);
  const [proposing, setProposing] = useState(false);
  const zoneQueue = useRef<Promise<unknown>>(Promise.resolve());

  const fail = useCallback((e: unknown) => setNotice({ kind: "error", text: messageOf(e) }), []);

  /** Saving zones that logs nothing says nothing about storage, so it must not clear a warning. */
  const noteResult = useCallback((r: MutationResult, savedZones = false) => {
    if (r.events.length > 0 || r.sink_error) setSinkError(r.sink_error);
    if (savedZones || r.events.length > 0 || r.config_error) setConfigError(r.config_error ?? null);
  }, []);

  const reload = useCallback(async () => {
    try {
      setState(await api.getState());
    } catch (e) {
      fail(e);
    }
  }, [fail]);

  useEffect(() => {
    // Initial load from the backend; the setState calls happen after the awaited requests.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void reload();
  }, [reload]);

  const loaded = state !== null;
  useEffect(() => {
    if (!loaded) return;
    const id = setInterval(() => {
      api
        .getState()
        .then((fresh) => setState((s) => (s ? mergeLive(s, fresh) : s)))
        .catch(() => {}); // a missed refresh is replaced by the next one
    }, POLL_MS);
    return () => clearInterval(id);
  }, [loaded]);

  const backfillEvents = useCallback(async () => {
    try {
      const recent = await api.recentEvents();
      setEvents((cur) => mergeEvents(cur, recent));
    } catch {}
  }, []);

  const pushEvent = useCallback((e: ZoneEvent) => setEvents((cur) => mergeEvents(cur, [e])), []);

  /** Replace the full zone list. Optimistic; the backend validates and the UI rolls back on 422. */
  const saveZones = useCallback(
    (zones: Zone[]) => {
      setState((s) => (s ? { ...s, zones } : s));
      zoneQueue.current = zoneQueue.current.then(async () => {
        try {
          const res = await api.saveZones(zones);
          setState((s) =>
            s ? { ...s, zones: res.zones, version: res.version, objects: res.objects } : s,
          );
          noteResult(res, true);
          setNotice(null);
        } catch (e) {
          fail(e);
          await reload(); // roll back to what the backend actually has
        }
      });
    },
    [fail, noteResult, reload],
  );

  /** Run a source change. The reply is the whole new state: new frame, no tracks, new status. */
  const changeSource = useCallback(
    async (label: string, job: () => Promise<DemoState>) => {
      setSourceBusy(label);
      try {
        setState(await job());
        setNotice(null);
      } catch (e) {
        fail(e);
      } finally {
        setSourceBusy(null);
      }
    },
    [fail],
  );

  const loadVideo = useCallback(
    (file: File) => {
      if (file.size > MAX_VIDEO_BYTES) {
        setNotice({
          kind: "error",
          text: `${file.name} is ${formatSize(file.size)}; the limit is ${formatSize(MAX_VIDEO_BYTES)}. Cut it shorter, or start pv-demo with --source ${file.name} to play it from disk.`,
        });
        return Promise.resolve();
      }
      return changeSource(`Sending ${file.name} (${formatSize(file.size)})…`, () =>
        api.uploadVideo(file),
      );
    },
    [changeSource],
  );
  const connectCamera = useCallback(
    (device: string) => changeSource("Connecting…", () => api.connectCamera(device)),
    [changeSource],
  );
  const stopSource = useCallback(
    () => changeSource("Stopping…", () => api.stopSource()),
    [changeSource],
  );

  /** Ask the API for zones from the painted floor lines. They are only shown until accepted. */
  const proposeZones = useCallback(
    async (point?: Point) => {
      setProposing(true);
      try {
        const found = await api.autoZones(point);
        if (found.polygons.length === 0) {
          setProposal(null);
          setNotice({
            kind: "warn",
            text: point
              ? "No bays were found in that color. Click a different spot on the line, or draw the zones by hand (press Z)."
              : "No painted floor lines were found. Try Pick line color and click on a painted line, or draw the zones by hand (press Z).",
          });
        } else {
          setProposal(found);
          setNotice(null);
        }
      } catch (e) {
        fail(e);
      } finally {
        setProposing(false);
      }
    },
    [fail],
  );
  const discardProposal = useCallback(() => setProposal(null), []);

  return {
    state,
    proposal,
    proposing,
    proposeZones,
    discardProposal,
    events,
    notice,
    sinkError,
    configError,
    sourceBusy,
    dismissNotice: () => setNotice(null),
    pushEvent,
    backfillEvents,
    saveZones,
    loadVideo,
    connectCamera,
    stopSource,
  };
}
