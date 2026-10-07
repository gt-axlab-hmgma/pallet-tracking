"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { defaultBox } from "@/lib/geometry";
import { createThrottle } from "@/lib/throttle";
import type { Box, DemoState, MockObject, MutationResult, Zone, ZoneEvent } from "@/lib/types";

const MAX_EVENTS = 300;
const SEND_INTERVAL_MS = 60; // ~16 updates per second while dragging

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

export function useDemo() {
  const [state, setState] = useState<DemoState | null>(null);
  const [events, setEvents] = useState<ZoneEvent[]>([]);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [sinkError, setSinkError] = useState<string | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const stateRef = useRef<DemoState | null>(null);
  const zoneQueue = useRef<Promise<unknown>>(Promise.resolve());
  const senders = useRef(new Map<number, ReturnType<typeof createThrottle<[Box]>>>());
  const sendSeq = useRef(new Map<number, number>());

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const fail = useCallback((e: unknown) => setNotice({ kind: "error", text: messageOf(e) }), []);

  /** Moves that log nothing say nothing about storage, so they must not clear a warning. */
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

  const patchObject = useCallback((id: number, patch: Partial<MockObject>) => {
    setState((s) =>
      s ? { ...s, objects: s.objects.map((o) => (o.id === id ? { ...o, ...patch } : o)) } : s,
    );
  }, []);

  const addObject = useCallback(
    async (objectType: string) => {
      const s = stateRef.current;
      if (!s) return;
      try {
        const res = await api.addObject(objectType, defaultBox(s.image_size, s.objects.length));
        setState((cur) => (cur ? { ...cur, objects: [...cur.objects, res.object] } : cur));
        noteResult(res);
        return res.object.id;
      } catch (e) {
        fail(e);
      }
    },
    [fail, noteResult],
  );

  const sendBox = useCallback(
    async (id: number, box: Box) => {
      const seq = (sendSeq.current.get(id) ?? 0) + 1;
      sendSeq.current.set(id, seq);
      try {
        const res = await api.updateObject(id, { box });
        noteResult(res);
        // Responses can arrive out of order while dragging; only the latest one may set zones.
        if (sendSeq.current.get(id) === seq) patchObject(id, { zones: res.object.zones });
      } catch (e) {
        fail(e);
      }
    },
    [fail, noteResult, patchObject],
  );

  /** Called on every pointer move while dragging (throttled). Does not touch React state. */
  const dragObject = useCallback(
    (id: number, box: Box) => {
      let sender = senders.current.get(id);
      if (!sender) {
        sender = createThrottle((b: Box) => void sendBox(id, b), SEND_INTERVAL_MS);
        senders.current.set(id, sender);
      }
      sender.call(box);
    },
    [sendBox],
  );

  /** Called on pointer up: commit the final position locally and make sure it is sent last. */
  const dropObject = useCallback(
    (id: number, box: Box) => {
      patchObject(id, { box });
      senders.current.get(id)?.cancel(); // drop a pending trailing update; the final box wins
      void sendBox(id, box);
    },
    [patchObject, sendBox],
  );

  const setObjectType = useCallback(
    async (id: number, objectType: string) => {
      patchObject(id, { object_type: objectType });
      try {
        noteResult(await api.updateObject(id, { object_type: objectType }));
      } catch (e) {
        fail(e);
        await reload();
      }
    },
    [fail, noteResult, patchObject, reload],
  );

  const removeObject = useCallback(
    async (id: number) => {
      senders.current.get(id)?.cancel();
      senders.current.delete(id);
      setState((s) => (s ? { ...s, objects: s.objects.filter((o) => o.id !== id) } : s));
      try {
        noteResult(await api.deleteObject(id));
      } catch (e) {
        fail(e);
        await reload();
      }
    },
    [fail, noteResult, reload],
  );

  return {
    state,
    events,
    notice,
    sinkError,
    configError,
    dismissNotice: () => setNotice(null),
    pushEvent,
    backfillEvents,
    saveZones,
    addObject,
    dragObject,
    dropObject,
    setObjectType,
    removeObject,
  };
}
