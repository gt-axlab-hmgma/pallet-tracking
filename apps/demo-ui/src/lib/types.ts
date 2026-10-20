export type Point = [number, number];
export type Box = [number, number, number, number]; // x1, y1, x2, y2 in image pixels

export interface Zone {
  id: string;
  polygon: Point[];
  min_dwell_s?: number;
}

/** A pallet the tracker is following in the current frame. */
export interface TrackedBox {
  id: number;
  box: Box;
  /** Zones the pallet's ground point is currently in, as decided by the backend. */
  zones: string[];
}

/** The video or camera being played. `message` explains an error. */
export interface SourceStatus {
  kind: "video" | "camera";
  name: string;
  state: "playing" | "ended" | "error";
  message: string | null;
}

export interface DemoState {
  camera_id: string;
  device_id: string;
  image_size: [number, number];
  /** Changes whenever the frame zones are drawn on changes (a new source), to refresh the image. */
  frame_version: number;
  /** False when the API was started without a model, so no video or camera can be played. */
  can_play: boolean;
  source: SourceStatus | null;
  version: number;
  zones: Zone[];
  objects: TrackedBox[];
  sink: string;
  warnings: string[];
}

export interface ZoneEvent {
  event_id: string;
  device_id: string;
  camera_id: string;
  seq: number;
  event_type: "enter" | "exit";
  zone_id: string;
  object_type: string;
  track_id: number;
  source: string;
  device_ts: string;
  config_version?: number;
  message: string;
}

/** Every mutation reports whether storing its events failed (for example Postgres is down). */
export interface MutationResult {
  events: ZoneEvent[];
  sink_error: string | null;
  /** The zone version could not be stored in the database (zones still saved locally). */
  config_error?: string | null;
}
