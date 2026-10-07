export type Point = [number, number];
export type Box = [number, number, number, number]; // x1, y1, x2, y2 in image pixels

export interface Zone {
  id: string;
  polygon: Point[];
  min_dwell_s?: number;
}

export interface MockObject {
  id: number;
  object_type: string;
  box: Box;
  /** Zones the object's ground point is currently in, as decided by the backend. */
  zones: string[];
}

export interface DemoState {
  camera_id: string;
  device_id: string;
  image_size: [number, number];
  version: number;
  zones: Zone[];
  objects: MockObject[];
  object_types: string[];
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
