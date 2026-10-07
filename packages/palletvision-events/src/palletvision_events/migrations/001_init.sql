-- Devices (Jetson Nanos) and the cameras attached to them.
CREATE TABLE devices (
    device_id     text PRIMARY KEY,
    name          text,
    registered_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE cameras (
    camera_id   text PRIMARY KEY,
    device_id   text,
    description text
);

-- Zone polygons are versioned per camera: cameras get bumped and zones get redrawn.
CREATE TABLE zone_configs (
    config_id  bigserial PRIMARY KEY,
    camera_id  text NOT NULL,
    version    integer NOT NULL,
    image_size integer[] NOT NULL,
    zones      jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (camera_id, version)
);

-- The hot table. No foreign keys on purpose: every insert stays cheap and a late-registered
-- device never blocks ingestion. event_id is generated on the device, which makes retries
-- idempotent (INSERT ... ON CONFLICT DO NOTHING). The partition key has to be in the PK.
CREATE TABLE events (
    event_id       uuid        NOT NULL,
    device_id      text        NOT NULL,
    camera_id      text        NOT NULL,
    seq            bigint      NOT NULL,
    event_type     text        NOT NULL CHECK (event_type IN ('enter', 'exit')),
    zone_id        text        NOT NULL,
    object_type    text        NOT NULL,
    track_id       bigint      NOT NULL,
    config_version integer,
    source         text        NOT NULL CHECK (source IN ('model', 'mock')),
    device_ts      timestamptz NOT NULL,
    received_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, device_ts)
) PARTITION BY RANGE (device_ts);

CREATE INDEX events_camera_ts ON events (camera_id, device_ts);
CREATE INDEX events_zone_ts ON events (zone_id, device_ts);
CREATE INDEX events_device_seq ON events (device_id, seq);

-- Creates the monthly partition that holds the given timestamp, if it is missing.
CREATE FUNCTION pv_ensure_events_partition(ts timestamptz) RETURNS void AS $$
DECLARE
    start_month date := date_trunc('month', ts AT TIME ZONE 'UTC')::date;
    end_month   date := (start_month + interval '1 month')::date;
    part_name   text := format('events_%s', to_char(start_month, 'YYYY_MM'));
BEGIN
    IF to_regclass(part_name) IS NULL THEN
        BEGIN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF events FOR VALUES FROM (%L) TO (%L)',
                part_name,
                start_month::timestamp AT TIME ZONE 'UTC',
                end_month::timestamp AT TIME ZONE 'UTC'
            );
        EXCEPTION WHEN duplicate_table THEN
            NULL;  -- another writer created it first
        END;
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT pv_ensure_events_partition(now());
SELECT pv_ensure_events_partition(now() + interval '1 month');

-- Dwell time is derived, never stored, so it cannot drift from the events.
-- Each enter is paired with the next event for the same track and zone; an open visit has
-- a NULL t_exit.
CREATE VIEW zone_visits AS
SELECT camera_id, device_id, track_id, object_type, zone_id,
       device_ts AS t_enter,
       CASE WHEN next_type = 'exit' THEN next_ts END AS t_exit,
       CASE WHEN next_type = 'exit'
            THEN EXTRACT(EPOCH FROM (next_ts - device_ts)) END AS dwell_s
FROM (
    SELECT camera_id, device_id, track_id, object_type, zone_id, event_type, device_ts,
           LEAD(event_type) OVER w AS next_type,
           LEAD(device_ts) OVER w AS next_ts
    FROM events
    WINDOW w AS (PARTITION BY device_id, camera_id, track_id, zone_id ORDER BY device_ts, seq)
) paired
WHERE event_type = 'enter';
