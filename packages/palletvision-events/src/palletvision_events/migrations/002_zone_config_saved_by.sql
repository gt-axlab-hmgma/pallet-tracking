-- Zone versions are immutable history. Record who saved each one ('demo' for the local demo;
-- a real user or service once the web app owns zones).
ALTER TABLE zone_configs ADD COLUMN saved_by text NOT NULL DEFAULT 'unknown';
