# After inference: from bounding box to stored event

What happens once the detector has drawn a box. The first diagram is the local demo; the second is the production setup with many Nanos and cameras. Boxes and lines drawn dashed are **not built yet**.

Related: [README](../README.md) for how to run things, [design.md](design.md) for the full design, especially *Events and distributed storage*.

## Local: the demo on a laptop

The mock boxes you drag in the browser stand in for the detector and tracker.

```mermaid
flowchart LR
  subgraph Browser
    UI["Next.js UI :3000<br/>draw zones, drag mock boxes"]
  end

  subgraph API["pv-demo API :8000"]
    ENG["Zone engine<br/>ground point inside polygon<br/>hysteresis"]
    FAC["Event factory<br/>event_id, seq, timestamp"]
    SINK{{"--sink"}}
  end

  UI -- "/api proxy: move box, save zones" --> ENG
  ENG -- "enter / exit" --> FAC
  FAC --> SINK

  SINK --> J["jsonl file<br/>runs/demo/events.jsonl"]
  SINK --> PG[("Postgres<br/>events, zone_configs")]
  SINK --> OB["Outbox<br/>SQLite file"]
  OB --> SYNC["pv-sync"]
  SYNC --> PG
  SINK --> OUT["stdout"]

  FAC -. "live event log (SSE)" .-> UI
  ENG -. "save zone versions<br/>(postgres sink only)" .-> PG
  PG -. "latest zones at startup" .-> ENG
```

- **`--sink`** picks where events are stored: `jsonl` (default), `postgres`, `outbox` or `stdout`. A *sink* is simply the destination an event is handed to; the engine does not care which one it is.
- **The browser's event log** shows events whatever the sink is, because events are also pushed straight to the page.
- **Zones** are saved to and loaded from Postgres only with `--sink postgres`. With any other sink they exist for the session only.
- **`outbox` + `pv-sync`** mimics a Jetson: events are queued locally and forwarded to Postgres later, once each.

## Production: many Nanos and cameras

```mermaid
flowchart TB
  subgraph Central["Central server (lab-controlled)"]
    WEB["Web app<br/>draw zones on a camera frame"]
    API["Ingest and config API"]
    PG[("Postgres<br/>events: monthly partitions<br/>zone_configs: immutable versions<br/>devices, cameras")]
    DASH["Dashboards and queries<br/>zone_visits dwell view"]
  end

  subgraph N1["Jetson Nano 1"]
    CA["Camera A"] --> DET1["Detector + tracker"]
    CB["Camera B"] --> DET1
    DET1 --> ZE1["Zone engine<br/>local copy of zones"]
    ZE1 -- "event" --> OB1["Outbox<br/>SQLite, survives power loss"]
    OB1 --> SY1["Sync agent<br/>pv-sync"]
  end

  subgraph N2["Jetson Nano 2"]
    CC["Camera C"] --> DET2["Detector + tracker"]
    DET2 --> ZE2["Zone engine<br/>local copy of zones"]
    ZE2 -- "event" --> OB2["Outbox"]
    OB2 --> SY2["Sync agent"]
  end

  NN["... more Nanos"]

  SY1 -- "batches, retried, in order" --> API
  SY2 -- "batches, retried, in order" --> API
  NN --> API
  API -- "insert, duplicates ignored" --> PG
  PG --> DASH
  WEB <--> API

  API -. "zone version N+1 (Nano pulls)" .-> SY1
  API -. "zone version N+1" .-> SY2
  SY1 -. "apply new zones" .-> ZE1
  SY2 -. "apply new zones" .-> ZE2
  DET1 -. "camera snapshot on demand" .-> API

  classDef planned stroke-dasharray: 5 5;
  class API,WEB,NN planned;
```

- **A Nano never talks to Postgres.** It writes each event to its local outbox, and the sync agent sends batches to the central API. A failed send is retried, so an outage delays events but loses none.
- **Resends are harmless.** Every event has an id made on the Nano, so the database ignores duplicates. A per-device `seq` counter exposes any gaps.
- **Zones flow the other way.** Someone draws zones in the web app, the API stores a new immutable version, and each Nano pulls the version for its cameras and applies it. The Nano keeps its last applied version, so it keeps working offline.
- **Why Nanos pull:** factory networks usually allow outbound connections but not inbound ones, and a pull needs no open ports or device addresses.

## What exists today

| Piece | Status |
| --- | --- |
| Zone engine (plain Python, golden-tested for a later C++ port) | Built |
| Event records, outbox, `pv-sync`, Postgres `events` table, `zone_visits` view | Built |
| Versioned `zone_configs` in Postgres, loaded and saved by the demo | Built |
| Browser demo: draw zones, drag mock boxes, live event log | Built |
| `pv-sync` writing events to Postgres | Built, directly to the database for now |
| Ingest and config API between Nanos and Postgres | Not built |
| Production web app with a real camera frame | Not built |
| Nano pulling and applying zone versions, acknowledgements | Not built |
| Camera snapshot on demand, live preview | Not built |
| Real detector and tracker feeding the engine | Not built |
