# HMGMA Pallet Vision

Pallet detection, tracking and zone analytics for HMGMA. A clean-room PyTorch detector is trained per frame; a tracker and zone engine turn detections into events such as "an object of type `pallet_type_a` joined `zone_2`". See [docs/design.md](docs/design.md) for the full design.

**Status**

- Working: training, ONNX export, benchmarking and video/webcam inference with a placeholder `TinyDetector`; the zone engine; the event pipeline (local queue, Postgres); and a browser demo where you draw zones and drag mock bounding boxes across them.
- Not yet: the real YOLO26 detector (M1), a tracker fed by real detections, the HTTP ingest service for many Jetsons, and the C++/TensorRT runtime.

## Licensing

Proprietary, all rights reserved (see [LICENSE](LICENSE)). Do not install or import `ultralytics`, copy its source, or start from its weights; the model is our own reimplementation. Footage from HMGMA is under NDA and must never be committed.

## Which command do I run for what?

| I want to... | Run |
| --- | --- |
| Demo zones and events in the browser | [`uv run pv-demo`](#pv-demo-the-demo-api) and [`npm run dev`](#the-zone-demo-run-it) in `apps/demo-ui` |
| Start the local Postgres | `docker compose up -d db`, then `uv run pv-db migrate` ([details](#database-local-postgres)) |
| Look at stored events | `docker compose exec db psql -U palletvision -d palletvision` |
| Send queued events to Postgres | [`uv run pv-sync`](#pv-sync-send-queued-events-to-postgres) |
| Train, export, benchmark, run on video | [`pv-train`, `pv-export`, `pv-bench`, `pv-infer`](#model-commands) |
| Run every check | `uv run pytest`, plus `npm test` in `apps/demo-ui` |

## Setup

You need three tools:

| Tool | Why | Check |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python 3.11 and every Python dependency from `uv.lock` | `uv --version` |
| [Node.js](https://nodejs.org) 20 or newer | The demo UI (Next.js) | `node -v` |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Only for the local Postgres. Skip it if you use the no-database demo path. | `docker info` |

```bash
uv sync --all-packages                    # Python packages
npm install --prefix apps/demo-ui         # UI packages
cp .env.example .env                      # optional: database settings (defaults work as is)
uv run pytest                             # sanity check
```

## How the pieces fit

Diagrams of both setups, including what is not built yet, are in [docs/post-inference.md](docs/post-inference.md).

**On your laptop (the demo):** everything runs on one machine.

```text
 browser (:3000)  ──►  Next.js UI  ──/api──►  pv-demo API (:8000)  ──►  zone engine  ──►  event
   draw zones                                   saves each zone set as a version in Postgres   │
   drag mock boxes  ◄── live event log (SSE) ◄──────────────────────────────────────────────────┤
                                                                                                ▼
                                                              jsonl file │ Postgres │ outbox (--sink)
```

**In production (many Jetson Nanos):** a Jetson never talks to the database directly. It queues events locally and a sync agent forwards them, so a network outage loses nothing.

```text
 Jetson:  detector → tracker → zone engine → event → outbox (SQLite, durable) → pv-sync ─┐
 Jetson:  ...                                                                  (batches, retried)
                                                                                         ▼
                                                      Postgres (lab-controlled server, partitioned events table)
```

`pv-sync` already exists and writes to Postgres. The HTTP ingest service that would sit in front of Postgres, so Jetsons hold no database credentials, is a later step.

## The zone demo: run it

You run two programs, the API and the UI, plus optionally the database. Use one terminal each, all from the repo root.

**Path A: no database (quickest).** Events go to `runs/demo/events.jsonl`.

```bash
# terminal 1: the API
uv run pv-demo

# terminal 2: the UI
npm run dev --prefix apps/demo-ui
```

Open <http://localhost:3000>.

**Path B: with Postgres.** Events are written to the database.

```bash
# terminal 1: the database (first time only: migrate creates the tables)
docker compose up -d db
uv run pv-db migrate

# terminal 2: the API, writing to Postgres
uv run pv-demo --sink postgres

# terminal 3: the UI
npm run dev --prefix apps/demo-ui
```

Open <http://localhost:3000>, then check the rows with the [queries below](#look-at-the-data).

**Use your own camera frame.** `uv run pv-demo --source data/videos/clip.mp4` uses the video's first frame, `--source frame.png` an image, and `--source 0` a snapshot from the laptop webcam. Without `--source` you get a generated slanted floor grid.

### Walkthrough

1. Click **Draw zone** (or press `Z`). Click the corners of a floor area one by one. Use as many corners as the outline needs, so slanted or L-shaped areas are fine. Click the first point or press `Enter` to close it.
2. In **Mock objects**, pick a type and click **Add**. A bounding box appears. The dot at the bottom centre of the box is the point the zones test.
3. Drag the box. When the dot crosses a zone edge, the **Event log** on the right shows `object of type pallet_type_a joined zone zone_1` or `... left zone zone_1`. Dragging from one zone straight into another logs a leave, then a join.
4. Draw a zone around a box that is already there, or reshape a zone under one: events are logged for that too.

### Controls

| Where | Action | Result |
| --- | --- | --- |
| Toolbar | **Select** `V` / **Draw zone** `Z` | Switch tool |
| Draw zone | Click | Place a corner |
| Draw zone | Click first point, or `Enter` | Close and save the zone |
| Draw zone | `Backspace` / `Esc` | Undo last corner / cancel |
| Select | Click a zone or its name in the list | Select it |
| Select | Drag a corner handle of the selected zone | Reshape (saved on release) |
| Select | Double-click a zone name in the list | Rename |
| Select | `Delete` | Remove the selected zone or object |
| Select | Drag a box / drag its corner square | Move / resize a mock object |
| Mock objects | Type dropdown | Change the object's type |
| Mock objects | **Log events while dragging** | On: events fire the moment the dot crosses an edge. Off: only when you release. |

**Where zones are kept.** Zones live in the database, not in a file. With `--sink postgres`, every save is stored as a new, immutable version in `zone_configs` (you keep the full history), the demo loads the latest version when it starts, and version numbers continue from the database. To start over, draw new zones; old versions stay in the history. With any other sink there is no database, so zones exist only while the API is running, and a bar at the top of the page says so. If the saved zones were drawn on a different frame size, a warning is shown.

### `pv-demo`: the demo API

```bash
uv run pv-demo --sink postgres --camera-id cam01 --device-id nano-01
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--sink` | `jsonl` | Where events go: `jsonl` (a file), `postgres` (events and zone versions), `outbox` (local queue for `pv-sync`) or `stdout`. |
| `--source` | none | Reference frame: image, video, or camera index. Default is a generated placeholder. |
| `--camera-id` | `cam01` | Names the zone file and tags the events. |
| `--device-id` | `demo-laptop` | Tags the events. Production uses one id per Jetson. |
| `--jsonl` | `runs/demo/events.jsonl` | File used by `--sink jsonl`. |
| `--outbox` | `runs/outbox.db` | Queue file used by `--sink outbox`. |
| `--db-url` | from `.env` | Postgres URL for `--sink postgres`. |
| `--hysteresis` | `1` | Consecutive frames needed to confirm a move. `1` reacts instantly; real video would use about `3` so one noisy frame cannot create events. |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where the API listens. If you change the port, start the UI with `PV_API_URL=http://127.0.0.1:<port>`. |

If Postgres is down while using `--sink postgres`, dragging keeps working and the event log shows a "not stored" warning, but those events are lost. To avoid losing events, use `--sink outbox` and run `pv-sync` (below).

## Database (local Postgres)

A Postgres 16 server in Docker, for local development and tests only. Never put sponsor data in a cloud database.

```bash
docker compose up -d db        # start (needs Docker Desktop running)
uv run pv-db migrate           # create or update the tables
uv run pv-db status            # which migrations are applied
docker compose stop db         # stop, keep the data
docker compose down -v         # stop and delete all data
```

Settings come from `.env` (copy `.env.example`): `PV_DB_USER`, `PV_DB_PASSWORD`, `PV_DB_NAME`, `PV_DB_PORT`. The defaults work as they are. To point at another server, set `PV_DATABASE_URL`.

### `pv-db`: manage the schema

| Command / flag | Meaning |
| --- | --- |
| `migrate` | Apply any new migrations. Safe to run repeatedly. |
| `status` | List applied and pending migrations. |
| `--url` | Postgres URL. Default: built from `.env`. |

### What is in the database

| Table or view | Holds |
| --- | --- |
| `events` | One row per join (`enter`) or leave (`exit`). Partitioned by month. |
| `devices`, `cameras` | Who sent events. Created automatically on first event. |
| `zone_configs` | Every saved zone set, kept forever as an immutable version per camera (never edited). Written by `pv-demo --sink postgres`, which records `saved_by = demo`. |
| `zone_visits` | View that pairs each `enter` with the next `exit` and computes `dwell_s`. An open visit has no `t_exit`. |

### Look at the data

```bash
docker compose exec db psql -U palletvision -d palletvision
```

```sql
-- last 20 events
select device_ts, event_type, object_type, zone_id, track_id
from events order by device_ts desc limit 20;

-- how long objects stayed in each zone
select zone_id, object_type, t_enter, t_exit, round(dwell_s::numeric, 1) as dwell_s
from zone_visits order by t_enter desc limit 20;

-- which polygons was each event computed against? (needs --sink postgres)
select e.device_ts, e.event_type, e.zone_id, e.config_version, z.zones
from events e
join zone_configs z on z.camera_id = e.camera_id and z.version = e.config_version
order by e.device_ts desc limit 10;

-- every saved version of a camera's zones
select version, saved_by, created_at, jsonb_array_length(zones) as zone_count
from zone_configs where camera_id = 'cam01' order by version;

-- are any events missing from a device? (seq should count up by one)
select device_id, min(seq), max(seq), count(*) from events group by device_id;
```

### An event

| Field | Meaning |
| --- | --- |
| `event_id` | UUID made on the device. Resending the same event never creates a duplicate. |
| `device_id`, `camera_id` | Which Jetson and which camera produced it. |
| `seq` | Per-device counter, used to spot gaps. |
| `event_type` | `enter` (joined) or `exit` (left). |
| `zone_id`, `object_type`, `track_id` | Which zone, what kind of object, which tracked object. |
| `config_version` | Version of the zone file the event was computed against. |
| `source` | `mock` (demo) or `model` (real detections). |
| `device_ts` | Time on the device clock. The database also stores `received_at` from its own clock. |

The formal definitions are in [`contracts/events.schema.json`](contracts/events.schema.json) and [`contracts/zones.schema.json`](contracts/zones.schema.json).

### `pv-sync`: send queued events to Postgres

This is the production path in miniature. `pv-demo --sink outbox` writes events to a local SQLite queue; `pv-sync` drains it to Postgres in order. A batch is only marked sent after Postgres accepted it, so a failed attempt is retried (with growing delays) and a replay creates no duplicates.

```bash
uv run pv-demo --sink outbox       # terminal 1: events are queued in runs/outbox.db
uv run pv-sync                     # terminal 2: forwards them to Postgres
```

Try it: stop the database (`docker compose stop db`), drag some boxes, then `docker compose start db`. The events arrive afterwards, once each.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--outbox` | `runs/outbox.db` | Queue file to drain. |
| `--db-url` | from `.env` | Postgres URL. |
| `--interval` | `1.0` | Seconds between polls when idle. |
| `--batch-size` | `200` | Events per send. |
| `--once` | off | Drain once and exit. |

## Model commands

The typical flow is **train → export → benchmark → infer**. Each step feeds the next:

```text
pv-train  ->  runs/stage1/last.pt  ->  pv-export  ->  model.onnx  ->  pv-bench / pv-infer
```

Use the same `size` everywhere: the value you trained with is the value you export, benchmark and infer with.

### `pv-train`: train a model

Trains the detector and saves a checkpoint to `<output_dir>/last.pt`. Arguments are [Hydra](https://hydra.cc) overrides written as `key=value` (defaults live in `configs/`).

```bash
uv run pv-train data=synthetic data.size=64 data.num_samples=16 model.input_size=64 train.max_steps=4
```

| Override | Default | Meaning |
| --- | --- | --- |
| `data` | `loco` | Dataset. Only `synthetic` works for now (generated boxes, no download). |
| `data.size` | `640` | Synthetic image size in pixels. |
| `data.num_samples` | `64` | Number of synthetic images. |
| `model.input_size` | `640` | Model input size. Keep equal to `data.size`. |
| `device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. |
| `train.epochs` | `100` | Passes over the data. |
| `train.batch_size` | `16` | Images per step. |
| `train.lr` | `0.001` | Learning rate. |
| `train.amp` | `true` | fp16 autocast, only used on cuda/mps. |
| `train.max_steps` | `null` | Stop after this many steps. Handy for quick smoke runs. |
| `train.output_dir` | `runs/stage1` | Where `last.pt` is written. |

### `pv-export`: convert a checkpoint to ONNX

Produces the `.onnx` file that `pv-bench` and `pv-infer` load (opset 13, tensor names from `contracts/model_io.yaml`).

```bash
uv run pv-export --checkpoint runs/stage1/last.pt --size 64 --out runs/export/model.onnx
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint written by `pv-train`. |
| `--size` | `640` | Input size the model was trained with. |
| `--out` | `runs/export/model.onnx` | Output ONNX path. |

### `pv-bench`: measure speed

Runs the detector on a random frame and reports mean/p50/p99 latency and fps, saved as JSON.

```bash
uv run pv-bench --weights runs/export/model.onnx --size 64
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--weights` | none | ONNX model. Omit to benchmark the dummy detector. |
| `--size` | `640` | Input size. |
| `--warmup` | `10` | Untimed runs before measuring. |
| `--iters` | `100` | Timed runs. |
| `--out` | `runs/bench/latest.json` | Where the JSON result is written. |

### `pv-infer`: run on a video or webcam

Detects on each frame, draws boxes and fps, and optionally writes an annotated video or opens a live window.

```bash
# video file -> annotated video
uv run pv-infer --source data/videos/clip.mp4 --weights runs/export/model.onnx --size 64 --out runs/out.mp4

# laptop webcam, live window (press q to quit)
uv run pv-infer --source 0 --weights runs/export/model.onnx --size 64 --show
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | required | Video path, or a camera index (`0` is the laptop webcam). |
| `--weights` | none | ONNX model. Omit to use the dummy detector, which draws a fixed box and only checks the video plumbing. |
| `--out` | none | Save the annotated video here. |
| `--fps` | `10` | Process this many frames per second (extra frames are skipped). |
| `--conf` | `0.25` | Drop detections below this confidence. |
| `--size` | `640` | Model input size. |
| `--show` | off | Show a live window. |

On macOS, grant camera access to your terminal (System Settings > Privacy & Security > Camera) before using `--source 0`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| UI says "Cannot reach the demo API" | Start the API: `uv run pv-demo`. If it is on another port, run the UI with `PV_API_URL=http://127.0.0.1:<port>`. |
| Top bar shows "Reconnecting" | The API stopped or restarted. The log fills back in when it returns. |
| `Cannot connect to the Docker daemon` | Open Docker Desktop and wait until it says it is running. |
| `pv-db: cannot reach Postgres` | `docker compose up -d db`, wait a few seconds, retry. |
| Port 5432 already in use | Set `PV_DB_PORT=5433` in `.env`, then `docker compose up -d db` again. |
| Event log says "Zones not saved" | The zones apply for this session, but Postgres could not store them. Start and migrate the database (`docker compose up -d db && uv run pv-db migrate`), then save the zones again or move a box: the missing version is stored first. |
| Bar says "Zones are kept in memory" | You started without `--sink postgres`, so zones vanish when the API stops. Restart with `uv run pv-demo --sink postgres`. |
| Drawn zones are gone after restarting the API | Same cause: the API was not using Postgres. |
| Event log says "Shown here but not stored" | Postgres is unreachable or not migrated: `docker compose up -d db && uv run pv-db migrate`. |
| Port 3000 or 8000 busy | `npm run dev --prefix apps/demo-ui -- --port 3001`, or `uv run pv-demo --port 8001` with `PV_API_URL` set for the UI. |
| Zones look misplaced after changing `--source` | The saved zones were drawn on another frame size (the page warns about it). Keep the same source, or use a new `--camera-id` to start with empty zones. |
| Webcam snapshot is black or fails | Grant camera access to your terminal (macOS Privacy settings) and retry. |
| Database tests are skipped | They need the local Postgres. Run `docker compose up -d db`. |

## Layout

| Path | What |
| --- | --- |
| `packages/palletvision-model` | Pure `nn.Module`s. Imports only `torch`. |
| `packages/palletvision-train` | Trainer, datasets, ONNX export (`pv-train`, `pv-export`). |
| `packages/palletvision-preprocess` | Letterbox shared by training and inference, so both see identical pixels. |
| `packages/palletvision-infer` | Frame source, backends (torch, onnxruntime), overlay, benchmark (`pv-infer`, `pv-bench`). |
| `packages/palletvision-zones` | Zone geometry and the enter/exit engine. Plain Python with no heavy dependencies, written to be ported to C++. |
| `packages/palletvision-events` | Event records, sinks (jsonl, SQLite outbox, Postgres), migrations, `pv-db` and `pv-sync`. |
| `packages/palletvision-demo` | The demo API (`pv-demo`). Kept separate so FastAPI never ships to a Jetson. |
| `apps/demo-ui` | The browser UI (Next.js). |
| `contracts/` | Model I/O, class lists, and the zone and event schemas shared by every runtime. |
| `configs/` | Hydra training configs. |
| `tests/golden/` | Language-neutral zone fixtures. The future C++ engine must reproduce them. |
| `data/` | Local datasets and videos. Git-ignored. |
| `docs/design.md` | Design document. |
| `docs/post-inference.md` | Plain-language diagrams of what happens after detection. |
| `docs/yolo26-bytetrack-implementation-guide.md` | Step-by-step guide to implementing the detector and tracker. |

## Laptop notes (Apple silicon)

- `device: auto` picks cuda, then mps, then cpu. Override with `device=cpu|cuda|mps`.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to CPU.
- fp16 autocast only on cuda/mps (`train.amp`); no DDP on mps.
- Docker is used only for Postgres. Python itself runs natively through `uv`, because a container cannot reach the Mac GPU or webcam; `uv.lock` and `.python-version` pin the environment.
- The demo UI is built for a desktop browser at about 1200 px wide or more.

## Datasets

| Dataset | Needed | Use |
| --- | --- | --- |
| [LOCO](https://github.com/tum-fml/loco) | Yes, for real stage-1 training | 5,593 real warehouse images in COCO format. Unpack to `data/loco/`; all pallet-like labels map to `pallet`. |
| A few short `.mp4` clips | Yes, for `pv-infer` demos | Put in `data/videos/`. |
| [SORDI](https://sordi.ai) | Optional | Small synthetic subset for extra stage-1 volume. |
| COCO subset (~10k) | Optional, M1 | Stage-0 check that the reimplementation matches published curves; run on a cluster. |
| HMGMA footage | Stage 2 only | Lab machine or PACE only, never this repo. |

The LOCO loader is not wired into `pv-train` yet; until then use `data=synthetic`. The zone demo needs no dataset.

## Development

```bash
uv run pre-commit install                                   # ruff and file checks on every commit
uv run ruff check . && uv run ruff format --check .         # Python lint and format
uv run mypy packages                                        # Python types
uv run pytest                                               # Python tests (database tests need docker compose up -d db)
cd apps/demo-ui && npm run format:check && npm run lint && npm test && npm run build
```

CI runs the same checks, with a Postgres service, on every push and pull request. Commits are small and conventional (`feat:`, `fix:`, `build:`, `chore:`, `docs:`, `test:`, `ci:`, `style:`) so each is easy to revert.
