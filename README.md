# HMGMA Pallet Vision

Pallet detection, tracking and zone analytics for HMGMA. A clean-room PyTorch detector is trained per frame; a tracker and zone engine turn detections into events such as "an object of type `pallet_type_a` joined `zone_2`". See [docs/design.md](docs/design.md) for the full design.

**Status**

- Working: training, ONNX export, benchmarking and video/webcam inference; the ByteTrack tracker; the zone engine; the event pipeline (local queue, Postgres); and a browser demo where you load a video or camera, draw zones and watch pallets enter and leave them (see [video and camera sources](#video-and-camera-sources)).
- Not yet: the HTTP ingest service for many Jetsons, and the C++/TensorRT runtime.

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
| Run every check | `uv run pytest`, plus `npm test` in `apps/demo-ui`. The overfit gate test is slow and runs on request: `uv run pytest -m slow` |
| Check the whole pipeline can learn | [`uv run pv-overfit`](#pv-overfit-the-overfit-gate) |

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
   load video/camera, draw zones                saves each zone set as a version in Postgres   │
   watch in/out     ◄── live event log (SSE) ◄──────────────────────────────────────────────────┤
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

You run two programs, the API and the UI, plus optionally the database. Use one terminal each, all from the repo root. The API needs the detector as an ONNX file. To make one from the model in use:

```bash
uv run pv-export --checkpoint weights/pallet_yolo26n_coco-init_loco-sordi-mixed.pt --out runs/export/pallet_yolo26n.onnx
```

**Path A: no database (quickest).** Events go to `runs/demo/events.jsonl`.

```bash
# terminal 1: the API
uv run pv-demo --weights runs/export/pallet_yolo26n.onnx

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
uv run pv-demo --weights runs/export/pallet_yolo26n.onnx --sink postgres

# terminal 3: the UI
npm run dev --prefix apps/demo-ui
```

Open <http://localhost:3000>, then check the rows with the [queries below](#look-at-the-data).

You can also start the API already playing something: `--source data/infer/clip.mp4`, `--source 0` (a camera) or `--source rtsp://...`. Without `--weights` the page still opens, so zones can be drawn, but no source can be played.

### Walkthrough

1. In **Source**, choose **Video clip** and pick a file from your computer, or **Camera** and connect one (see below). The video plays behind the zones, and a box with an id follows every pallet. The dot at the bottom centre of a box is the point the zones test; it turns green inside a zone.
2. Click **Draw zone** (or press `Z`). Click the corners of a floor area one by one. Use as many corners as the outline needs, so slanted or L-shaped areas are fine. Click the first point or press `Enter` to close it. You can draw zones before loading a source, on the placeholder frame, but a zone is drawn in the pixels of the current frame, so load the source first if you can.
   Or click **Auto-detect zones** to have the painted floor lines found for you. The proposed zones show dashed on the frame; **Add** keeps them as normal zones (nothing is saved before that) and **Discard** drops them. Then drag corners, rename or delete them like any zone, and draw any bay it missed. It finds the paint color from the frame by itself (any color; it keeps the one that forms bays). If it finds nothing, click **Pick line color** and then click on a painted line: the color, and how pale or dark it looks, is read from that spot, which works for any camera or lighting. It looks at the first frame of the source, and needs a base line with at least two walls standing on it, so treat the result as a starting point. It is tuned on generated footage only, so check it on your own camera.
3. Watch the **Event log** on the right. When a pallet's dot crosses a zone edge it shows `object of type pallet_unknown joined zone zone_1` or `... left zone zone_1`.
4. Draw a zone around a pallet that is already there, or reshape a zone under one: events are logged for that too.

### Video and camera sources

- **Video clip:** the file is sent to the API (which runs on your machine) and saved under `runs/demo/uploads/` (git-ignored, never committed). The video loops; at each restart the tracker and all open visits are reset, so exit events are logged. `--no-loop` stops at the end.
- **Camera:** type an index (`0` is the first camera on the machine running the API; macOS asks the terminal for camera permission the first time) or an `rtsp://` / `http(s)://` stream URL. Only indexes and stream URLs are accepted, never file paths, and a password in a URL is never shown or logged.
- **Stop** removes the source: pallets still inside a zone exit it.
- **A hidden pallet keeps its zone.** Only pallets seen in the current frame move through zones. A pallet hidden behind a forklift stays "inside" until the tracker gives up on it, `--lost-seconds` after it was last seen, and only then is an exit logged.
- **IDs are not appearance-based.** Tracking is by box overlap and motion only, so two pallets that cross paths can swap ids, and a pallet hidden longer than `--lost-seconds` comes back with a new one.
- **One class for now.** Every pallet is `pallet_unknown`; class voting is not built yet.

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
| Select | `Delete` | Remove the selected zone |
| Source | **Choose a video…** / **Connect** / **Stop** | Load a clip, connect a camera, or remove the source |

**Where zones are kept.** Zones live in the database, not in a file. With `--sink postgres`, every save is stored as a new, immutable version in `zone_configs` (you keep the full history), the demo loads the latest version when it starts, and version numbers continue from the database. To start over, draw new zones; old versions stay in the history. With any other sink there is no database, so zones exist only while the API is running, and a bar at the top of the page says so. If the saved zones were drawn on a different frame size, a warning is shown.

### `pv-demo`: the demo API

```bash
uv run pv-demo --sink postgres --camera-id cam01 --device-id nano-01
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--sink` | `jsonl` | Where events go: `jsonl` (a file), `postgres` (events and zone versions), `outbox` (local queue for `pv-sync`) or `stdout`. |
| `--weights` | none | ONNX model. Needed to play a video or camera. |
| `--source` | none | Start on this video file, camera index or stream URL. Otherwise choose one in the UI. |
| `--fps` | `10` | Frames per second the detector and tracker run at (a file is thinned to this). |
| `--conf` | `0.1` | Lowest detection score kept. ByteTrack's second round uses low scores, so this is lower than the `0.25` of `pv-infer`. |
| `--lost-seconds` | `5` | How long a hidden pallet keeps its id. |
| `--no-loop` | loop | Stop at the end of a video instead of restarting. |
| `--upload-dir` | `runs/demo/uploads` | Where videos sent from the UI are kept. |
| `--camera-id` | `cam01` | Names the zone file and tags the events. |
| `--device-id` | `demo-laptop` | Tags the events. Production uses one id per Jetson. |
| `--jsonl` | `runs/demo/events.jsonl` | File used by `--sink jsonl`. |
| `--outbox` | `runs/outbox.db` | Queue file used by `--sink outbox`. |
| `--db-url` | from `.env` | Postgres URL for `--sink postgres`. |
| `--hysteresis` | `1` | Consecutive frames needed to confirm a move. `1` reacts instantly; real video would use about `3` so one noisy frame cannot create events. |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where the API listens. If you change the port, start the UI with `PV_API_URL=http://127.0.0.1:<port>`. |

If Postgres is down while using `--sink postgres`, tracking keeps working and the event log shows a "not stored" warning, but those events are lost. To avoid losing events, use `--sink outbox` and run `pv-sync` (below).

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
| `source` | `model` (detections from the detector and tracker). `mock` is no longer produced by the demo. |
| `device_ts` | Time on the device clock. The database also stores `received_at` from its own clock. |

The formal definitions are in [`contracts/events.schema.json`](contracts/events.schema.json) and [`contracts/zones.schema.json`](contracts/zones.schema.json).

### `pv-sync`: send queued events to Postgres

This is the production path in miniature. `pv-demo --sink outbox` writes events to a local SQLite queue; `pv-sync` drains it to Postgres in order. A batch is only marked sent after Postgres accepted it, so a failed attempt is retried (with growing delays) and a replay creates no duplicates.

```bash
uv run pv-demo --sink outbox       # terminal 1: events are queued in runs/outbox.db
uv run pv-sync                     # terminal 2: forwards them to Postgres
```

Try it: stop the database (`docker compose stop db`), let some pallets cross a zone, then `docker compose start db`. The events arrive afterwards, once each.

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

Trains the YOLO26 detector (both heads, the task-aligned assigner and the loss). It writes to `<output_dir>` (default `runs/stage1`): `last.pt`, a checkpoint saved after every epoch; `best.pt`, saved whenever a validation score improves (once evaluation exists, #40); and `metrics.jsonl`, one JSON line per step and per epoch, written as the run goes, so `tail -f runs/stage1/metrics.jsonl` shows a live run. Arguments are [Hydra](https://hydra.cc) overrides written as `key=value` (defaults live in `configs/`).

```bash
# a few steps on generated boxes (no download)
uv run pv-train data=synthetic data.size=64 data.num_samples=16 model.input_size=64 train.max_steps=4

# real LOCO images (needs data/loco): the first 8 images only, to check everything agrees
uv run pv-train data=loco data.limit=8 data.num_workers=0 model.input_size=256 train.epochs=12 train.batch_size=4 device=cpu
```

| Override | Default | Meaning |
| --- | --- | --- |
| `data` | `loco` | Dataset: `loco` (needs `data/loco`), `loco_sordi` (LOCO plus the SORDI synthetic scenes in `data/sordid-*`, see below) or `synthetic` (generated boxes, no download). |
| `data.size` | `640` | Synthetic image size in pixels. |
| `data.num_samples` | `64` | Number of synthetic images. |
| `data.limit` | `null` | Keep only the first N training images (smoke runs, overfit checks). With `loco_sordi` this limits the real images; `data.sordi.limit` limits the synthetic pool. |
| `data.sordi.share` | `null` | `loco_sordi` only. `null` (default): real and synthetic images are one pool and every epoch is a uniform random draw of `data.sordi.epoch_images` (5,640) from it, so with all the SORDI scenes about 1% of an epoch is real. A number such as `0.5`: every real image once plus a fresh random draw of synthetic ones making that share of the epoch (the setting of experiment 002). The real LOCO validation score is logged either way and still picks `best.pt`. |
| `data.augment.enabled` | `true` (loco) | Training augmentation: mosaic, zoom and shift, colour jitter and a flip, with the YOLO26 report's strengths (`data.augment.mosaic` 1.0, `scale` 0.5, `translate` 0.1, `fliplr` 0.5, `hsv_h/s/v` 0.02/0.7/0.4). Mosaic switches off for the last `data.augment.close_mosaic` (8) epochs. Validation is never augmented. `false` trains on plain letterboxed images. |
| `data.num_workers` | `4` (loco) | Loader worker processes. |
| `model.input_size` | `640` | Model input size; images are letterboxed to it. Keep equal to `data.size` for synthetic data. |
| `device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. |
| `train.epochs` | `150` | Passes over the data (the report's pretraining recipe). About 1.5 min each on the M5 Pro. |
| `train.batch_size` | `16` | Images per step. |
| `train.lr` | `0.001` | Learning rate. |
| `train.amp` | `false` | Mixed precision: the forward and backward pass run in 16 bits, the weights, loss and optimiser stay float32. Works on cuda, mps and (bfloat16) cpu. Check held-out mAP against a float32 run before relying on it (#38). |
| `train.amp_dtype` | `auto` | `bfloat16`, `float16` or `auto` (cuda: bfloat16 if the GPU supports it, else float16 with loss scaling; mps: float16). |
| `train.max_steps` | `null` | Stop after this many steps. Handy for quick smoke runs. |
| `train.output_dir` | `runs/stage1` | Where `last.pt`, `best.pt` and `metrics.jsonl` are written. |
| `train.val_every` | `0` | Score the real LOCO validation split (sub-set 1) with mAP50-95 every N epochs and after the last; the best score writes `best.pt`. With `data=loco_sordi` the synthetic validation split (a fixed `data.sordi.val_images` draw) is scored too and logged as `sordi_val_map50_95`, but it does not pick `best.pt`. `0` never scores. Needs `data=loco` or `data=loco_sordi`. |
| `data.val_limit` | `null` | Score only the first N validation images (smoke runs). |
| `train.optimizer` | `musgd` | `musgd` (YOLO26's: SGD plus a Muon update on the convolution weights), `sgd`, or `adamw`. Weight decay is applied to the convolution weights only. |
| `train.lr` | `0.01` | Peak learning rate (use `0.001` for `adamw`). |
| `train.momentum` | `0.937` | Momentum (AdamW: its first beta). |
| `train.muon_weight`, `train.sgd_weight` | `0.45`, `0.55` | How MuSGD mixes the two updates on the convolution weights (the YOLO26n values). |
| `train.nominal_batch` | `128` | Gradients of `round(128 / batch_size)` batches are averaged before each update, so small batches act like one of 128 (the report's batch; a real batch of 128 needs about 80 GB of memory, and memory grows about 0.6 GB per image); weight decay is scaled to match. `null` turns it off. |
| `train.warmup_epochs`, `train.lr_final`, `train.lr_schedule` | `1`, `0.01`, `linear` | The rate climbs from 0 over the first epoch (at least 100 iterations), then falls along a straight line (or `cosine`) to 1% of `lr` at the last epoch. |
| `train.init_from` | `null` | A checkpoint whose weights start this run (pretraining then fine-tuning). Weights only: unlike `resume` it does not restore the optimiser or the epoch. Tensors whose shape differs (the class heads, when the number of classes differs) keep their fresh initialisation. |
| `train.pretrained_lr_mult` | `1.0` | With `init_from`: the copied (pretrained) tensors train at this multiple of the learning rate, such as `0.2`, while the fresh ones (the new class branch) keep the full rate. One phase, no freezing. `1.0` changes nothing. |
| `train.alpha_schedule`, `train.alpha_init`, `train.alpha_final` | `epoch`, `0.8`, `0.1` | The one-to-many loss weight moves from init to final once per epoch (the report's) or, with `step`, every step. Use `step` when a run has only a few epochs: a 1-epoch run never leaves its starting weight otherwise. A converged pretrained model can start lower (`alpha_init=0.3`). |
| `train.ema`, `train.ema_decay`, `train.ema_tau` | `false`, `0.9999`, `2000` | Keep an exponential moving average of the weights. It is what is scored and saved as `model` in `last.pt` and `best.pt` (so `pv-eval` and `pv-export` use it); the weights being trained are kept as `raw_model` for resuming. The average looks back about `1 / (1 - decay)` updates: 0.99 to 0.995 for a fine-tune of about a thousand updates. |
| `train.resume` | `null` | `null` starts fresh, a path resumes from that checkpoint, `auto` resumes from `<output_dir>/last.pt` if it exists and starts fresh if not. Needs the same `train.epochs`. |

#### Stopping and resuming

A checkpoint holds the weights, the optimiser and the state of every random number generator, so a resumed run follows exactly the loss curve an uninterrupted one would. If a run dies or you stop it, start it again with the same command plus `train.resume=auto`; it continues from the last finished epoch (the unfinished one is redone from its start) and trims the log so each step appears once. Checkpoints are written to a temporary file and renamed, so a crash while saving never damages the previous one. `train.max_steps` is for quick smoke tests; a run cut short by it leaves a checkpoint for inspection, and resuming it replays the unfinished epoch.

#### Training with SORDI synthetic data

```bash
uv run pv-train data=loco_sordi train.val_every=5 train.epochs=30 train.output_dir=runs/loco_sordi
```

[SORDI](https://sordi.ai) scenes are synthetic warehouse renders with pallet and stillage boxes. Put each scene in `data/sordid-NN/` (with `images/` and `labels/json/`) and list it under `data.sordi.roots` in `configs/data/loco_sordi.yaml`. Pallets and stillages train as the one class `pallet`, as with LOCO; every other SORDI class is dropped. **Splits.** LOCO is split by whole recording site (train sub-sets 2, 3, 5; validation sub-set 1; test sub-set 4), because its frames are video and neighbouring frames are near-identical. SORDI frames are independent views (a neighbouring frame differs from a given one about as much as a random one, and DR levels do not share camera poses), so every SORDI image is assigned to train, val or test, 70/15/15, by a hash of its scene, level and number (`sordi.split_of`). A hash rather than a shuffle so that adding a scene never moves an image that already has a split. Training uses the train parts only. Report real (LOCO) and synthetic (SORDI) scores separately: pooled, the number would be almost all SORDI.

### `pv-doctor`: check a machine and its speed

```bash
uv run pv-doctor --batch 8                 # environment, warnings, then timed training steps
uv run pv-doctor --device cuda --batch 16 --size 640
```

Prints what PyTorch sees (version, CUDA build, GPU, memory, bfloat16 support), warns about the usual install mistakes (a CPU-only torch, a GPU newer than the PyTorch build), then times real training steps in float32 and with mixed precision on generated crowded images and reports seconds per step, images per second and peak memory. Run it first on any new machine; `docs/cuda-setup.md` has the NVIDIA setup (Windows, Linux, rented GPUs).

### Training on several GPUs

```bash
uv run torchrun --standalone --nproc_per_node=4 -m palletvision_train.cli.train data=loco_sordi \
  train.batch_size=32 data.num_workers=5 train.val_every=5 train.resume=auto
```

One process per GPU, gradients averaged each update; `train.batch_size` is per GPU and `train.nominal_batch` counts all GPUs. Rank 0 writes the logs and checkpoints and validates. Details, a Mac workaround for `--standalone` and what is and isn't tested are in `docs/cuda-setup.md`.

### `pv-job`: the whole overnight run on another machine

```bash
bash scripts/pallet-job.sh run --smoke   # a few minutes: prove the machine (Mac: double-click scripts/pallet-smoke.command)
bash scripts/pallet-job.sh               # the full job (Mac: pallet-job.command, Windows: pallet-job.bat)
uv run pv-job check                      # what data or credentials are missing?
```

Checks the machine and data, downloads the missing public data (LOCO, SORDI scenes), trains the overnight recipe with restarts, scores `best.pt` on the LOCO and SORDI val and test splits, plots, and writes `runs/results-<name>/` (and a zip): weights, curves, scores, logs, environment, settings, and a manifest with the git commit and the image counts per split. The bundle is built from a fixed file list, so no image or video can leave the machine. The Mac path is tested end to end; the Windows scripts, the Slurm template, Kaggle downloads and several GPUs are not yet run on real hardware. Full instructions for whoever sits at the other machine: `docs/remote-training.md`. `run --scenes 02,05` uses only those SORDI scenes, `run --pool-images N` keeps a fixed random subsample of the synthetic train images, and `run --budget-hours H --amp` sizes one big run to a time budget (every epoch is the whole train set; see `docs/remote-training.md`). Starting from pretrained weights (`run --init-from`) and the licence findings on Objects365 and COCO: `docs/pretraining.md`.

### `pv-eval`: score a checkpoint

Scores a checkpoint on LOCO the standard COCO way, so you can tell whether a run learned anything that generalises. It rebuilds the model from the config saved in the checkpoint.

```bash
uv run pv-eval --checkpoint runs/stage1/best.pt                        # the real validation split (LOCO sub-set 1)
uv run pv-eval --checkpoint runs/stage1/last.pt --split train --limit 8 --out runs/eval.json
uv run pv-eval --checkpoint runs/stage1/best.pt --split test                      # the held-out LOCO site, sub-set 4: run once, at the end
uv run pv-eval --checkpoint runs/stage1/best.pt --split test --source sordi --limit 2000   # the synthetic test split
```

It prints mAP50-95 (the headline number), mAP50, mAP75, AP for small, medium and large objects, average recall, and the confidence threshold that gives the best F1, which is the value to pass to `pv-infer --conf`. Details that matter for the numbers to mean something: images are letterboxed and nothing else, boxes are mapped back to the original image's pixels, the score threshold is 0.001 (mAP needs the weak tail, not a 0.25 cut-off), and up to 300 detections per image are kept (LOCO images can hold over 100 objects). The validation split (1,075 images) takes about 20 seconds on the M5 Pro. The LOCO splits are by recording site, so they measure how the model does in places it has not seen. Choose settings and `best.pt` on validation, and look at `--split test` only for the final number; a test score that has been tuned against is just another validation score.

### `pv-plot`: draw a run's curves

```bash
uv run pv-plot runs/stage1            # writes runs/stage1/curves.png (--window sets the smoothing)
```

Reads `metrics.jsonl` and draws three panels: the loss at every step (faint) and smoothed, for the blended loss and each branch; the loss of each epoch with the validation metric (every `train.val_every` epochs); and the learning rate with the Progressive Loss weight alpha. It works on a run that is still going, so you can check on an overnight run from the log.

### `pv-overfit`: the overfit gate

Before any real training, check that the model, assigner, loss and data reader agree. It trains a fresh model on a handful of images with no augmentation, then checks the model's own final detections against the labels of those same images.

```bash
uv run pv-overfit --device mps                              # 8 real LOCO images, 800 steps (about 4 minutes on the M5 Pro)
uv run pv-overfit --source synthetic --images 2 --size 64 --steps 300 --device cpu   # about a minute, no download
```

By default it trains with AdamW at 1e-3, the settings it was validated with; `--optimizer musgd --lr 0.01 --momentum 0.937` checks the training optimiser the same way (also `sgd`).

It passes when all three hold: the loss falls by at least 90%, AP50 on the training images is at least 0.95, and no image gets more than 1.25 times its object count (plus one) of confident boxes, which would mean duplicates. It exits with status 1 and lists the reasons when it fails. It also writes `overfit_<i>.jpg` pictures to `--out` (default `runs/overfit`), with labels in green and detections in red, so you can look at them. A falling loss alone proves little: after 120 steps on the synthetic images the loss was already down 94% while AP50 was 0.04.

### `pv-export`: convert a checkpoint to ONNX

Rebuilds the model from the config saved in the checkpoint (the same way `pv-eval` does) and produces the `.onnx` file that `pv-bench`, `pv-infer` and `pv-demo` load (opset 13, tensor names from `contracts/model_io.yaml`).

```bash
uv run pv-export --checkpoint runs/stage1/last.pt --out runs/export/model.onnx
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint written by `pv-train`. |
| `--size` | the size in the checkpoint | Input size to export at. Leave it unset unless you want another one. |
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
| `packages/palletvision-track` | The ByteTrack tracker: Kalman filter, IoU matching, track lifecycle. numpy and scipy only. |
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

`pv-train` reads LOCO through the official train files listed in `configs/data/loco.yaml`. The zone demo needs no dataset.

## Development

```bash
uv run pre-commit install                                   # ruff and file checks on every commit
uv run ruff check . && uv run ruff format --check .         # Python lint and format
uv run mypy packages                                        # Python types
uv run pytest                                               # Python tests (database tests need docker compose up -d db)
cd apps/demo-ui && npm run format:check && npm run lint && npm test && npm run build
```

CI runs the same checks, with a Postgres service, on every push and pull request. Commits are small and conventional (`feat:`, `fix:`, `build:`, `chore:`, `docs:`, `test:`, `ci:`, `style:`) so each is easy to revert.
