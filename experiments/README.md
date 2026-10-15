# Experiments

One folder per experiment, `experiments/NNN-short-name/README.md`, containing:

- **Goal:** the question being answered.
- **Config:** the exact `pv-train` command and git commit.
- **Data:** dataset and split (split by clip, never by frame).
- **Result:** metrics (mAP50, mAP50-95, latency, fps) and where the run is stored.
- **Conclusion:** what to do next.

Checkpoints and logs live under `runs/exp-NNN-topic/` (git-ignored); the model in use is in `weights/`; the one-page summary is `FINDINGS.md`; only the write-up is committed.
