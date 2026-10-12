#!/usr/bin/env bash
# Double-click on a Mac: runs the whole overnight job and keeps the window open at the end.
# (First time: right-click > Open, since macOS asks before running a downloaded script.)
bash "$(dirname "$0")/pallet-job.sh" "$@"
status=$?
echo
read -r -p "Finished (exit code $status). Press Enter to close this window."
exit $status
