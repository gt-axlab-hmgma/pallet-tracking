#!/usr/bin/env bash
# Double-click on a Mac: the short check (a few minutes) that the machine can run the job.
bash "$(dirname "$0")/pallet-job.sh" run --smoke "$@"
status=$?
echo
read -r -p "Finished (exit code $status). Press Enter to close this window."
exit $status
