#!/usr/bin/env bash
# vision/after.sh PID GPU: start the GPU's queue worker once process PID has exited.
while kill -0 "$1" 2>/dev/null; do sleep 20; done
exec "$(dirname "$0")/worker.sh" "$2"
