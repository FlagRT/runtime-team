#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/count_device_containers.sh"
while true; do
  n=$("$SCRIPT")
  echo "$(date '+%H:%M:%S') device-containers=$n"
  if [ "$n" -le 2 ]; then
    echo "CAPACITY_AVAILABLE (n=$n, room for 1 more within cap 3)"
    break
  fi
  sleep 20
done
