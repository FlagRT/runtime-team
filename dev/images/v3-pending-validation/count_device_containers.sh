#!/usr/bin/env bash
count=0
for id in $(docker ps -q); do
  devs=$(docker inspect "$id" --format '{{len .HostConfig.Devices}}' 2>/dev/null)
  if [ -n "$devs" ] && [ "$devs" -gt 0 ] 2>/dev/null; then
    count=$((count+1))
  fi
done
echo "$count"
