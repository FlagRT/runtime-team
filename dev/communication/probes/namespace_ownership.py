"""Correlate saved UDA snapshots; never stop containers or change driver state.

Inputs are host-local diagnostic snapshots, not artifacts intended for GitHub.
An owner mapping is not a statement that the device is running compute work.
"""
import argparse
import json
import re
from pathlib import Path


def correlate(udevice, namespace_node, containers):
    owners = {}
    for line in namespace_node.splitlines():
        match = re.match(r"ns_id (\d+) .*?root_tgid (\d+) dev_num (\d+) namespace (\S+)", line)
        if match:
            ns_id, pid, count, namespace = match.groups()
            key = int(ns_id)
            if key in owners:
                raise ValueError("duplicate namespace ID; take a fresh snapshot")
            owners[key] = {"root_pid": int(pid), "device_count": int(count),
                           "namespace": namespace.rstrip(",")}
    by_pid = {}
    for item in containers:
        pid = int(item["State"]["Pid"])
        if pid > 0:
            by_pid[pid] = item["Name"].lstrip("/")
    devices = []
    current = None
    for line in udevice.splitlines():
        match = re.match(r"udevid (\d+)\b", line)
        if match:
            current = {"udevid": int(match.group(1)), "state": "no_exclusive_owner_in_snapshot"}
            devices.append(current)
        if current is None:
            continue
        match = re.search(r"has used by container, ns id (\d+)", line)
        if match:
            ns_id = int(match.group(1))
            owner = owners.get(ns_id)
            current.update(state="exclusive_namespace_registered", ns_id=ns_id)
            if owner:
                current.update(owner)
                current["container"] = by_pid.get(owner["root_pid"])
            else:
                current["state"] = "unresolved_namespace_snapshot"
        if "shared dev used by ns id:" in line:
            current["state"] = "shared_namespace_registration"
    if not devices:
        raise ValueError("no UDA device records; empty input is not proof of availability")
    return {"schema_version": 1, "devices": devices,
            "warning": "Snapshot correlation only; verify freshness and live PID before action. "
                       "No owner in snapshot does not prove device availability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udevice", type=Path, required=True)
    parser.add_argument("--namespace-node", type=Path, required=True)
    parser.add_argument("--containers", type=Path, required=True,
                        help="docker inspect JSON array; retain locally only")
    args = parser.parse_args()
    try:
        result = correlate(args.udevice.read_text(), args.namespace_node.read_text(),
                           json.loads(args.containers.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"diagnostic input error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
