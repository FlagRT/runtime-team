"""Recount the published, redacted receipts; does not replay training or verify raw files."""

import json
import re
from pathlib import Path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data = json.loads(Path(__file__).with_name("summary.json").read_text(encoding="utf-8"))
    platforms = ("910C", "P800", "PPU")
    models = set(data["datasets"])
    cells = {(p, m) for p in platforms for m in models}
    check(len(models) == 9, "Expected nine designated models")
    sources = data["source_digests"]
    for tag, source in sources.items():
        for key in ("receipt_sha256", "raw_archive_sha256"):
            if key == "receipt_sha256" or key in source:
                check(re.fullmatch(r"[0-9a-f]{64}", source[key]), f"{tag}: invalid {key}")

    seen, native, passed, historical = set(), set(), set(), set()
    states = {}
    configs = set()
    for group in ("specified", "historical"):
        for row in data[group]:
            cell = row["platform"], row["model"]
            config = group, *cell, row.get("config")
            check(config not in configs and cell in cells, f"Invalid/duplicate config: {config}")
            configs.add(config)
            check(row["source"] and set(row["source"]) <= sources.keys(), f"No source for {config}")
            state, obs = row["state"], row.get("observed", {})
            if group == "specified":
                seen.add(cell)
                check(states.setdefault(cell, state) == state, f"Conflicting states: {cell}")
            else:
                check(state == "three_routes_pass", f"Unexpected historical state: {config}")
            if state == "not_trained":
                check(not obs, f"Training evidence on untrained case: {cell}")
                continue
            if state == "native_failed":
                check(obs == {"native": {"steps": 0, "status": "FAIL"}}, f"Native failure: {cell}")
                continue
            check(state in ("three_routes_pass", "native_pass"), f"Unknown state: {state}")
            check(obs["native"]["steps"] == 3, f"Native updates: {cell}")
            if group == "specified":
                native.add(cell)
            for control in ("off", "unknown"):
                if control in obs:
                    ctrl = obs[control]
                    check((ctrl["steps"], ctrl["forward"], ctrl["backward"]) == (3, 0, 0),
                          f"{config}: {control}")
            if state == "three_routes_pass":
                check({"native", "on", "off", "unknown"} <= obs.keys(), f"Missing control: {config}")
                on = obs["on"]
                check(on["steps"] == 3 and min(on["forward"], on["backward"]) > 0,
                      f"No completed target training: {config}")
                (passed if group == "specified" else historical).add(cell)
            else:
                check("on" not in obs, f"Unqualified case counts as on: {config}")

    check(seen == cells, "Incomplete matrix")
    check(historical - passed == {("P800", "DCN-Mix")}, "Historical-only result changed")
    for group, platform in (("specified", "910C"), ("specified", "PPU"), ("historical", "P800")):
        check({c[3] for c in configs if c[:3] == (group, platform, "DCN-Mix")}
              == {"sigmoid", "tanh", "sigmoid+tanh"}, f"DCN-Mix single/combined: {platform}")
    for platform in platforms:
        n, s, a = (sum(p == platform for p, _ in cellset)
                   for cellset in (native, passed, passed | historical))
        check(data["counts_by_platform"][platform] ==
              {"native": n, "specified_three_routes": s, "any_tested_input_three_routes": a},
              f"Published totals differ: {platform}")
        print(f"{platform}: native {n}/9, three routes on designated data {s}/9")
    print(f"Total: native {len(native)}/{len(cells)}, designated three routes {len(passed)}/{len(cells)};"
          f" including other tested inputs {len(passed | historical)}/{len(cells)}.")


if __name__ == "__main__":
    main()
