#!/usr/bin/env python3
"""
Stage 1 sanity check for DARPA2000.

This does NOT do any preprocessing (that's Stage 2). Its only job is to prove
that what we downloaded is actually what the thesis needs: real multi-phase
attack campaigns with per-phase ground-truth labels, not just an undifferentiated
traffic dump. If a scenario has only one phase, or the truth files are missing
or empty, that is a red flag to raise now rather than after Stage 2/4 are built
on top of it.

Usage:
    python scripts/stage1_verify_darpa2000.py [data/raw/darpa2000]
"""
import sys
import json
from pathlib import Path

def find_scenario_dirs(root: Path):
    # each scenario extracts to <root>/<name>/<name>/data_and_labeling/...
    # (top-level dir from our download script, inner dir from the tar.gz itself)
    dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        matches = list(child.rglob("data_and_labeling"))
        if matches:
            dirs.append((child.name, matches[0]))
    return dirs

def inspect_capture_dir(capture_dir: Path):
    """capture_dir is e.g. .../tcpdump_inside or .../tcpdump_dmz"""
    if not capture_dir.is_dir():
        return None
    phases = {}
    for f in sorted(capture_dir.iterdir()):
        name = f.name
        if "phase-" in name and "-dump" in name:
            try:
                phase_num = int(name.split("phase-")[1].split("-")[0])
            except (IndexError, ValueError):
                continue
            phases.setdefault(phase_num, {})["dump_bytes"] = f.stat().st_size
        elif "phase-" in name and name.endswith(".list"):
            try:
                phase_num = int(name.split("phase-")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            phases.setdefault(phase_num, {})["list_bytes"] = f.stat().st_size
        elif "mid-level-phase-" in name and name.endswith(".xml"):
            try:
                phase_num = int(name.split("mid-level-phase-")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            phases.setdefault(phase_num, {})["truth_xml_bytes"] = f.stat().st_size
    return phases

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/darpa2000")
    if not root.exists():
        print(f"ERROR: {root} does not exist. Run stage1_download_darpa2000.sh first.")
        sys.exit(1)

    scenarios = find_scenario_dirs(root)
    if not scenarios:
        print(f"ERROR: no extracted scenarios found under {root}. "
              f"Expected .../<scenario>/data_and_labeling/")
        sys.exit(1)

    report = {}
    problems = []

    for name, data_and_labeling in scenarios:
        report[name] = {}
        for tier_dirname in ["tcpdump_inside", "tcpdump_dmz"]:
            tier_dir = data_and_labeling / tier_dirname
            phases = inspect_capture_dir(tier_dir)
            if phases is None:
                problems.append(f"{name}/{tier_dirname}: directory missing")
                continue
            report[name][tier_dirname] = phases

            n_phases = len(phases)
            if n_phases < 2:
                problems.append(
                    f"{name}/{tier_dirname}: only {n_phases} phase(s) found — "
                    f"this would NOT support a multi-stage correlation claim"
                )

            for phase_num, info in phases.items():
                if info.get("dump_bytes", 0) == 0:
                    problems.append(f"{name}/{tier_dirname} phase {phase_num}: dump file is empty")
                if "truth_xml_bytes" not in info:
                    problems.append(
                        f"{name}/{tier_dirname} phase {phase_num}: NO ground-truth XML — "
                        f"cannot use this phase for labeled evaluation"
                    )
                elif info["truth_xml_bytes"] == 0:
                    problems.append(f"{name}/{tier_dirname} phase {phase_num}: ground-truth XML is empty")

    print("=" * 70)
    print("DARPA2000 Stage 1 inventory")
    print("=" * 70)
    for name, tiers in report.items():
        print(f"\n[{name}]")
        for tier_dirname, phases in tiers.items():
            print(f"  {tier_dirname}: {len(phases)} phase(s)")
            for phase_num in sorted(phases):
                info = phases[phase_num]
                dump_kb = info.get("dump_bytes", 0) / 1024
                has_truth = "truth_xml_bytes" in info
                print(f"    phase {phase_num}: dump={dump_kb:8.1f} KB  "
                      f"ground_truth_xml={'yes' if has_truth else 'MISSING'}")

    print("\n" + "=" * 70)
    if problems:
        print(f"SANITY CHECK: {len(problems)} problem(s) found — investigate before proceeding:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("SANITY CHECK PASSED: multiple phases with non-empty dumps and "
              "ground-truth labels found for each scenario. This supports the "
              "multi-stage correlation claim — proceed to Stage 2.")
    print("=" * 70)

    out_path = Path("data/processed")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "darpa2000_inventory.json", "w") as f:
        json.dump({"report": report, "problems": problems}, f, indent=2)
    print(f"\nFull inventory written to {out_path / 'darpa2000_inventory.json'}")

    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
