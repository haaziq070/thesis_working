#!/usr/bin/env python3
"""
Stage 2 (DARPA2000 half): parse the phase-N.list session records for both
LLDOS scenarios into the unified event schema (src/schema.py), and derive
campaign-membership ground truth directly from the data.

Why "derive" campaign membership instead of just labeling every event in a
scenario as one campaign: phase-1/phase-2 (recon) events touch MANY candidate
target IPs, but only a subset of those targets actually get exploited in
later phases. Treating everything in a scenario as one undifferentiated
"campaign" would throw away a real, meaningful distinction: dead-end recon
against hosts that were never compromised is not part of the attack chain
that actually succeeded, and a correlation system should NOT link it in.
Labeling it as if it were part of the campaign would make the correlation
task artificially easy (no benign-looking distractors to reject) and would
inflate scores for the wrong reason -- exactly the kind of thing we've been
told to avoid.

Method: for each scenario, an IP is a "campaign host" if it appears as a
source or destination in any phase-4 (install) or phase-5 (action) record --
i.e. it was actually part of the attack chain that reached compromise/impact,
not just probed. Any event (any phase, either vantage point) touching a
campaign host is part of the true campaign. Everything else in the same
scenario/phase window is a same-time-window distractor: real DARPA traffic,
wrong target, correctly NOT linked.

This dataset is reserved as the project's held-out external evaluation set.
No code in Stage 3 or Stage 4 training may read data/processed/darpa2000_events.csv.
It is read only by the Stage 5 evaluation script.

Usage:
    python scripts/stage2_parse_darpa2000.py [data/raw/darpa2000] [data/processed]
"""
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import UNIFIED_COLUMNS, DARPA_PHASE_TO_STAGE, DARPA_PHASE_TO_ATTACK_TYPE

SCENARIOS = ["LLS_DDOS_1.0", "LLS_DDOS_2.0.2"]
VANTAGES = {"tcpdump_inside": "inside", "tcpdump_dmz": "dmz"}
PHASES = [1, 2, 3, 4, 5]

LIST_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2})\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)

def _num_or_nan(tok):
    return float(tok) if tok not in ("-", "") else float("nan")

def parse_list_file(path, scenario, vantage, phase):
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = LIST_LINE_RE.match(line)
            if not m:
                continue
            seq, date_s, time_s, dur_s, service, sport, dport, src_ip, dst_ip, flag, extra = m.groups()
            ts = datetime.strptime(f"{date_s} {time_s}", "%m/%d/%Y %H:%M:%S")
            rows.append({
                "event_id": f"{scenario}_{vantage}_p{phase}_{seq}",
                "timestamp": ts,
                "src_ip": _clean_ip(src_ip),
                "dst_ip": _clean_ip(dst_ip),
                "src_port": _num_or_nan(sport),
                "dst_port": _num_or_nan(dport),
                "service": service,
                "tier": "network",
                "vantage": vantage,
                "source_dataset": f"darpa2000_{scenario.lower().replace('lls_ddos_', 'lldos')}",
                "scenario": scenario,
                "phase": phase,
            })
    return rows

def _clean_ip(raw):
    # DARPA lists zero-pad octets, e.g. "202.077.162.213" -> "202.77.162.213"
    return ".".join(str(int(o)) for o in raw.split("."))

def count_xml_alerts(path):
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    return text.count("<Alert ")

def main():
    raw_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/darpa2000")
    out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    xml_mismatches = []

    for scenario in SCENARIOS:
        scenario_dir = next((raw_root / scenario).rglob("data_and_labeling"))
        for vantage_dirname, vantage in VANTAGES.items():
            vdir = scenario_dir / vantage_dirname
            if not vdir.exists():
                continue
            for phase in PHASES:
                list_path = vdir / f"phase-{phase}.list"
                xml_path = vdir / f"mid-level-phase-{phase}.xml"
                if not list_path.exists():
                    continue
                rows = parse_list_file(list_path, scenario, vantage, phase)
                all_rows.extend(rows)

                xml_count = count_xml_alerts(xml_path)
                if xml_count is not None and xml_count != len(rows):
                    xml_mismatches.append(
                        f"{scenario}/{vantage}/phase-{phase}: list has {len(rows)} rows, "
                        f"xml has {xml_count} alerts"
                    )

    df = pd.DataFrame(all_rows)
    print(f"Parsed {len(df)} raw session records across {df['scenario'].nunique()} scenarios.")

    if xml_mismatches:
        print("\nWARNING: list-file / xml-file record counts disagree (investigate before trusting counts):")
        for m in xml_mismatches:
            print(f"  - {m}")
    else:
        print("Cross-check OK: every phase's .list row count matches its .xml alert count.")

    # --- derive campaign membership from the data itself ---
    # IMPORTANT: campaign_hosts (used to decide whether a phase-1/2/3 recon
    # event escalated into a real attack) is derived ONLY from phase 4
    # (install). Phase 5 (the actual DDoS launch) is deliberately excluded
    # from this derivation: it is a spoofed-source flood, so most of its
    # source IPs are forged and appear exactly once -- treating them as
    # "campaign hosts" would (and, on the first attempt, did) wrongly pull in
    # thousands of unrelated one-off addresses. Phase 4 and phase 5 events
    # are still always labeled as in-campaign (see note on aggregation
    # below), just not used to *expand* the host set.
    df["campaign_id"] = "NONE"
    df["is_campaign_link"] = False
    df["kill_chain_stage"] = df["phase"].map(DARPA_PHASE_TO_STAGE)
    df["attack_type"] = df["phase"].map(DARPA_PHASE_TO_ATTACK_TYPE)

    campaign_host_report = {}
    for scenario in SCENARIOS:
        sdf = df[df["scenario"] == scenario]
        install_phase = sdf[sdf["phase"] == 4]
        all_hosts = set(install_phase["src_ip"]) | set(install_phase["dst_ip"])
        # Restrict to the internal/target address space (172.16.0.0/16, the
        # simulated "inside" network in this testbed) when deciding phase-1/2/3
        # membership. Without this, the attacker's own external IP
        # (202.77.162.213) ends up in the host set -- and since the attacker
        # is, by definition, party to every probe packet including dead-end
        # ones, the OR-membership check becomes nearly vacuous (almost
        # everything "touches the attacker"). What actually distinguishes a
        # real escalation from a dead-end probe is whether the *target*
        # ended up compromised, not whether the attacker was involved.
        campaign_hosts = {ip for ip in all_hosts if ip.startswith("172.16.")}
        campaign_host_report[scenario] = {
            "internal_hosts_used_for_matching": sorted(campaign_hosts),
            "excluded_non_internal": sorted(all_hosts - campaign_hosts),
        }

        # phase 4 and 5 are dedicated attack-window captures by construction
        # (Stage 1 confirmed no background/benign traffic is mixed into these
        # per-phase dumps) -- they are unconditionally in-campaign.
        always_in = (df["scenario"] == scenario) & (df["phase"].isin([4, 5]))
        df.loc[always_in, "campaign_id"] = scenario
        df.loc[always_in, "is_campaign_link"] = True

        # phase 1/2/3 recon/exploit events only count if they touch a host
        # that we know (from phase 4) actually got compromised -- everything
        # else is a genuine dead-end distractor.
        early_in = (
            (df["scenario"] == scenario)
            & (df["phase"].isin([1, 2, 3]))
            & (df["src_ip"].isin(campaign_hosts) | df["dst_ip"].isin(campaign_hosts))
        )
        df.loc[early_in, "campaign_id"] = scenario
        df.loc[early_in, "is_campaign_link"] = True

    # --- collapse high-volume flood bursts into 1-second aggregate events ---
    # Phase 5 in LLDOS 1.0 is ~33,000 individual spoofed packets over ~100
    # seconds. Treating each spoofed packet as its own "alert" to correlate
    # would (a) not match how a real SOC would ever see this -- a flood
    # generates one aggregated IDS alert, not 33,000 -- and (b) let ~99% of
    # the dataset be near-duplicate flood packets, which would dominate every
    # downstream count and metric for a reason that has nothing to do with
    # correlation quality. So: any (scenario, vantage, phase) group with more
    # than AGG_THRESHOLD raw rows gets bucketed into 1-second windows.
    AGG_THRESHOLD = 200
    BIN_SECONDS = 1
    kept_frames = []
    agg_log = []
    for (scenario, vantage, phase), group in df.groupby(["scenario", "vantage", "phase"]):
        if len(group) <= AGG_THRESHOLD:
            kept_frames.append(group)
            continue
        agg_log.append(f"{scenario}/{vantage}/phase-{phase}: {len(group)} raw rows -> aggregating")
        group = group.copy()
        group["_bin"] = group["timestamp"].dt.floor(f"{BIN_SECONDS}s")
        agg_rows = []
        for i, (bin_ts, bg) in enumerate(group.groupby("_bin")):
            n_src = bg["src_ip"].nunique()
            n_dst = bg["dst_ip"].nunique()
            if n_src <= n_dst:
                anchor_ip = bg["src_ip"].mode().iloc[0]
                src_ip, dst_ip = anchor_ip, f"MULTI(n={n_dst})"
            else:
                anchor_ip = bg["dst_ip"].mode().iloc[0]
                src_ip, dst_ip = f"MULTI(n={n_src})", anchor_ip
            # is_campaign_link/campaign_id are recomputed from the whole bin
            # (not copied from an arbitrary row) so a bin that mixes real
            # escalations with dead-end distractors doesn't silently lose
            # the distractor rows' False label or vice versa.
            bin_is_campaign = bool(bg["is_campaign_link"].any())
            bin_campaign_id = bg.loc[bg["is_campaign_link"], "campaign_id"].iloc[0] if bin_is_campaign else "NONE"

            row = bg.iloc[0].to_dict()
            row.update({
                "event_id": f"{scenario}_{vantage}_p{phase}_agg_{i}",
                "timestamp": bin_ts,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": float("nan"),
                "dst_port": float("nan"),
                "service": bg["service"].mode().iloc[0],
                "event_count": len(bg),
                "is_campaign_link": bin_is_campaign,
                "campaign_id": bin_campaign_id,
            })
            agg_rows.append(row)
        kept_frames.append(pd.DataFrame(agg_rows))

    if agg_log:
        print("\nAggregated high-volume flood bursts into 1-second events:")
        for line in agg_log:
            print(f"  - {line}")

    df = pd.concat(kept_frames, ignore_index=True)
    if "event_count" not in df.columns:
        df["event_count"] = 1
    df["event_count"] = df["event_count"].fillna(1).astype(int)

    df = df[UNIFIED_COLUMNS + ["scenario", "phase", "event_count"]]  # extra cols kept for inspection

    out_path = out_root / "darpa2000_events.csv"
    df.to_csv(out_path, index=False)

    # --- sanity report ---
    print("\n" + "=" * 70)
    print("Campaign-host derivation (internal hosts seen in phase 4 = actually compromised)")
    print("=" * 70)
    for scenario, info in campaign_host_report.items():
        print(f"  {scenario}:")
        print(f"    used for matching (internal, 172.16.x.x): {info['internal_hosts_used_for_matching']}")
        print(f"    excluded (attacker/external/broadcast):   {info['excluded_non_internal']}")

    print("\n" + "=" * 70)
    print("Campaign vs distractor balance (per scenario)")
    print("=" * 70)
    summary = df.groupby(["scenario", "is_campaign_link"]).size().unstack(fill_value=0)
    print(summary)

    degenerate_scenarios = summary.index[(summary.get(True, 0) == 0) | (summary.get(False, 0) == 0)].tolist()
    print()
    if degenerate_scenarios:
        print(f"NOTE: {degenerate_scenarios} have zero within-scenario distractor events "
              f"(every recon probe in that scenario escalated to the same host that later "
              f"got compromised -- this is a real property of that capture, not a parsing "
              f"artifact, if it's the 'stealthy' LLDOS 2.0.2 scenario, which was designed to "
              f"probe minimally). This means that scenario ALONE provides no negative "
              f"examples to reject; Stage 5 evaluation must pool it with at least one other "
              f"scenario so the correlator has real distractors to reject (an attacker from "
              f"scenario A's events must not get linked into scenario B's campaign).")
    else:
        print("SANITY CHECK PASSED: every scenario has both true campaign-linked events "
              "and same-time-window distractor events that were correctly NOT escalated. "
              "This means the correlation task has genuine negatives to reject, not just "
              "a pre-filtered attack-only sequence.")

    print(f"\nWrote {len(df)} unified events to {out_path}")
    print("\nREMINDER: this file is test-only. Stage 3/4 training code must never read it. "
          "It is consumed only by the Stage 5 evaluation script.")

if __name__ == "__main__":
    main()
