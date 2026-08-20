#!/usr/bin/env python3
"""
Stage 2 (CICIDS2017 half): parse the BCCC re-extraction CSVs into the unified
event schema and derive campaign-membership ground truth.

CICIDS2017 does NOT ship native multi-stage campaign labels the way DARPA2000
does -- it's a set of separately-labeled, single-attack-type flow captures.
Worse for our purposes: every attack-type file uses the exact same fixed
attacker/victim IP pair (172.16.0.1 <-> 192.168.10.50), so "shared IP" -- the
signal DARPA's campaign derivation relied on -- is completely useless here;
every attack of every type shares the same two addresses. Campaigns here have
to be derived from TIME structure instead: attack-type blocks run close
together in time (small gaps) are treated as one multi-stage campaign (e.g.
Thursday's brute-force -> XSS -> SQLi block, which is documented in the
original CIC-IDS2017 paper as one red-team exercise), while blocks separated
by a large gap are treated as separate campaigns.

This is a DERIVED grouping, not official ground truth -- unlike DARPA's
phase structure, which came from the dataset's own file layout. Document this
distinction explicitly in the thesis (Stage 6 limitations).

Stage 4 addendum: every row is also scored by the trained Stage 3 classifier
(data/processed/stage3/random_forest_identifier.joblib) to get a
predicted_attack_type column, alongside the existing ground-truth
attack_type. This matters because the design commitment is "the identifier
labels events -> those labeled events become the input state for the
correlator" -- the DQN's state must see Stage 3's *actual* predictions,
including its real errors (e.g. weak Web_XSS/Web_Brute_Force recall, weak
Benign precision), not an idealized oracle label. Using ground truth here
would test Stage 4 in a condition Stage 5's real evaluation will never see.

To keep this memory-safe on a 2-core/4GB box (Stage 3's first training
attempt was OOM-killed at full width -- see stage3_train_identifier.py's
comments), each file is loaded at full column width only long enough to
compute predictions, then immediately trimmed back down to the lightweight
USECOLS subset plus the new prediction column before the next file loads.

Usage:
    python scripts/stage2_parse_cicids2017.py [data/raw/cicids2017/CSVs] [data/processed]
"""
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import UNIFIED_COLUMNS, CICIDS_LABEL_TO_STAGE, CICIDS_FILE_TO_DAY, CICIDS_LEAKAGE_COLUMNS
from src.aggregate import aggregate_high_volume
from src.leakage import assert_no_campaign_leakage

USECOLS = ["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "label"]
GAP_THRESHOLD_MINUTES = 60          # merge same-day attack blocks within this gap into one campaign
AGG_THRESHOLD = 200                 # per-campaign row count above which flood aggregation kicks in
BENIGN_MULTIPLIER = 5               # sample this many benign rows per attack row, per day
BENIGN_CAP = 50000                  # ...but never more than this many benign rows per day
RNG_SEED = 42
STAGE3_DIR = Path("data/processed/stage3")

ATTACK_FILES = [f for f, day in CICIDS_FILE_TO_DAY.items() if "benign" not in f]
BENIGN_FILES = [f for f, day in CICIDS_FILE_TO_DAY.items() if "benign" in f]


def load_stage3_model():
    clf = joblib.load(STAGE3_DIR / "random_forest_identifier.joblib")
    protocol_encoder = joblib.load(STAGE3_DIR / "protocol_encoder.joblib")
    return clf, protocol_encoder


def predict_attack_type(df_full, clf, protocol_encoder):
    """df_full: a raw CICIDS2017 CSV loaded at full column width. Returns an
    array of predicted labels, built with EXACTLY the same feature selection
    and ordering stage3_train_identifier.py used at training time (drop
    CICIDS_LEAKAGE_COLUMNS + source_file/day, protocol moved to the end and
    label-encoded, float32). Passing a DataFrame (not a bare ndarray) to
    sklearn lets it validate feature names match what the model was fit on --
    it will raise rather than silently mispredict if this ever drifts out of
    sync with Stage 3's own feature construction.
    """
    drop_cols = set(CICIDS_LEAKAGE_COLUMNS) | {"source_file", "day"}
    numeric_feature_cols = [c for c in df_full.columns if c not in drop_cols and c != "protocol"]
    X = df_full[numeric_feature_cols].astype(np.float32)
    X["protocol"] = protocol_encoder.transform(df_full["protocol"].astype(str)).astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return clf.predict(X)


def load_file_with_predictions(path, clf, protocol_encoder, usecols=USECOLS):
    df_full = pd.read_csv(path)
    predicted = predict_attack_type(df_full, clf, protocol_encoder)
    df = df_full[usecols].copy()
    df["predicted_attack_type"] = predicted
    del df_full
    gc.collect()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def main():
    raw_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/cicids2017/CSVs")
    out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading Stage 3 classifier from {STAGE3_DIR}/ to score every event with "
          f"predicted_attack_type...")
    clf, protocol_encoder = load_stage3_model()

    # --- load every attack-type file in full ---
    attack_frames = []
    for fname in ATTACK_FILES:
        path = raw_root / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        df = load_file_with_predictions(path, clf, protocol_encoder)
        df["day"] = CICIDS_FILE_TO_DAY[fname]
        df["source_file"] = fname
        attack_frames.append(df)
        print(f"  loaded+scored {fname}: {len(df)} rows")
    attacks = pd.concat(attack_frames, ignore_index=True)
    del attack_frames
    gc.collect()
    print(f"Loaded {len(attacks)} attack-type rows total.")

    # --- derive campaigns via time-gap clustering, per day ---
    attacks["campaign_id"] = "NONE"
    campaign_log = []
    rng = __import__("random").Random(RNG_SEED)

    for day, day_df in attacks.groupby("day"):
        # one block per source file: its own [min_ts, max_ts] time range
        blocks = (
            day_df.groupby("source_file")["timestamp"]
            .agg(["min", "max"])
            .sort_values("min")
            .reset_index()
        )
        campaign_idx = 0
        current_campaign_files = [blocks.iloc[0]["source_file"]]
        current_end = blocks.iloc[0]["max"]

        def flush(files, idx):
            cid = f"cicids2017_{day}_c{idx}"
            mask = attacks["source_file"].isin(files)
            attacks.loc[mask & (attacks["day"] == day), "campaign_id"] = cid
            campaign_log.append(f"{cid}: files={files}")

        for i in range(1, len(blocks)):
            row = blocks.iloc[i]
            gap = (row["min"] - current_end).total_seconds() / 60.0
            if gap <= GAP_THRESHOLD_MINUTES:
                current_campaign_files.append(row["source_file"])
                current_end = max(current_end, row["max"])
            else:
                flush(current_campaign_files, campaign_idx)
                campaign_idx += 1
                current_campaign_files = [row["source_file"]]
                current_end = row["max"]
        flush(current_campaign_files, campaign_idx)

    print("\nDerived campaigns (time-gap clustering, threshold={}min):".format(GAP_THRESHOLD_MINUTES))
    for line in campaign_log:
        print(f"  - {line}")

    attacks["is_campaign_link"] = True  # every row here is a genuine attack event
    attacks["kill_chain_stage"] = attacks["label"].map(CICIDS_LABEL_TO_STAGE).fillna("unknown")
    attacks["attack_type"] = attacks["label"]
    attacks["source_dataset"] = "cicids2017"
    attacks["tier"] = "network"  # see note below on why NOT "web"
    attacks["vantage"] = "N/A"

    # --- sample benign background as distractor negatives, per day ---
    benign_frames = []
    for fname in BENIGN_FILES:
        path = raw_root / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        day = CICIDS_FILE_TO_DAY[fname]
        # sample BEFORE scoring (not after) -- these benign files are the largest in the
        # dataset (up to ~495k rows) and scoring is the expensive/heavy-column step, so
        # trimming to the target sample size first keeps peak memory down to one sampled
        # subset's full-width slice rather than the whole file's.
        n_available = sum(1 for _ in open(path)) - 1
        n_attack_this_day = (attacks["day"] == day).sum()
        n_sample = min(BENIGN_CAP, max(1000, n_attack_this_day * BENIGN_MULTIPLIER), n_available)
        rng_np = np.random.default_rng(RNG_SEED)
        skiprows = sorted(rng_np.choice(range(1, n_available + 1), size=n_available - n_sample, replace=False))
        df_full = pd.read_csv(path, skiprows=skiprows)
        predicted = predict_attack_type(df_full, clf, protocol_encoder)
        df = df_full[USECOLS].copy()
        df["predicted_attack_type"] = predicted
        del df_full
        gc.collect()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["day"] = day
        df["source_file"] = fname
        df["campaign_id"] = "NONE"
        df["is_campaign_link"] = False
        df["kill_chain_stage"] = "unknown"
        df["attack_type"] = "benign"
        df["source_dataset"] = "cicids2017"
        df["tier"] = "network"
        df["vantage"] = "N/A"
        benign_frames.append(df)
        print(f"  sampled {n_sample} / {n_available} benign rows from {fname} "
              f"(day has {n_attack_this_day} attack rows)")

    benign = pd.concat(benign_frames, ignore_index=True) if benign_frames else pd.DataFrame()
    print(f"\nSampled {len(benign)} benign distractor rows total "
          f"(rule: {BENIGN_MULTIPLIER}x that day's attack row count, capped at {BENIGN_CAP}).")

    combined = pd.concat([attacks, benign], ignore_index=True)
    combined["event_id"] = [f"cicids2017_{i}" for i in range(len(combined))]

    # --- aggregate high-volume flood-type bursts, same policy as DARPA2000 ---
    combined, agg_log = aggregate_high_volume(
        combined, group_cols=["campaign_id", "attack_type"], threshold=AGG_THRESHOLD, bin_seconds=1
    )
    if agg_log:
        print(f"\nAggregated {len(agg_log)} high-volume group(s) into 1-second events:")
        for line in agg_log:
            print(f"  - {line}")

    combined["service"] = combined["protocol"]
    if "predicted_attack_type_agreement" not in combined.columns:
        combined["predicted_attack_type_agreement"] = 1.0
    # Correctness fix: this must be derived from predicted_attack_type via the SAME
    # CICIDS-vocabulary mapping used for the ground-truth kill_chain_stage above, not
    # reused/aliased from it -- the state features Stage 4 trains on must only ever see
    # predicted labels, never ground truth. (This mirrors the DARPA-side fix in
    # stage5_evaluate_darpa.py; see docs/stage5_findings.md for the bug this replaced.)
    combined["predicted_kill_chain_stage"] = combined["predicted_attack_type"].map(CICIDS_LABEL_TO_STAGE).fillna("unknown")
    combined = combined[UNIFIED_COLUMNS + ["day", "source_file", "event_count",
                                            "predicted_attack_type", "predicted_attack_type_agreement",
                                            "predicted_kill_chain_stage"]]
    out_path = out_root / "cicids2017_events.csv"
    combined.to_csv(out_path, index=False)

    # --- campaign-level train/val split, enforced and verified in code ---
    all_campaigns = sorted(c for c in attacks["campaign_id"].unique() if c != "NONE")
    rng.shuffle(all_campaigns)
    n_val = max(1, round(len(all_campaigns) * 0.3))
    val_campaigns = sorted(all_campaigns[:n_val])
    train_campaigns = sorted(all_campaigns[n_val:])

    train_ids = combined.loc[combined["campaign_id"].isin(train_campaigns), "campaign_id"]
    val_ids = combined.loc[combined["campaign_id"].isin(val_campaigns), "campaign_id"]
    check = assert_no_campaign_leakage(train_ids, val_ids, context="cicids2017 train/val")

    split_path = out_root / "cicids2017_split.csv"
    pd.DataFrame(
        [{"campaign_id": c, "split": "train"} for c in train_campaigns]
        + [{"campaign_id": c, "split": "val"} for c in val_campaigns]
    ).to_csv(split_path, index=False)

    print("\n" + "=" * 70)
    print("Campaign-level split (Stage 3/4 hyperparameter tuning must use this)")
    print("=" * 70)
    print(f"  train campaigns ({len(train_campaigns)}): {train_campaigns}")
    print(f"  val campaigns   ({len(val_campaigns)}): {val_campaigns}")
    print(f"  leakage check: {check}")

    print("\n" + "=" * 70)
    print("Event counts")
    print("=" * 70)
    print(f"  total unified events: {len(combined)}")
    print(f"  is_campaign_link=True : {combined['is_campaign_link'].sum()}")
    print(f"  is_campaign_link=False: {(~combined['is_campaign_link']).sum()}")
    print("\n  attack_type distribution:")
    print(combined["attack_type"].value_counts().to_string())

    print(f"\nWrote {len(combined)} unified events to {out_path}")
    print(f"Wrote campaign-level split to {split_path}")
    print("\nSCOPE NOTE: tier is set to 'network' for ALL CICIDS2017 events, including "
          "web_sql_injection/web_xss. These are still network-flow-level captures (5-tuple + "
          "flow statistics), not HTTP-layer WAF alerts (no method/URL/payload field exists in "
          "this schema). Genuine web/WAF-tier alerts still depend on the deferred DVWA + "
          "OWASP-CRS/ModSecurity source -- do not present CICIDS2017's web_* files as web-tier "
          "correlation input without this caveat.")

if __name__ == "__main__":
    main()
