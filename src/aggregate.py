"""
Shared high-volume-burst aggregation, used by every Stage-2 parser.

Rationale (established while parsing DARPA2000's DDoS flood phase, and
reused here for CICIDS2017's DoS/portscan/DDoS floods for the same reason):
a real SOC never receives one alert per raw packet/flow during a flood --
it receives one aggregated alert per burst. Leaving tens of thousands of
near-duplicate flow records as individual "events" would (a) not match how
this data would actually reach a correlation engine, and (b) let sheer
repetition dominate every downstream count/metric for a reason that has
nothing to do with correlation quality. So any group of raw records that
exceeds a volume threshold gets collapsed into fixed-width time buckets,
with the bucket's dominant/anchor identity preserved and a count kept.
"""
import pandas as pd


def aggregate_high_volume(df, group_cols, threshold=200, bin_seconds=1,
                           timestamp_col="timestamp", src_col="src_ip", dst_col="dst_ip"):
    """
    df: dataframe with at least [timestamp_col, src_col, dst_col] plus
        group_cols and whatever other columns should be carried through
        (the first row of each output bucket is used as the template for
        all non-aggregated columns).
    group_cols: columns whose combination defines a "burst" that should be
        volume-checked independently (e.g. ["source_dataset", "campaign_id",
        "phase"] for DARPA, or ["source_dataset", "campaign_id", "attack_type"]
        for CICIDS2017).
    Returns (result_df, agg_log) where agg_log is a list of human-readable
    strings describing what got aggregated, for the parser to print.
    """
    kept = []
    agg_log = []

    for key, group in df.groupby(group_cols):
        if len(group) <= threshold:
            kept.append(group)
            continue

        key_str = key if isinstance(key, str) else "/".join(str(k) for k in key)
        agg_log.append(f"{key_str}: {len(group)} raw rows -> aggregating into {bin_seconds}s bins")

        g = group.copy()
        g["_bin"] = g[timestamp_col].dt.floor(f"{bin_seconds}s")
        agg_rows = []
        for i, (bin_ts, bg) in enumerate(g.groupby("_bin")):
            n_src = bg[src_col].nunique()
            n_dst = bg[dst_col].nunique()
            if n_src <= n_dst:
                anchor = bg[src_col].mode().iloc[0]
                src_val, dst_val = anchor, f"MULTI(n={n_dst})"
            else:
                anchor = bg[dst_col].mode().iloc[0]
                src_val, dst_val = f"MULTI(n={n_src})", anchor

            row = bg.iloc[0].to_dict()
            row[timestamp_col] = bin_ts
            row[src_col] = src_val
            row[dst_col] = dst_val
            row["event_count"] = len(bg)
            if "is_campaign_link" in bg.columns:
                row["is_campaign_link"] = bool(bg["is_campaign_link"].any())
            if "campaign_id" in bg.columns and "is_campaign_link" in bg.columns:
                true_rows = bg[bg["is_campaign_link"]]
                row["campaign_id"] = true_rows["campaign_id"].iloc[0] if len(true_rows) else "NONE"
            if "predicted_attack_type" in bg.columns:
                # unlike the grouped columns above, the classifier's predicted label is NOT
                # constant within a burst -- it varies row to row, including its real errors.
                # Majority vote within the bin, plus the agreement fraction so a bin where the
                # classifier was split down the middle doesn't look as confident as one where
                # it was unanimous.
                vc = bg["predicted_attack_type"].value_counts()
                row["predicted_attack_type"] = vc.index[0]
                row["predicted_attack_type_agreement"] = float(vc.iloc[0]) / len(bg)
            agg_rows.append(row)
        kept.append(pd.DataFrame(agg_rows))

    result = pd.concat(kept, ignore_index=True)
    if "event_count" not in result.columns:
        result["event_count"] = 1
    result["event_count"] = result["event_count"].fillna(1).astype(int)
    if "predicted_attack_type_agreement" in result.columns:
        # non-aggregated rows (below threshold) never went through the majority-vote
        # step above, so they trivially "agree" with themselves.
        result["predicted_attack_type_agreement"] = result["predicted_attack_type_agreement"].fillna(1.0)
    return result, agg_log
