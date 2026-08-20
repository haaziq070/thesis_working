#!/usr/bin/env python3
"""
Stage 5: the headline evaluation. Reconstruct campaigns on DARPA2000 --
external data this project did not generate, and which no training or
tuning code in Stages 3/4 has ever read -- using the DQN trained in Stage 4,
and compare against a simple rule-based baseline using clustering-
appropriate metrics (Adjusted Rand Index, kill-chain completeness).

IMPORTANT DISCLOSED LIMITATION: Stage 3's classifier was trained on
CICIDS2017's flow-statistics feature schema (122 ML features -- packet
counts, inter-arrival-time statistics, header byte stats, etc.). DARPA2000's
events come from parsed tcpdump session records with a structurally
different, much sparser schema (service/ports/IPs/duration only -- no flow
statistics exist to compute). Stage 3's model architecturally cannot be
applied to DARPA data; there is no bridging feature set. For DARPA, this
script uses the dataset's own ground-truth-derived attack_type/
kill_chain_stage (already present in darpa2000_events.csv from Stage 2's
parsing of DARPA's documented phase structure) as the "identification"
signal fed into the DQN's state. This is an IDEALIZED identification
condition specific to DARPA -- unlike CICIDS2017, where Stage 4 trained
against Stage 3's actual predictions, including its real errors. Report this
explicitly: DARPA's correlation task is being evaluated under a more
favorable identification assumption than CICIDS2017's training ever was.
This is a genuine, structural cross-dataset limitation (not fixable without
building a second, DARPA-schema-specific identifier the proposal never
scoped), and belongs in Stage 6.

Usage:
    python scripts/stage5_evaluate_darpa.py [data/processed] [data/processed/stage5]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.correlation_env import _feature_vector
from src.campaign_reconstruction import reconstruct_campaigns, rule_based_decision, make_dqn_decision_fn
from src.evaluation_metrics import compute_ari, kill_chain_completeness, pairwise_precision_recall


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
    out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/stage5")
    out_root.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_root / "darpa2000_events.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], format="mixed")
    # see module docstring: DARPA has no Stage-3-compatible feature schema, so its own
    # ground-truth-derived attack_type/kill_chain_stage stand in for the "predicted_*"
    # fields here -- an idealized identification condition, disclosed, not a realistic
    # prediction. NOTE: attack_type and kill_chain_stage are two SEPARATE fields derived
    # independently from DARPA's phase number (DARPA_PHASE_TO_ATTACK_TYPE vs
    # DARPA_PHASE_TO_STAGE) -- kill_chain_stage must be aliased directly here, not
    # re-derived from attack_type, since attack_type's vocabulary (recon/exploit/
    # backdoor_install/ddos) doesn't match the shared recon/exploit/install/action
    # vocabulary predicted_kill_chain_stage needs (this was the exact bug fixed in
    # src/correlation_env.py -- see docs/stage5_findings.md).
    events["predicted_attack_type"] = events["attack_type"]
    events["predicted_kill_chain_stage"] = events["kill_chain_stage"]

    print(f"Loaded {len(events)} DARPA2000 events (source_dataset values: "
          f"{sorted(events['source_dataset'].unique())}) -- confirmed external, never touched "
          f"by Stage 3 or Stage 4 training.")

    policy_net = joblib.load(data_root / "stage4" / "dqn_policy_net.joblib")
    dqn_decision_fn = make_dqn_decision_fn(policy_net, _feature_vector)

    print("\nReconstructing campaigns with the trained DQN (greedy policy, no exploration)...")
    dqn_assignment = reconstruct_campaigns(events, dqn_decision_fn)

    print("Reconstructing campaigns with the rule-based baseline "
          "(link iff shared IP AND within 30 minutes)...")
    baseline_assignment = reconstruct_campaigns(events, rule_based_decision)

    results = {}
    for name, assignment in [("dqn", dqn_assignment), ("rule_based_baseline", baseline_assignment)]:
        ari = compute_ari(events, assignment)
        completeness = kill_chain_completeness(events, assignment)
        pairwise = pairwise_precision_recall(events, assignment)
        n_predicted_clusters = len(set(assignment.values()))
        results[name] = {
            "adjusted_rand_index": ari,
            "n_predicted_clusters": n_predicted_clusters,
            "kill_chain_completeness": completeness,
            "pairwise": pairwise,
        }

    print("\n" + "=" * 70)
    print("HEADLINE RESULT: Adjusted Rand Index (clustering quality vs ground truth)")
    print("=" * 70)
    print(f"  DQN (Stage 4 trained agent):  ARI = {results['dqn']['adjusted_rand_index']:.4f} "
          f"({results['dqn']['n_predicted_clusters']} predicted clusters)")
    print(f"  Rule-based baseline:          ARI = {results['rule_based_baseline']['adjusted_rand_index']:.4f} "
          f"({results['rule_based_baseline']['n_predicted_clusters']} predicted clusters)")
    diff = results['dqn']['adjusted_rand_index'] - results['rule_based_baseline']['adjusted_rand_index']
    print(f"  Difference (DQN - baseline): {diff:+.4f}")

    print("\n" + "=" * 70)
    print("Pairwise link precision/recall/F1 (a second, more directly interpretable metric --")
    print("ARI's chance-correction can behave counter-intuitively when partitions have very")
    print("different cluster-count/size profiles, which is the case here; cross-check against this)")
    print("=" * 70)
    for name in ["dqn", "rule_based_baseline"]:
        p = results[name]["pairwise"]
        print(f"  {name:22s} precision={p['precision']:.3f} recall={p['recall']:.3f} f1={p['f1']:.3f} "
              f"(tp={p['tp_pairs']} fp={p['fp_pairs']} fn={p['fn_pairs']})")

    if results["dqn"]["pairwise"]["f1"] > results["rule_based_baseline"]["pairwise"]["f1"]:
        print("\n  -> DQN outperforms the naive rule-based baseline on this external test set (by pairwise F1).")
    else:
        print("\n  -> DQN did NOT outperform the naive baseline on this external test set (by pairwise F1) "
              "-- report this honestly, do not reframe the comparison to hide it.")

    print("\n" + "=" * 70)
    print("Kill-chain completeness per true campaign")
    print("=" * 70)
    for name in ["dqn", "rule_based_baseline"]:
        print(f"\n  [{name}]")
        for campaign_id, info in results[name]["kill_chain_completeness"].items():
            frag = " (FRAGMENTED across multiple predicted clusters)" if info["fragmentation"] else ""
            print(f"    {campaign_id}: completeness={info['completeness']:.2f} "
                  f"true_stages={info['true_stages']} captured={info['captured_stages']}{frag}")

    # --- crosstab heatmap: true campaign vs predicted cluster, for both methods ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name, assignment in [(axes[0], "DQN", dqn_assignment), (axes[1], "Rule-based baseline", baseline_assignment)]:
        events_copy = events.copy()
        events_copy["predicted_cluster"] = events_copy["event_id"].map(assignment)
        ct = pd.crosstab(events_copy["campaign_id"], events_copy["predicted_cluster"])
        im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(ct.columns)))
        ax.set_xticklabels(ct.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel("predicted cluster id")
        ax.set_title(f"{name}\n(true campaign vs predicted cluster)")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = ct.values[i, j]
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                             color="white" if v > ct.values.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_root / "campaign_reconstruction_crosstab.png", dpi=150)
    print(f"\nSaved crosstab visualization to {out_root / 'campaign_reconstruction_crosstab.png'}")

    # --- save everything ---
    events_out = events.copy()
    events_out["dqn_predicted_cluster"] = events_out["event_id"].map(dqn_assignment)
    events_out["baseline_predicted_cluster"] = events_out["event_id"].map(baseline_assignment)
    events_out.to_csv(out_root / "darpa2000_reconstruction.csv", index=False)

    with open(out_root / "stage5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved full reconstruction assignments and results JSON to {out_root}/")
    print("\nSCOPE REMINDER: DARPA2000 was read ONLY by this script. Stage 3 training data, "
          "Stage 4 training data, and Stage 4's leak checks all excluded darpa2000_* rows -- "
          "verified in code at each of those stages, not just asserted here.")
    print("\nLIMITATION REMINDER: DARPA's 'predicted_attack_type' is ground-truth-derived "
          "(idealized), not a genuine Stage-3 classifier prediction -- Stage 3's model cannot "
          "run on DARPA's feature schema. See module docstring and docs/stage5_findings.md.")


if __name__ == "__main__":
    main()
