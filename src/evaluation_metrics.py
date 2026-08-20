"""
Stage 5: correlation-appropriate evaluation metrics.

Campaign reconstruction is a clustering problem, not a classification
problem -- "did we group the right events together" -- so classification
metrics (precision/recall/F1 on individual link decisions, which Stage 4
used for training feedback) are not the right headline metric here.
Adjusted Rand Index is the standard clustering-comparison metric and is
what's used for the headline number; kill-chain completeness is a
domain-specific second metric asking a different question: even where
events did get grouped correctly, did the reconstructed cluster capture the
attack's full multi-stage narrative, or only a fragment of it.
"""
from itertools import combinations
from sklearn.metrics import adjusted_rand_score


def build_true_labels(events_df):
    """Ground-truth cluster labels for ARI: real campaign members share their
    campaign_id; every distractor ("NONE") event is its own singleton
    cluster, since distractor events are not truly related to EACH OTHER
    either -- treating them as one big "noise" cluster would incorrectly
    reward a predicted cluster that lumps multiple unrelated distractors
    together."""
    labels = []
    singleton_counter = 0
    for _, row in events_df.iterrows():
        if row["campaign_id"] == "NONE":
            labels.append(f"__singleton_{singleton_counter}")
            singleton_counter += 1
        else:
            labels.append(row["campaign_id"])
    return labels


def compute_ari(events_df, predicted_cluster_assignment):
    true_labels = build_true_labels(events_df)
    pred_labels = [predicted_cluster_assignment[eid] for eid in events_df["event_id"]]
    return adjusted_rand_score(true_labels, pred_labels)


def pairwise_precision_recall(events_df, predicted_cluster_assignment):
    """A second, complementary clustering metric alongside ARI. ARI's
    chance-correction is known to behave counter-intuitively when comparing
    partitions with very different cluster-count/size profiles (exactly the
    situation here: a handful of large predicted clusters vs a ground truth
    dominated by 134 true singletons) -- worth cross-checking against a more
    directly interpretable metric. Pairwise precision/recall asks a simpler
    question directly: of all pairs of events actually predicted to be
    linked (same predicted cluster), what fraction truly belong together
    (precision)? Of all pairs that truly belong together, what fraction did
    the method actually find (recall)?
    """
    events_df = events_df.copy()
    events_df["predicted_cluster"] = events_df["event_id"].map(predicted_cluster_assignment)

    true_pairs = set()
    for cid, grp in events_df.groupby("campaign_id"):
        if cid == "NONE":
            continue
        for a, b in combinations(grp["event_id"].tolist(), 2):
            true_pairs.add(frozenset((a, b)))

    pred_pairs = set()
    for cid, grp in events_df.groupby("predicted_cluster"):
        for a, b in combinations(grp["event_id"].tolist(), 2):
            pred_pairs.add(frozenset((a, b)))

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp_pairs": tp, "fp_pairs": fp, "fn_pairs": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_predicted_pairs": len(pred_pairs), "n_true_pairs": len(true_pairs),
    }


def kill_chain_completeness(events_df, predicted_cluster_assignment):
    """For each true campaign, find the predicted cluster that contains the
    most of its members (the 'best-matching' cluster), then measure what
    fraction of the true campaign's distinct kill-chain stages appear
    somewhere in that cluster. 1.0 = the reconstructed cluster captured the
    full multi-stage narrative; a low score means the campaign got
    fragmented across multiple predicted clusters (or diluted by noise) and
    the multi-stage story was lost.
    """
    events_df = events_df.copy()
    events_df["predicted_cluster"] = events_df["event_id"].map(predicted_cluster_assignment)

    results = {}
    for campaign_id in sorted(set(events_df["campaign_id"]) - {"NONE"}):
        camp = events_df[events_df["campaign_id"] == campaign_id]
        true_stages = set(camp["kill_chain_stage"]) - {"unknown"}
        if not true_stages:
            continue

        best_cluster = camp["predicted_cluster"].value_counts().idxmax()
        cluster_events = events_df[events_df["predicted_cluster"] == best_cluster]
        captured_stages = set(cluster_events.loc[cluster_events["campaign_id"] == campaign_id, "kill_chain_stage"]) - {"unknown"}

        completeness = len(captured_stages) / len(true_stages)
        n_predicted_clusters_touched = camp["predicted_cluster"].nunique()
        results[campaign_id] = {
            "true_stages": sorted(true_stages),
            "captured_stages": sorted(captured_stages),
            "completeness": completeness,
            "n_true_events": len(camp),
            "n_predicted_clusters_touched": int(n_predicted_clusters_touched),
            "fragmentation": int(n_predicted_clusters_touched) > 1,
        }
    return results
