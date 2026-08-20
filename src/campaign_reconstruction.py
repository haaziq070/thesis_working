"""
Stage 5: turn a stream of pairwise link/don't-link decisions into actual
campaign clusters, using ONE shared procedure for both the trained DQN and
the rule-based baseline (so the comparison between them isolates the
decision function, not the reconstruction algorithm around it).

Algorithm: process events in chronological order. Maintain a set of "open"
clusters, each represented by its most recently added event (the anchor)
and its last-update time. For each new event, compare it against every open
cluster whose anchor is within TIME_WINDOW_SECONDS (clusters older than that
are considered closed -- a real SOC does not keep comparing new alerts
against an incident from hours ago indefinitely). Among clusters that vote
"link", join whichever was most recently active. If none vote link, the
event starts a new cluster.

This is deliberately the same production-style procedure a real correlation
engine would run online -- decide against every recent open incident, not
just the immediately previous event -- unlike Stage 4's training episodes,
which only ever compared against a single evolving anchor for tractability.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIME_WINDOW_SECONDS = 3600.0  # matches Stage 4's TIME_CAP_SECONDS


@dataclass
class Cluster:
    cluster_id: int
    anchor_event: pd.Series
    last_update: pd.Timestamp
    member_event_ids: list = field(default_factory=list)


def reconstruct_campaigns(events_df, decision_fn, time_window_seconds=TIME_WINDOW_SECONDS):
    """
    events_df: dataframe with at least [event_id, timestamp, src_ip, dst_ip,
        service, predicted_attack_type], sorted or not (will be sorted here).
    decision_fn(anchor_event: pd.Series, candidate_event: pd.Series) -> bool
        True means "link".
    Returns a Series aligned to events_df.index giving each event's assigned
    cluster_id (int, 0-indexed, unique per predicted campaign).
    """
    events_df = events_df.sort_values("timestamp").reset_index(drop=False)  # keep original index as 'index'
    open_clusters = []  # list of Cluster, most-recently-updated last
    assignment = {}
    next_cluster_id = 0

    for _, event in events_df.iterrows():
        candidates = [
            c for c in open_clusters
            if (event["timestamp"] - c.last_update).total_seconds() <= time_window_seconds
        ]
        # most-recently-active candidates checked first
        candidates.sort(key=lambda c: c.last_update, reverse=True)

        linked_cluster = None
        for c in candidates:
            if decision_fn(c.anchor_event, event):
                linked_cluster = c
                break

        if linked_cluster is not None:
            linked_cluster.anchor_event = event
            linked_cluster.last_update = event["timestamp"]
            linked_cluster.member_event_ids.append(event["event_id"])
            assignment[event["event_id"]] = linked_cluster.cluster_id
        else:
            c = Cluster(cluster_id=next_cluster_id, anchor_event=event,
                        last_update=event["timestamp"], member_event_ids=[event["event_id"]])
            open_clusters.append(c)
            assignment[event["event_id"]] = next_cluster_id
            next_cluster_id += 1

    return assignment


def rule_based_decision(anchor, candidate, time_window_minutes=30):
    """Naive baseline: link iff they share an IP AND fall within a short
    fixed window. No learning, no other signal -- the kind of static
    correlation rule a SIEM would ship with out of the box."""
    within_time = (candidate["timestamp"] - anchor["timestamp"]).total_seconds() <= time_window_minutes * 60
    shares_ip = (
        anchor["src_ip"] == candidate["src_ip"] or anchor["src_ip"] == candidate["dst_ip"]
        or anchor["dst_ip"] == candidate["src_ip"] or anchor["dst_ip"] == candidate["dst_ip"]
    )
    return bool(within_time and shares_ip)


def make_dqn_decision_fn(policy_net, feature_fn):
    def decision_fn(anchor, candidate):
        state = feature_fn(anchor, candidate)
        q = policy_net.predict(state.reshape(1, -1))[0]
        return bool(np.argmax(q) == 1)
    return decision_fn
