#!/usr/bin/env python3
"""
Stage 4: train the DQN correlation agent on CICIDS2017 training campaigns,
tracking training curves and periodically evaluating on the held-out
validation campaigns (from Stage 2's campaign-level split). DARPA2000 is
never read by this script -- it is reserved entirely for Stage 5.

Usage:
    python scripts/stage4_train_dqn.py [data/processed] [data/processed/stage4]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.correlation_env import CorrelationEnv, STATE_DIM
from src.dqn_agent import DQNAgent
from src.leakage import assert_no_campaign_leakage, assert_dataset_is_test_only

N_EPISODES = 3000
EVAL_EVERY = 100
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 2000   # linear decay from EPS_START to EPS_END over this many episodes
RNG_SEED = 42

DARPA_SOURCE_NAMES = ["darpa2000_lldos1.0", "darpa2000_lldos2.0.2"]


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def run_episode(env, agent, campaign_id, epsilon, train=True):
    state = env.reset(campaign_id)
    if state is None:
        return None
    total_reward = 0.0
    n_steps = 0
    tp = fp = fn = tn = 0
    while True:
        action = agent.act(state, epsilon)
        result = env.step(action)
        total_reward += result.reward
        n_steps += 1

        if result.info["true_link"] and action == 1:
            tp += 1
        elif not result.info["true_link"] and action == 1:
            fp += 1
        elif result.info["true_link"] and action == 0:
            fn += 1
        else:
            tn += 1

        if train:
            agent.remember(state, action, result.reward,
                            result.state if not result.done else None, result.done)
            agent.learn()

        if result.done:
            break
        state = result.state

    return {
        "total_reward": total_reward, "n_steps": n_steps,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
    out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/stage4")
    out_root.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_root / "cicids2017_events.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], format="mixed")
    split = pd.read_csv(data_root / "cicids2017_split.csv")

    # --- leak-free split, re-verified here (defense in depth, not just trusted from Stage 2) ---
    train_campaigns = split.loc[split["split"] == "train", "campaign_id"].tolist()
    val_campaigns = split.loc[split["split"] == "val", "campaign_id"].tolist()
    train_ids = events.loc[events["campaign_id"].isin(train_campaigns), "campaign_id"]
    val_ids = events.loc[events["campaign_id"].isin(val_campaigns), "campaign_id"]
    leak_check = assert_no_campaign_leakage(train_ids, val_ids, context="stage4 dqn train/val")
    assert_dataset_is_test_only(events, "source_dataset", DARPA_SOURCE_NAMES, context="stage4 training data")
    print(f"Leak-free split confirmed: {leak_check}")
    print(f"Train campaigns: {train_campaigns}")
    print(f"Val campaigns:   {val_campaigns}")
    print(f"Confirmed: no darpa2000_* rows present in this training dataframe (DARPA is test-only, Stage 5 only).")

    env = CorrelationEnv(events, rng_seed=RNG_SEED)
    agent = DQNAgent(state_dim=STATE_DIM, seed=RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)

    history = []
    eval_history = []

    for episode in range(1, N_EPISODES + 1):
        epsilon = epsilon_at(episode)
        campaign_id = train_campaigns[rng.integers(0, len(train_campaigns))]
        result = run_episode(env, agent, campaign_id, epsilon, train=True)
        if result is None:
            continue

        precision, recall, f1 = precision_recall_f1(result["tp"], result["fp"], result["fn"])
        history.append({
            "episode": episode, "campaign_id": campaign_id, "epsilon": epsilon,
            "total_reward": result["total_reward"], "n_steps": result["n_steps"],
            "precision": precision, "recall": recall, "f1": f1,
            "avg_loss": np.mean(agent.loss_history[-result["n_steps"]:]) if agent.loss_history else None,
        })

        if episode % EVAL_EVERY == 0:
            val_tp = val_fp = val_fn = val_tn = 0
            val_reward = 0.0
            for vc in val_campaigns:
                vres = run_episode(env, agent, vc, epsilon=0.0, train=False)
                if vres is None:
                    continue
                val_tp += vres["tp"]; val_fp += vres["fp"]; val_fn += vres["fn"]; val_tn += vres["tn"]
                val_reward += vres["total_reward"]
            vp, vr, vf1 = precision_recall_f1(val_tp, val_fp, val_fn)
            eval_history.append({
                "episode": episode, "val_precision": vp, "val_recall": vr, "val_f1": vf1,
                "val_reward": val_reward, "val_tp": val_tp, "val_fp": val_fp,
                "val_fn": val_fn, "val_tn": val_tn,
            })
            print(f"[ep {episode:5d}] eps={epsilon:.3f} train_f1(last)={f1:.3f} "
                  f"VAL: P={vp:.3f} R={vr:.3f} F1={vf1:.3f} reward={val_reward:.1f} "
                  f"(tp={val_tp} fp={val_fp} fn={val_fn} tn={val_tn})")

    history_df = pd.DataFrame(history)
    eval_df = pd.DataFrame(eval_history)
    history_df.to_csv(out_root / "training_history.csv", index=False)
    eval_df.to_csv(out_root / "eval_history.csv", index=False)
    joblib.dump(agent.policy_net, out_root / "dqn_policy_net.joblib")

    # --- honest stability assessment: compare first vs last third of training ---
    n = len(eval_df)
    first_third = eval_df.iloc[: n // 3]
    last_third = eval_df.iloc[-(n // 3):]  # NOT `-n // 3`: that's (-n)//3 by precedence, off by one unless n%3==0
    stability_report = {
        "val_f1_first_third_mean": float(first_third["val_f1"].mean()) if len(first_third) else None,
        "val_f1_last_third_mean": float(last_third["val_f1"].mean()) if len(last_third) else None,
        "val_f1_last_third_std": float(last_third["val_f1"].std()) if len(last_third) else None,
        "val_reward_first_third_mean": float(first_third["val_reward"].mean()) if len(first_third) else None,
        "val_reward_last_third_mean": float(last_third["val_reward"].mean()) if len(last_third) else None,
    }

    print("\n" + "=" * 70)
    print("Stability assessment (first third of training vs last third)")
    print("=" * 70)
    for k, v in stability_report.items():
        print(f"  {k}: {v}")

    improved = (stability_report["val_f1_last_third_mean"] or 0) > (stability_report["val_f1_first_third_mean"] or 0)
    stable = (stability_report["val_f1_last_third_std"] or 1.0) < 0.15
    print()
    if improved and stable:
        print("VERDICT: learning improved and the last third is reasonably stable (std < 0.15).")
    elif improved and not stable:
        print("VERDICT: learning improved on average but the last third is NOT stable (std >= 0.15) "
              "-- report this honestly as noisy/unstable convergence, do not claim a clean result.")
    else:
        print("VERDICT: no clear improvement from first to last third of training -- report this "
              "honestly; do not claim the agent learned if the validation curve doesn't show it.")

    with open(out_root / "stability_report.json", "w") as f:
        json.dump(stability_report, f, indent=2)

    print(f"\nSaved training history, eval history, stability report, and policy network to {out_root}/")


if __name__ == "__main__":
    main()
