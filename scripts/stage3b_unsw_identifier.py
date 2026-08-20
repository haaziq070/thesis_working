#!/usr/bin/env python3
"""
Stage 3b (optional, secondary cross-check): train an independent Random
Forest identifier on UNSW-NB15, using the SAME methodology and leakage
discipline as Stage 3, but on a genuinely separate dataset -- not by
reusing Stage 3's CICIDS2017-trained model (architecturally impossible:
UNSW-NB15's feature schema, 49 columns of flow statistics, shares no
columns with CICIDS2017's 122). This checks whether the *approach*
(leakage-aware supervised classification of network flows) generalizes,
not whether one specific trained model does.

UNSW-NB15 ships its own canonical, pre-labeled train/test split
(UNSW_NB15_training-set.csv / testing-set.csv, 175,341 / 82,332 rows) --
genuinely independent of anything built in this project, not something we
constructed ourselves.

LEAKAGE FINDING, checked directly against the raw data before any model was
trained (not assumed from prior literature, though prior literature does
flag this): sttl (source TTL) is a near-perfect discriminator for reasons
that have nothing to do with attack behavior -- Normal traffic is
overwhelmingly sttl=31 while most attack categories are 97-100% sttl=254
(Shellcode 100%, Fuzzers 99.6%, Generic 99.4%, Reconnaissance 99.5%). This
is a simulation-configuration artifact (attack traffic generated from VMs
with a different default TTL/hop-count than the "normal" traffic generator),
not a learnable attack signature -- a model that keys on it would look
artificially strong while having learned nothing generalizable. dttl shows a
messier but directionally similar pattern. Both are trained WITH and
WITHOUT these two columns, and the difference is reported explicitly --
this is the demonstration, not just an assertion.

Usage:
    python scripts/stage3b_unsw_identifier.py [data/raw/unsw-nb15/CSVs] [data/processed/stage3b]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

NON_FEATURE_COLS = ["id", "attack_cat", "label"]
CATEGORICAL_COLS = ["proto", "service", "state"]
LEAKY_TTL_COLS = ["sttl", "dttl"]
NEAR_PERFECT_F1_THRESHOLD = 0.98
RNG_SEED = 42


def load(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def build_encoders(train_df, test_df, cols):
    encoders = {}
    for c in cols:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[c], test_df[c]]).astype(str))
        encoders[c] = le
    return encoders


def build_X(df, feature_cols, categorical_cols, encoders):
    X = df[feature_cols].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = encoders[c].transform(df[c].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def train_and_evaluate(name, train_df, test_df, feature_cols, encoders, out_root):
    X_train = build_X(train_df, feature_cols, CATEGORICAL_COLS, encoders)
    X_test = build_X(test_df, feature_cols, CATEGORICAL_COLS, encoders)
    y_train, y_test = train_df["attack_cat"].values, test_df["attack_cat"].values

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=20, min_samples_leaf=3,
        class_weight="balanced", n_jobs=2, random_state=RNG_SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n{'=' * 70}\n[{name}] classification report ({len(feature_cols)} features)\n{'=' * 70}")
    print(report_text)

    labels_sorted = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(out_root / f"confusion_matrix_{name}.csv")

    test_support = pd.Series(y_test).value_counts()
    flagged = []
    for cls, metrics in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if metrics["f1-score"] >= NEAR_PERFECT_F1_THRESHOLD:
            flagged.append((cls, metrics["f1-score"], int(test_support.get(cls, 0))))
    if flagged:
        print(f"  NEAR-PERFECT CLASSES in [{name}]:")
        for cls, f1, n in flagged:
            print(f"    - {cls}: F1={f1:.4f} (test_n={n})")

    joblib.dump(clf, out_root / f"random_forest_{name}.joblib")
    with open(out_root / f"classification_report_{name}.json", "w") as f:
        json.dump(report, f, indent=2)

    importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    importances.to_csv(out_root / f"feature_importances_{name}.csv")

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "flagged_near_perfect": [f[0] for f in flagged],
        "top_features": importances.head(8).to_dict(),
    }


def main():
    raw_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/unsw-nb15/CSVs")
    out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/stage3b")
    out_root.mkdir(parents=True, exist_ok=True)

    train_df = load(raw_root / "UNSW_NB15_training-set.csv")
    test_df = load(raw_root / "UNSW_NB15_testing-set.csv")
    print(f"Loaded {len(train_df)} training rows, {len(test_df)} testing rows "
          f"(UNSW-NB15's own canonical split -- not constructed by this project).")
    print(f"attack_cat classes: {sorted(train_df['attack_cat'].unique())}")

    all_feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    honest_feature_cols = [c for c in all_feature_cols if c not in LEAKY_TTL_COLS]
    encoders = build_encoders(train_df, test_df, CATEGORICAL_COLS)

    results = {}
    results["naive_with_ttl"] = train_and_evaluate(
        "naive_with_ttl", train_df, test_df, all_feature_cols, encoders, out_root
    )
    results["honest_without_ttl"] = train_and_evaluate(
        "honest_without_ttl", train_df, test_df, honest_feature_cols, encoders, out_root
    )

    print("\n" + "=" * 70)
    print("LEAKAGE DEMONSTRATION: naive (with sttl/dttl) vs honest (without)")
    print("=" * 70)
    n, h = results["naive_with_ttl"], results["honest_without_ttl"]
    print(f"  accuracy:    naive={n['accuracy']:.4f}  honest={h['accuracy']:.4f}  "
          f"delta={n['accuracy'] - h['accuracy']:+.4f}")
    print(f"  macro F1:    naive={n['macro_f1']:.4f}  honest={h['macro_f1']:.4f}  "
          f"delta={n['macro_f1'] - h['macro_f1']:+.4f}")
    print(f"  weighted F1: naive={n['weighted_f1']:.4f}  honest={h['weighted_f1']:.4f}  "
          f"delta={n['weighted_f1'] - h['weighted_f1']:+.4f}")
    if n["accuracy"] - h["accuracy"] > 0.02:
        print("\n  -> Removing sttl/dttl produces a real, measurable drop -- confirms these columns "
              "were doing meaningful (and, per the simulation-artifact argument, illegitimate) work "
              "in the naive model. The honest numbers are what should be reported as the headline.")
    else:
        print("\n  -> Removing sttl/dttl did not meaningfully change performance -- the suspected "
              "leakage may not be as load-bearing for this model/feature-set as the raw TTL "
              "distribution suggested. Report this too: a plausible leakage vector that DIDN'T "
              "turn out to matter much is still worth stating, not just the ones that did.")

    with open(out_root / "stage3b_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved both models, reports, confusion matrices, and results JSON to {out_root}/")


if __name__ == "__main__":
    main()
