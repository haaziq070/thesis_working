# RL-Based Adaptive Alert Correlation for Multi-Stage Attack Detection

MS thesis project. Built stage by stage; see `docs/` for design notes and
per-stage findings. Scope commitments (never violated — checked explicitly
in Stage 6): this is a post-hoc correlation system, not a real-time
detector, and reinforcement learning is used only for the link/don't-link
correlation decision — attack typing is a separate supervised step (Stage 3).

**Start here for the full picture:** `docs/stage6_limitations.md` (honest
threats-to-validity) and `docs/stage5_findings.md` (the headline result —
which is not a clean win for the RL approach, and that's the point).

## Status

- [x] Stage 1 — data acquisition (DARPA2000 scripted; CICIDS2017 manual; UNSW-NB15 manual, obtained late — see Stage 3b)
- [x] Stage 2 — preprocessing / unified schema / campaign-level split
- [x] Stage 3 — supervised attack-identification classifier
- [x] Stage 4 — DQN correlation agent
- [x] Stage 5 — campaign reconstruction and evaluation (headline result)
- [x] Stage 6 — threats to validity / limitations

## Layout

```
data/raw/           downloaded datasets, untouched
data/processed/      Stage 2+ outputs (unified schema, splits, models, results)
scripts/             one script per pipeline step, numbered by stage
src/                 shared library code (schema, env, agent, metrics)
docs/                design notes, manual-download instructions, per-stage findings
venv/                Python 3.12 virtualenv, --system-site-packages (not committed)
```

## Setup on a fresh Linux machine

Dependencies install via `apt`, not `pip` — PyPI downloads repeatedly
stalled under this project's network conditions, while apt was fast and
reliable every time (see `requirements.txt` for the full explanation and a
pip-based fallback if your machine doesn't have this problem).

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-numpy python3-scipy \
    python3-pandas python3-sklearn python3-matplotlib python3-joblib \
    python3-requests python3-tqdm unzip curl aria2

python3 -m venv --system-site-packages venv
source venv/bin/activate
python3 -c "import numpy, scipy, pandas, sklearn, matplotlib, joblib; print('OK')"
```

## Running the full pipeline, in order

Each stage's script reads the previous stage's output from `data/processed/`.
Run them in this order:

```bash
source venv/bin/activate

# Stage 1 — data acquisition
bash scripts/stage1_download_darpa2000.sh data/raw/darpa2000
python scripts/stage1_verify_darpa2000.py data/raw/darpa2000
# CICIDS2017 needs your own manual step first — see docs/stage1_manual_downloads.md,
# then: bash scripts/stage1_download_cicids2017.sh data/raw/cicids2017

# Stage 2 — unified schema, campaign labels, leak-free splits
python scripts/stage2_parse_darpa2000.py data/raw/darpa2000 data/processed
# stage2_parse_cicids2017.py depends on Stage 3's trained model for predicted_attack_type
# (see the ordering note below) -- run Stage 3 first, then come back to this:
python scripts/stage2_parse_cicids2017.py data/raw/cicids2017/CSVs data/processed

# Stage 3 — supervised identifier (train BEFORE the Stage 2 CICIDS2017 parse above,
# on the very first run, since that parse enriches events with this model's predictions;
# on a fresh clone, run stage3 first, then stage2_parse_cicids2017.py, in that order)
python scripts/stage3_train_identifier.py data/raw/cicids2017/CSVs data/processed/stage3
python scripts/stage3_plot_confusion_matrix.py data/processed/stage3

# Stage 4 — DQN correlation agent (CICIDS2017 only; DARPA2000 never touched here)
python scripts/stage4_train_dqn.py data/processed data/processed/stage4
python scripts/stage4_plot_training_curves.py data/processed/stage4

# Stage 5 — headline evaluation on DARPA2000 (external, held out, read only here)
python scripts/stage5_evaluate_darpa.py data/processed data/processed/stage5

# Stage 3b (optional, secondary) — independent methodology cross-check on UNSW-NB15,
# using its own canonical split; does not touch Stage 3's actual CICIDS2017 model
python scripts/stage3b_unsw_identifier.py data/raw/unsw-nb15/CSVs data/processed/stage3b
python scripts/stage3b_plot_confusion_matrix.py data/processed/stage3b
```

**Actual first-run dependency order** (since Stage 2's CICIDS2017 parser
calls Stage 3's trained model to compute `predicted_attack_type`, but Stage
3 trains on the *raw* CICIDS2017 CSVs, not Stage 2's output — so there's no
circular dependency, just a specific order):

1. `stage1_download_darpa2000.sh` + `stage1_verify_darpa2000.py`
2. `stage1_download_cicids2017.sh` (after your manual step)
3. `stage2_parse_darpa2000.py` (independent of everything else)
4. `stage3_train_identifier.py` (reads raw CICIDS2017 CSVs directly)
5. `stage2_parse_cicids2017.py` (reads raw CICIDS2017 CSVs + Stage 3's saved model)
6. `stage4_train_dqn.py` (reads Stage 2's CICIDS2017 output)
7. `stage5_evaluate_darpa.py` (reads Stage 2's DARPA output + Stage 4's trained agent)

Plotting scripts (`stage3_plot_confusion_matrix.py`,
`stage4_plot_training_curves.py`) can run any time after their stage's main
script.

## Where the results are

- `data/processed/stage3/` — trained classifier, confusion matrix, per-class report
- `data/processed/stage4/` — trained DQN (current, correct), training/eval history, curves. `stage4_run1_fn5x/` and `stage4_run2_fn2.5x/` keep the two disclosed reward-tuning attempts that predate a later state-vector correctness fix (see next line) — kept on disk rather than deleted, for anyone comparing before/after.
- `data/processed/stage5/` — DARPA2000 reconstruction, headline metrics, crosstab visualization (current, correct). `stage5_buggy_run/` is the pre-fix evaluation.
- A full-codebase correctness review (see conversation history / `docs/stage5_findings.md`'s "Correctness note") found a real bug: one of the DQN's 7 state features was silently constant for every DARPA2000 pair due to a label-vocabulary mismatch between the two datasets. It was fixed in `src/correlation_env.py`, Stage 2's CICIDS2017 parser was re-run to add a corrected `predicted_kill_chain_stage` column, Stage 4 was retrained, and Stage 5 was re-evaluated. `stage4/` and `stage5/` hold the corrected results; the `*_buggy_run`/`_fn5x`/`_fn2.5x` directories hold what came before, disclosed rather than discarded.
- `data/processed/stage3b/` — Stage 3b's independent UNSW-NB15 models (naive + honest), confusion matrices, feature importances
- `docs/stage{2,3,3b,4,5,6}_findings.md` (or `stage6_limitations.md`) — the honest write-up for each stage, viva-ready
- `docs/defense_summary.html` — the defense-ready summary artifact, source file (also published live, see the link shared in conversation)
