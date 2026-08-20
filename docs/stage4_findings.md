# Stage 4 findings — DQN correlation agent

## Correctness note

A later full-codebase correctness review found that the 7th state feature
(stage_progression) was silently always 0 for DARPA2000 specifically (a
CICIDS-only vocabulary lookup that DARPA's labels never matched -- see
`docs/stage5_findings.md` for the full account). This did NOT affect the two
CICIDS2017 runs documented below -- for CICIDS2017, predicted_attack_type
does match that vocabulary, so the feature worked correctly in both runs'
training. The bug's impact was concentrated entirely in Stage 5's DARPA
evaluation. After the fix, Stage 4 was retrained a third time with the same
reward configuration as Run 2 (FN penalty 2.5x); its CICIDS2017 validation
numbers are essentially unchanged from Run 2 (mean F1 0.873 vs 0.889 --
within noise), consistent with the bug having no real effect here. That
corrected model is what `data/processed/stage4/` now holds and what Stage 5
evaluates. Runs 1 and 2 below remain accurate accounts of the reward-tuning
investigation on CICIDS2017; treat this note as the one addition to them.

## What's implemented

- `src/correlation_env.py`: the pairwise correlation environment. State =
  7 features built ONLY from Stage 3's *predicted* labels (never ground
  truth) plus time/IP/service structure. Action = discrete {don't-link,
  link}. Episodes = one campaign's true events (subsampled if very large,
  capped at 60/episode) chronologically interleaved with same-day distractor
  events; the anchor is seeded at the campaign's first true event (this
  project decides *whether two events belong together*, not *where a
  campaign starts* -- a deliberate, stated scope simplification).
- `src/dqn_agent.py`: DQN with a real 2-hidden-layer neural-network
  Q-function (sklearn `MLPRegressor` with `partial_fit`, not PyTorch --
  documented reason: no CPU PyTorch wheel via apt, and PyPI downloads of
  that size repeatedly stalled under this environment's network throttling,
  same issue Stage 3's setup hit before switching to apt), experience
  replay, a separate target network synced every 200 steps, epsilon-greedy
  exploration with linear decay, and a genuine discount factor (episodes
  have real multi-step structure: a wrong link moves the anchor and changes
  every subsequent comparison).
- `scripts/stage4_train_dqn.py`: trains on CICIDS2017's 4 train campaigns
  (from Stage 2's campaign-level split), evaluates on the 2 held-out val
  campaigns every 100 episodes, re-verifies the leak-free split AND that no
  `darpa2000_*` rows are present in the training data (defense in depth,
  not just trusted from Stage 2).

Trained only on CICIDS2017. DARPA2000 was never read by this stage --
`assert_dataset_is_test_only` checks this in code at the start of every
training run, not just asserted in prose.

## Two runs, both reported (not just the better one)

The first run's reward used a 5x asymmetric penalty for missed links
(false negative = -5.0 vs link reward = +1.0), matching the original design
justification (missing a real campaign link is worse than one bad
grouping). Its validation curve is genuinely disappointing: precision
oscillates noisily with no improving trend, recall stays pinned near 1.0
throughout -- i.e. the agent converged toward an "almost always link"
policy. This was diagnosed, not hand-waved: TD loss dropped 2.46 -> 1.41
over training (the network *was* learning to fit its own bootstrapped
targets), but the policy itself never discovered a sharper discriminator.
The likely mechanism: with the false-negative penalty dominating the reward
scale, "link when uncertain" is a locally rational strategy, and with only
4 training campaigns (one -- the Heartbleed one -- has just 12 events) there
isn't much state diversity to learn a sharper boundary from.

| metric (mean across 30 eval checkpoints) | Run 1 (FN=-5.0, 5x) | Run 2 (FN=-2.5, 2.5x) |
|---|---|---|
| val F1 | 0.838 (std 0.031) | **0.889** (std 0.034) |
| val precision | 0.730 (std 0.041) | **0.808** (std 0.048) |
| val recall | 0.984 (std 0.020) | 0.989 (std 0.023) |
| best single checkpoint F1 | 0.886 (ep 1200) | **0.946** (ep 1200) |

Reducing the asymmetry to 2.5x (still asymmetric -- a missed link is still
treated as worse than a false one, just not so dominant it swamps the
state-based signal) produced a real, substantial improvement across every
averaged metric, not just a lucky single checkpoint: +0.051 mean F1, +0.078
mean precision, at essentially unchanged (still near-perfect) recall. This
was one disclosed, one-time adjustment based on a specific diagnosed
failure mode -- not repeated tuning toward a target number, and both runs'
full results are kept in `data/processed/stage4_run1_fn5x/` and
`data/processed/stage4_run2_fn2.5x/` for inspection. `data/processed/stage4/`
(the version Stage 5 will use) is a copy of run 2.

## Honest characterization of run 2's training curve

This is NOT a clean, monotonically-improving curve, and it should not be
presented as one. Performance peaked around episodes 1100-1300 (F1 up to
0.946), then partially regressed in the back third (episodes 2200-2900 dip
as low as F1=0.818) before recovering somewhat by the final checkpoint
(F1=0.885). This is visible directly in `data/processed/stage4/training_curves.png`.

A methodological note worth keeping for the thesis write-up: the training
script's own automated stability check (compares the *first* third of
checkpoints' mean F1 against the *last* third's) reports "no clear
improvement" for run 2, because the first third's average happens to be
pulled up by the strong 1100-1300 window while the last third includes the
2200-2900 dip. That check is a reasonable coarse heuristic but it cannot
distinguish "never learned anything" from "learned, peaked, then partially
drifted" -- which is what actually happened here, as the full-run mean
comparison against run 1 makes clear. Report both the automated verdict AND
the fuller checkpoint-by-checkpoint comparison; don't rely on one summary
statistic alone, which is exactly the kind of single-number-hides-the-story
trap this project is committed to avoiding elsewhere (e.g. Stage 3's
near-perfect-class investigation).

## What this means going into Stage 5

The agent's *headline* evaluation still has to happen on DARPA2000 (Stage
5), which this training never touched. Two things to carry forward
honestly: (1) recall is consistently strong (~0.98-0.99) but precision is
moderate (~0.75-0.85) and noisy -- expect Stage 5's campaign reconstruction
to show a similar pattern: DARPA campaigns likely mostly get found, but
with some real false-link noise to report, not hidden. (2) The training
curve's mid-run peak-then-partial-regression pattern means the specific
episode at which training was stopped matters somewhat; the saved policy
network is from the full 3000-episode run's final state, not a
cherry-picked best checkpoint -- this is the honest choice (using the best
validation checkpoint would be a legitimate technique in general ML
practice, but for this project's specific commitment to not tuning toward
favorable numbers, using the final-state policy as trained is more
defensible and is what's reported here).
