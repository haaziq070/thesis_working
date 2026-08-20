# Stage 5 findings — headline evaluation on DARPA2000

## Correctness note: these are the corrected numbers

A full codebase correctness review found a real bug in the DQN's state
construction: the "stage progression" feature (does this pair look like a
kill-chain escalation) was silently always `0.0` for every DARPA2000 event
pair, because it was derived via a CICIDS2017-only vocabulary lookup that
DARPA's labels never match (confirmed empirically before the fix: exactly
one distinct value across all 315 DARPA event pairs). This affected the
actual Stage 4 training and Stage 5 evaluation reported earlier. It has
been fixed (`src/correlation_env.py`, plus a correctly-derived
`predicted_kill_chain_stage` column added to both datasets' Stage 2 output),
Stage 4 was retrained from scratch, and Stage 5 was re-run. The numbers
below are the corrected results. The pre-fix numbers are kept in
`data/processed/stage5_buggy_run/` and `data/processed/stage4_run2_fn2.5x/`
(same reward configuration as the current `stage4/`, but trained before the
state-vector fix) for anyone who wants to compare, and the delta is discussed explicitly below
rather than silently replaced -- a wrong number, caught and disclosed, is
more defensible than a right number with no paper trail.

## What's implemented and evaluated

`scripts/stage5_evaluate_darpa.py` reconstructs campaigns on all 316
DARPA2000 events (both LLDOS scenarios pooled by their real timestamps --
they are over a month apart, March 7 vs April 16 2000, so pooling requires
no artificial time-shifting; any reasonable time-windowed method naturally
keeps them separate) using a shared online reconstruction procedure
(`src/campaign_reconstruction.py`): process events chronologically, compare
each new event against every still-"open" cluster (updated within the last
hour) using a decision function, join the most-recently-active cluster that
votes "link," otherwise start a new one. The SAME procedure is used for both
the trained Stage 4 DQN (greedy policy) and a naive rule-based baseline
(link iff shared IP AND within 30 minutes) -- isolating the comparison to
the decision function itself, not the surrounding algorithm.

DARPA2000 was read only by this script. Stage 3 and Stage 4 training data
excluded it by construction and by an explicit runtime check
(`assert_dataset_is_test_only`) -- this is genuinely, verifiably external
data.

## Disclosed limitation: DARPA's "identification" is idealized

Stage 3's classifier cannot run on DARPA2000 at all -- it was trained on
CICIDS2017's 122-column flow-statistics schema, and DARPA's parsed
tcpdump session records have no equivalent features (no packet-timing
statistics, no header-byte distributions, nothing comparable). This is a
structural, not fixable-within-scope, gap. For this evaluation, DARPA's own
ground-truth-derived `attack_type`/`kill_chain_stage` (already present from
Stage 2, and legitimately dataset-native, not invented) stand in for the
`predicted_*` fields in the DQN's state -- and, since the correctness fix,
`kill_chain_stage` is aliased *directly* into `predicted_kill_chain_stage`
rather than re-derived through a mismatched lookup table. This means DARPA's
correlation task runs under an idealized identification assumption that
CICIDS2017's training never had -- Stage 4 trained against Stage 3's real
predictions, warts included. Keep this asymmetry in mind reading every
number below: if anything, it should make DARPA look *easier* for the
identification-derived state features, not harder.

## The headline result, reported honestly

**By pairwise link precision/recall/F1** (the more directly interpretable
metric -- see caveat on ARI below):

| method | precision | recall | F1 |
|---|---|---|---|
| DQN (Stage 4 trained agent) | 0.317 | 0.208 | **0.251** |
| Rule-based baseline (shared IP + 30min) | 0.251 | 0.746 | **0.375** |

**The DQN still does not outperform the naive baseline on pairwise F1**, but
the fix moved every one of the DQN's numbers in the right direction:
precision 0.259->0.317, recall 0.160->0.208, F1 0.198->0.251. The gap to the
baseline narrowed (F1 deficit went from -0.177 to -0.124) but did not close.
This is reported as the honest headline finding, not reframed or hidden.

**By Adjusted Rand Index** (the metric named in the proposal), the picture
changed direction: **DQN=0.0806, baseline=0.0369 -- the DQN now leads.**
Before the fix this was reversed (DQN 0.0224 vs baseline 0.0369). Both
numbers are still close to zero and not hugely informative on their own; the
real story is that **ARI and pairwise F1 now disagree about which method is
better**, which is itself a finding worth stating plainly rather than
picking whichever metric tells the preferred story. The mechanism, checked
against the actual cluster composition
(`data/processed/stage5/darpa2000_reconstruction.csv`): the fix let the DQN
form fewer, more purposeful clusters (17, down from 22) with noticeably
better true-campaign concentration per cluster -- e.g. one 38-event cluster
is 89% LLS_DDOS_1.0 members -- which is exactly the kind of balanced,
correctly-separated partition ARI's chance-correction rewards. But the
DQN's *raw* pairwise recall is still far below the baseline's, because the
baseline's brute-force "link anything sharing an IP" strategy still finds
more true pairs in absolute terms than the DQN's more selective, still
somewhat fragmented approach. Both metrics are reporting something true;
they're just weighting "purposeful but incomplete" against "complete but
indiscriminate" differently. Report both, do not average them into a single
number that hides the disagreement.

## What's actually going on, investigated (not just accepted)

**The baseline over-links.** Its rule (any shared IP within 30 minutes)
transitively chains almost everything into one 262-event cluster (83% of
all events), because DARPA2000's fixed small IP space means many unrelated
events share an address with *something* nearby in time. This inflates its
recall (finds most true pairs, because it finds almost *all* pairs) at the
cost of precision -- 26,063 false-positive pairs. It "wins" on pairwise
recall/F1 largely by brute-force over-connection, not by discriminating
well -- and this is exactly what ARI's structure penalizes it for, which is
why the two metrics disagree.

**The DQN still under-links relative to the baseline, even post-fix.**
LLS_DDOS_2.0.2 (32 events, zero within-scenario distractors -- confirmed in
Stage 2, no risk of incorrect merging at all) should trivially cluster as
one group. The baseline still gets this exactly right (all 32 in one
cluster, completeness 1.00). The DQN now fragments it into 3 clusters
(improved from 4 pre-fix, but still not 1; completeness 0.75, missing the
"exploit" stage in its best-matching cluster). Some improvement, not a
resolution -- the DQN is still more conservative about linking than the
ground truth would reward, even in a low-ambiguity case.

The most defensible explanation, unchanged by the fix, is a genuine
**cross-dataset generalization gap**: the DQN was trained entirely on
CICIDS2017's campaigns, which have very different event volumes, densities,
and temporal structure (from a 12-event Heartbleed campaign to a
4,153-event merged Friday botnet/scan/DDoS block) than DARPA2000's cleaner,
more sparsely-timed LLDOS scenarios. A policy that learned useful
state-feature thresholds on CICIDS2017's specific distribution does not
necessarily carry over cleanly to a structurally different external
dataset -- which is exactly the kind of honest limitation a
same-distribution validation split (Stage 4's CICIDS2017 val campaigns)
cannot reveal, and exactly why the proposal's insistence on an external,
never-touched test set matters: Stage 4's validation F1 (~0.87-0.89 mean,
essentially unchanged by the fix) looked reasonably good, and Stage 5 shows
that number does not fully transfer, even with the feature bug corrected.

## Kill-chain completeness: still a mixed picture

| campaign | DQN completeness | Baseline completeness |
|---|---|---|
| LLS_DDOS_1.0 | **1.00** (all 4 stages captured in best cluster, despite fragmentation elsewhere) | 0.75 (missed "exploit") |
| LLS_DDOS_2.0.2 | 0.75 (missed "exploit") | **1.00** |

Unchanged by the fix. Not a clean sweep either way. On LLS_DDOS_1.0
specifically -- the scenario with real distractors to reject, the more
realistic test -- the DQN's best-matching cluster still captures the full
multi-stage narrative better than the baseline's, even though the DQN's
overall pairwise recall is lower there too.

## Bottom line for the thesis

This is the headline result, and it is still not a clean success story for
the RL approach on external data -- report it as such, including the fact
that a real bug was found and fixed mid-project and the corrected picture
is more nuanced (ARI now favors the DQN, pairwise F1 still favors the
baseline) rather than uniformly negative. The value of this project is not
"the DQN won"; it is: (1) a working, leak-free, honestly-evaluated two-stage
pipeline was built and run end to end; (2) every stage's results were
checked for inflation, bugs, and suspicious patterns, including a
self-directed full-codebase correctness review that caught a real,
consequential bug after the headline numbers had already been reported once
-- and the response was to disclose it, fix it, retrain, and re-report, not
to quietly patch the number; (3) the central, most important finding -- a
real cross-dataset generalization gap between same-distribution validation
and genuinely external test performance, which persists even after the fix
-- is exactly the failure mode a thesis explicitly designed to avoid a
prior work's inflated 100%-on-self-generated-data result should be able to
surface and explain, not paper over. This belongs prominently in Stage 6.
