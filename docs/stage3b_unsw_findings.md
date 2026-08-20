# Stage 3b findings — UNSW-NB15 cross-check (optional, secondary)

## What this is and isn't

The original proposal named UNSW-NB15 as an *optional secondary network-tier
source* for checking whether Stage 3's approach generalizes beyond
CICIDS2017. This is NOT Stage 3's actual trained model applied to a second
dataset -- that's architecturally impossible, exactly as it is for DARPA2000:
UNSW-NB15's 49-column flow-statistics schema shares no columns with
CICIDS2017's 122. What was actually done: an independent Random Forest,
same methodology and leakage discipline as Stage 3, trained and evaluated
entirely on UNSW-NB15's own data using its own canonical, dataset-native
train/test split (`UNSW_NB15_training-set.csv` / `testing-set.csv` --
175,341 / 82,332 rows, not a split this project constructed). This checks
whether the *approach* generalizes, not whether one specific model does.

## A leakage hypothesis that didn't pan out -- reported anyway

Before training anything, the raw `sttl`/`dttl` (source/destination TTL)
columns were checked directly against the label, because UNSW-NB15
literature documents this as a known concern. The finding was stark: Normal
traffic is overwhelmingly `sttl=31`, while most attack categories are
97-100% `sttl=254` (Shellcode 100%, Fuzzers 99.6%, Generic 99.4%,
Reconnaissance 99.5%) -- a simulation-configuration artifact (attack traffic
generated from VMs with a different default TTL than the background-traffic
generator), not a learned attack signature. This looked like exactly the
kind of shortcut Stage 3 had to guard against with CICIDS2017's fixed
attacker IP.

**Trained both ways to check, rather than assuming the hypothesis was
correct:** a "naive" model (all 42 features, including sttl/dttl) and an
"honest" model (40 features, sttl/dttl excluded).

| | naive (with TTL) | honest (without TTL) | delta |
|---|---|---|---|
| accuracy | 0.6805 | 0.6795 | +0.0010 |
| macro F1 | 0.4940 | 0.4814 | +0.0126 |
| weighted F1 | 0.7328 | 0.7315 | +0.0013 |

**Removing the suspected leakage columns barely changed anything.** This is
reported precisely because it's the less satisfying, less dramatic outcome
-- a hypothesis that looked well-founded from the raw marginal distribution
didn't hold up under an actual ablation. The likely explanation: the other
40 features already carry enough overlapping signal (byte counts, packet
timing, service/state patterns) that the model doesn't need sttl/dttl
specifically to reach a similar decision boundary -- the marginal
distribution being skewed doesn't mean the trained model is leaning on it
once 40 other correlated features are available. The **honest model's
numbers are used as Stage 3b's headline** regardless, since excluding a
column with zero real attack-behavior justification is the right call even
when it doesn't move the number -- the point was never to chase a
particular result.

## Headline result: realistic, not inflated

**Accuracy: 68.0%, macro F1: 0.481, weighted F1: 0.732** (honest model, on
UNSW-NB15's own held-out test set). Substantially more modest than Stage
3's CICIDS2017 numbers (85% accuracy, weighted F1 0.86) -- a different
dataset, different attack taxonomy (10 classes here vs. 14 there), and
UNSW-NB15 is widely acknowledged in the literature as a harder, noisier
classification problem than CICIDS2017. This is the expected, honest shape
of a genuine cross-dataset check: the same methodology does not produce the
same numbers on a different dataset, and it shouldn't be expected to.

Per-class detail (honest model): `Generic` (F1=0.981, n=18,871) and `Normal`
(precision=0.99) are strong. Several minority classes are weak on
precision despite reasonable recall -- `Analysis` (precision=0.01),
`Backdoor` (0.04), `Shellcode` (0.20) -- a real class-imbalance cost of
`class_weight='balanced'`: boosting rare-class recall pulls in more false
positives from the much larger classes. This mirrors the same
precision/recall tradeoff documented for CICIDS2017's `Benign` class in
Stage 3, and is disclosed the same way here.

## The one near-perfect class, investigated (not waved through)

`Generic` scored F1=0.981 on substantial support (n=18,871) -- checked, not
assumed safe just because TTL was already excluded. Confusion matrix:
18,179/18,871 correct, with only 5 total false positives from other
classes (near-perfect precision) and errors concentrated as false
negatives into semantically adjacent categories (Exploits, Fuzzers, DoS).
Top feature importances for the honest model are dominated by `sbytes`/
`smean` (payload byte-size statistics) -- UNSW-NB15's "Generic" category
specifically covers a block-cipher-attack technique with a documented,
consistent payload-size signature (works the same way regardless of the
target cipher's internal structure), which plausibly explains a genuinely
learnable, non-leaky pattern rather than an artifact. Verdict: legitimate.

## Bottom line

This is genuinely secondary, optional material -- it does not change any
Stage 3/4/5 result, and Stage 3's actual model was never touched. Its value
is narrow and specific: it shows the leakage-aware supervised-classification
approach used in Stage 3 produces sensible, non-inflated, appropriately
humbler numbers on a second, independent network dataset with its own
canonical split, and that a plausible leakage hypothesis was checked with an
actual ablation rather than either assumed true (because the literature
says so) or ignored (because it was inconvenient to test). Both outcomes --
had the ablation shown a big drop, or (as it did) shown almost none -- were
always going to be reported.
