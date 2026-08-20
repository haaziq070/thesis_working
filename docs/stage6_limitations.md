# Stage 6 — Threats to validity and limitations

This section is what makes the thesis credible. Every item below is a real
finding surfaced during the project, not a boilerplate disclaimer — each one
traces back to a specific investigation in `docs/stage{2,3,4,5}_findings.md`.
The goal here is not to apologize for imperfect numbers; it is to state
precisely what was implemented, what was evaluated, what was only
conceptually proposed, and why each limitation exists, so a committee
question about any of them has a specific, checkable answer rather than a
hand-wave.

## 1. Scope commitments — restated, and never violated

Two commitments were binding throughout: (1) this is a post-hoc/forensic
correlation tool, not a real-time detector; (2) reinforcement learning is
used only for the link/don't-link correlation decision, never for attack
identification. Both held throughout — Stage 3 (identification) is an
ordinary Random Forest with zero RL involvement; Stage 4 (correlation) is
the only stage where RL appears, and it never classifies raw traffic. No
step in the pipeline runs inline; every stage processes bounded, offline
batches of already-collected events. Section 7 below expands on exactly
what "post-hoc" cost in terms of design assumptions.

## 2. The central finding: a real cross-dataset generalization gap

This is the single most important result in the whole project and belongs
first, not buried. Stage 4's DQN reached a mean validation F1 of ~0.87-0.89
on CICIDS2017's held-out campaigns (same-distribution validation, disclosed
reward-asymmetry adjustment included). Stage 5's evaluation on DARPA2000 —
genuinely external data, never read by any training or tuning code, verified
in code at every stage via `assert_dataset_is_test_only` — showed the DQN's
pairwise-link F1 (0.251) below a naive rule-based baseline's (0.375), though
the DQN leads on Adjusted Rand Index (0.081 vs 0.037) -- the two metrics
disagree, and both are reported rather than one selected to tell a cleaner
story (see `docs/stage5_findings.md` for the full mechanism: the DQN forms
fewer, more concentrated clusters, which ARI rewards, while the baseline's
brute-force over-linking still catches more raw true pairs). Investigation
traced the DQN's shortfall partly to under-linking events even in a
near-zero-ambiguity case (`LLS_DDOS_2.0.2`, 32 events, no distractors) that
the naive baseline got exactly right.

This is precisely the failure mode a same-distribution validation split
cannot reveal, and precisely why the proposal's insistence on an external
test set mattered: a strong validation number here would have been
misleading on its own. The likely mechanism is that a policy learned on
CICIDS2017's specific campaign structure (event volumes from 12 to 4,153
per campaign, a fixed two-host attacker/victim topology) does not transfer
cleanly to DARPA2000's structurally different event density and timing.
A thesis that only reported the ~0.87-0.89 validation number would have
reproduced exactly the kind of inflated, non-generalizing result this
project was explicitly designed to avoid.

## 3. Data limitations

**CICIDS2017's campaign labels are derived, not native ground truth.**
Unlike DARPA2000 (where phase structure is the dataset's own documented
design), CICIDS2017 ships no multi-stage campaign labels at all — every
attack type is a separate single-technique capture. Stage 2 derived
campaigns via time-gap clustering (attack blocks within 60 minutes of each
other merged into one campaign), cross-checked against the officially
published CIC-IDS2017 attack schedule (Sharafaldin et al. 2018) for face
validity, but this remains a constructed grouping, not something the
dataset's creators labeled as multi-stage. Every result trained or tuned
against this split should be read with that caveat.

**No genuine web/WAF-tier alert data exists anywhere in this project.**
The proposal's three-tier design (network/firewall, web/WAF, host/endpoint)
was never fully realized: CICIDS2017's `web_sql_injection`/`web_xss` files
are still network-flow-level captures (5-tuple + flow statistics), not
HTTP-layer alerts with method/URL/payload visibility — verified directly
against the schema in Stage 2, not assumed. The planned DVWA +
OWASP-CRS/ModSecurity supplementary source, which would have produced real
WAF-style alerts, was deferred early in the project (a scoping decision
made explicitly, not an oversight) and never built. Concretely: this
project's "web tier" is, in every result reported, actually still
network-tier data describing web-layer attacks — a real gap between the
proposal's three-tier design and what was actually evaluated.

**UNSW-NB15 was obtained and cross-checked, but only after a manual
download step, and the check has real limits.** It was scoped as an
optional secondary network-tier source; it sat gated behind a
browser-only SharePoint folder with no scriptable access until it was
downloaded manually and transferred onto this machine. Stage 3b (see
`docs/stage3b_unsw_findings.md`) trained an independent Random Forest,
same methodology as Stage 3, on UNSW-NB15's own canonical train/test
split — not a re-application of Stage 3's actual CICIDS2017-trained model,
which is architecturally impossible here for the same reason it is for
DARPA2000 (no shared feature schema). Headline numbers were substantially
more modest than CICIDS2017's (68% accuracy vs. 85%, macro F1 0.48 vs.
0.65) — the honest, expected shape of the same approach meeting a
different, harder dataset, not a failure. A documented UNSW-NB15 leakage
concern (`sttl`/`dttl`, a simulation-configuration artifact reported in the
literature and independently confirmed here against the raw label
distribution) was tested with an actual before/after ablation rather than
assumed — and the ablation showed almost no effect (accuracy delta
+0.001), a result reported precisely because it complicates the tidier
"we found and removed a leak" story. What this check does NOT establish:
Stage 3's *actual* CICIDS2017 model was never touched or re-validated by
this exercise, so it remains true that Stage 3's specific classifier has
only ever been evaluated on CICIDS2017-derived data — Stage 3b shows the
*approach* transfers reasonably, not that the deployed model does.

**Benign/distractor sampling used a fixed heuristic, not an exhaustive
population.** Stage 2 sampled benign CICIDS2017 rows at 5x each day's
attack-row count (capped at 50,000/day) for computational tractability on
a 2-core/4GB machine. This is a defensible, disclosed design choice, but a
different sampling rate or method could shift the class-imbalance handling
and downstream results somewhat — it was not exhaustively tested for
sensitivity.

## 4. DARPA2000-specific limitations

**Age.** DARPA2000 (LLDOS 1.0 / 2.0.2) is from March/April 2000 — it
predates modern web attacks entirely (no HTTP-layer traffic exists in it at
all) and reflects network/host-layer attack tooling and IDS-evasion
techniques of that era (the sadmind RPC exploit, the mstream DDoS trojan).
**Standard justification for still using it, stated explicitly rather than
assumed:** it remains the canonical labeled multi-stage correlation
benchmark in the intrusion-detection-correlation literature specifically
*because* it has genuine, dataset-native multi-phase campaign ground truth
(step 1 leads to step 2 leads to step 3, as designed by MIT Lincoln
Laboratory) — a property this project's own Stage 2 work demonstrated that
newer flow-centric datasets like CICIDS2017 simply do not have natively
(Section 3 above). The tradeoff is real and was not avoided: correlation
research needs genuine multi-stage ground truth, and as of this writing no
modern replacement dataset offers it at DARPA2000's level of documented
fidelity.

**Single environment, extremely small N.** The entire external headline
evaluation rests on two real campaigns (LLDOS 1.0 and LLDOS 2.0.2) from one
simulated network topology, one red team's tooling, one specific evaluation
exercise. Statistical power at N=2 campaigns is minimal — Stage 5's finding
that the DQN underperforms the baseline is a real, investigated result, not
a fluke, but "the DQN generalizes worse than a naive baseline to DARPA2000"
is not the same claim as "the DQN generalizes worse than a naive baseline
to external network intrusion data in general." The two campaigns also
differ sharply in character (LLDOS 1.0 is noisy with 134 real distractor
events to reject; LLDOS 2.0.2 is quiet with zero) — useful for exposing two
different failure modes, but not a broad sample of attacker behavior.

**No web/WAF tier represented at all.** DARPA2000 predates HTTP-based
attacks; its events are network (tcpdump) and host (BSM audit) only. The
three-tier correlation design is therefore only partially exercised even in
concept on the headline evaluation set — genuinely three-tier correlation
(network + web/WAF + host) was only ever exercised on the CICIDS2017/DVWA
side of the project, and DVWA was never built (Section 3).

**DARPA's identification signal is idealized, not predicted.** Stage 3's
classifier cannot run on DARPA2000 at all — its 122-column CICIDS2017 flow-
statistics feature schema has no equivalent in DARPA's parsed
tcpdump-session records. Stage 5 used DARPA's own ground-truth-derived
`attack_type`/`kill_chain_stage` as the DQN's "identification" input,
explicitly disclosed as an idealized condition CICIDS2017 training never
had. If anything this should have made DARPA's correlation task *easier*
for the DQN than CICIDS2017's training was — which makes Section 2's
finding (the DQN still underperformed there) more concerning, not less.

## 5. Model and training-methodology limitations

**One disclosed reward-asymmetry adjustment.** The DQN's false-negative
penalty was changed from 5x to 2.5x the link reward after diagnosing that
5x produced an "almost-always-link" degenerate policy (Stage 4). This was a
single, disclosed, diagnosis-driven change with both runs' full results
kept and reported side by side — not repeated tuning toward a target score.
It should still be named as a limitation on the purity of "first-principles
reward design, never touched again": the final reward weights were informed
by one round of observed training behavior, which is a common and
defensible practice in RL but is not the same as a reward design validated
purely analytically before any training occurred.

**Limited training campaign diversity.** The DQN trained on only 4
CICIDS2017 campaigns (Stage 2's leak-free split), one of which (the
Heartbleed-derived campaign) has just 12 events. This is a narrow
experience base for learning a generalizable Q-function, and is a
plausible contributor to Section 2's generalization gap.

**sklearn `MLPRegressor` instead of a standard deep-RL framework.**
Documented at the time (Stage 4): no CPU PyTorch package was available via
apt, and PyPI downloads of that size repeatedly stalled under this
environment's network conditions. The implementation has all of DQN's real
components (experience replay, target network, epsilon-greedy decay, a
genuine discount factor over real multi-step episodes) built on a small
2-hidden-layer network appropriate to the 7-dimensional state space used.
It is a defensible engineering substitution for this project's constraints,
but a reader should know the specific optimizer/framework differs from
what most published DQN work uses, and this could plausibly affect training
dynamics in ways not explored here.

**Stage 3's real classification errors propagate into Stage 4/5 by
design — and that design choice has a cost.** The correlator's CICIDS2017
training used Stage 3's actual predictions (Benign precision only 0.49,
Web_XSS F1 only 0.30), not ground truth, which is the more honest,
realistic condition — but it also means any weakness in Stage 4's
CICIDS2017 results could originate in Stage 3's imperfections rather than
Stage 4's own design, and the two are not cleanly separated in this
project's evaluation. No ablation isolating "DQN performance given perfect
identification, on CICIDS2017" was run to disentangle the two effects.

## 6. Evaluation-methodology limitations

**Adjusted Rand Index and pairwise F1 disagree on this data, and both are
reported rather than one being picked.** ARI ranks the DQN ahead of the
rule-based baseline (0.081 vs 0.037); pairwise F1 ranks the baseline ahead
(0.375 vs 0.251). This is a known behavior of ARI's chance-correction when
comparing partitions with very different cluster-count/size profiles --
exactly this situation, since the ground truth is dominated by 134 true
singleton distractor "clusters" while the baseline collapses 83% of DARPA's
events into one giant cluster. Pairwise precision/recall/F1 was added as a
cross-check specifically because of this and is reported as the primary
metric, with ARI (the metric named in the
proposal) kept and reported alongside it rather than dropped. This is
disclosed as a real metric-choice limitation, not a hypothetical one: the
two metrics don't just differ quantitatively here, they disagree on which
method is better, and a thesis defense should be prepared to explain both
readings rather than cite whichever one is more favorable.

**A self-directed full-codebase correctness review found and fixed three
real bugs mid-project, after headline numbers had already been reported
once.** The significant one: one of the DQN's 7 state features
(stage_progression) was silently constant for every DARPA2000 pair due to a
label-vocabulary mismatch between the two datasets. Two smaller ones: an
operator-precedence bug in Stage 4's stability-report slicing (`-n // 3`
parses as `(-n) // 3`, not `-(n // 3)` -- didn't affect the numbers actually
reported, since the run happened to have an evenly-divisible episode count,
but would silently misreport for other configurations) and a distractor-pool
gap in the training environment (sampled distractor events positioned
before the seeded anchor were never reachable as comparison candidates,
quietly reducing negative-example exposure below what the sampling budget
implied). All three were caught, disclosed, and fixed; the two affected
stages (Stage 4 training, Stage 5 evaluation) were re-run rather than the
bugs being quietly patched with the old numbers left standing. The corrected
numbers are what's reported throughout this document and
`docs/stage5_findings.md`; the pre-fix numbers are kept on disk
(`data/processed/stage4_run2_fn2.5x/`, `data/processed/stage5_buggy_run/`)
rather than deleted.

A second, follow-up review pass was run specifically to check whether the
fix itself introduced any new problems and whether anything else had been
missed. It found no further correctness bugs, and went beyond re-reading the
code: it empirically verified that no leftover references to the broken
lookup pattern remained anywhere in the codebase, confirmed event IDs are
genuinely unique across both processed datasets (0 duplicates in 316 DARPA
and 58,057 CICIDS2017 rows -- several downstream functions key dictionaries
by event ID, so a collision there would have silently corrupted results),
and directly tested the DQN's target-network sync mechanism (unusual sklearn
usage -- manually copying `MLPRegressor.coefs_`/`intercepts_` rather than
using a framework's built-in target-network support) by running it and
confirming policy and target networks start identical, measurably diverge
between sync points, and exactly re-converge at each scheduled sync -- i.e.
confirming the target network is actually doing its job, not silently
always equal to (or always different from) the policy network.

This whole episode is listed here as a limitation, not a strength,
deliberately: it means the results in this thesis were not correct on the
first attempt, and a careful reader should ask what else might not have
been caught. The honest answer is that two correctness passes, however
thorough -- including one that went as far as empirically testing the
trickiest custom mechanism in the codebase rather than only reading it --
are still not a guarantee against further undiscovered bugs, in this
codebase or any other. What can be said concretely is what was specifically
checked and how (listed above), so a defense question about verification
depth has a precise answer rather than a general assurance.

**Only one external dataset.** DARPA2000 is the only genuinely external
test set used. A more thorough validation of the generalization-gap finding
in Section 2 would test against at least one more independent multi-stage-
labeled external source — none was available within this project's scope
(the search for one, and the reasoning for why DARPA2000 was the best
available choice, is documented in Stage 1).

## 7. Post-hoc, non-real-time scope — what this cost in design terms

Every design choice in this pipeline assumes offline, batch access to a
bounded window of already-collected events: Stage 4's episodes process a
fixed campaign's events (capped at 60 per episode) in one pass; Stage 5's
reconstruction algorithm compares each new event against clusters "open"
within the last hour, which is a reasonable batch/forensic-analysis window
but not a latency or throughput guarantee suitable for inline traffic
inspection. No part of this system was measured for or designed around
real-time performance constraints (events-per-second throughput, decision
latency), and none of the reported results should be read as evidence this
approach would perform acceptably as a real-time/inline system — that was
explicitly out of scope from the original proposal and remained so
throughout.

## 8. What's implemented, experimentally evaluated, and conceptual — project-wide summary

| Component | Status |
|---|---|
| Three-tier unified event schema | Implemented (network tier only actually populated; web/host tiers conceptual, see Section 3) |
| DARPA2000 campaign parsing + leak-free holdout | Implemented and evaluated |
| CICIDS2017 campaign derivation (time-gap clustering) | Implemented and evaluated; derived, not native, ground truth |
| Stage 3 supervised identifier (extended multi-class RF) | Implemented and evaluated on CICIDS2017 only; Stage 3's actual model never re-validated on a second dataset |
| DVWA + OWASP-CRS/ModSecurity web-tier alerts | Never implemented (deferred at Stage 1, never revisited) |
| Stage 4 DQN correlator | Implemented and evaluated on CICIDS2017 (same-distribution validation) |
| Stage 5 DARPA2000 external evaluation | Implemented and evaluated; the project's real headline result |
| Real-time / inline operation | Never implemented, never claimed — explicitly out of scope throughout |
| Stage 3b UNSW-NB15 methodology cross-check | Implemented and evaluated (independent model, dataset's own split, not Stage 3's actual classifier) — see Section 3 and `docs/stage3b_unsw_findings.md` |

## 9. What a follow-up project could do differently

Not offered as excuses for the limitations above, but as concrete, specific
next steps a reader might reasonably ask about: (1) train the DQN across a
wider, more diverse set of campaigns — potentially synthesizing additional
campaign structures from CICIDS2017's data at different densities to
deliberately expose the network to more of the event-volume range Section 2
identifies as a likely generalization driver; (2) build the deferred DVWA +
ModSecurity pipeline to get genuine web/WAF-tier alerts and re-run the full
three-tier design as originally scoped; (3) re-validate Stage 3's actual
CICIDS2017-trained model's predictions against UNSW-NB15 in some
feature-bridging way (e.g. mapping both datasets down to a small shared
feature subset), since Stage 3b showed the general approach transfers but
never touched the deployed model itself; (4) obtain a second external multi-stage-labeled correlation
benchmark beyond DARPA2000 to check whether Section 2's generalization gap
is DARPA-specific or a broader pattern; (5) run the ablation separating
Stage 3's identification errors from Stage 4's correlation-learning
weaknesses on CICIDS2017, to know how much of any given result belongs to
which stage.
