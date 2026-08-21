---
title: "Reinforcement Learning-Based Adaptive Alert Correlation for Detecting Multi-Stage Cyber Attacks"
author: "Haaziq Rasool"
date: "August 2026"
---

# Abstract

Security operations centers routinely face the problem of *alert correlation*:
deciding which of the many individual alerts fired by network, web, and host
sensors belong to the same underlying multi-stage attack campaign. This
thesis investigates whether a reinforcement-learning (RL) agent can be used
for the *correlation* decision — whether two alerts belong to the same
campaign — while keeping *attack identification* (what technique an alert
represents) as a separate, conventional supervised-learning problem. The
system is explicitly scoped as a **post-hoc, forensic** correlation tool
that operates over a bounded window of already-collected events, not a
real-time or inline detector, and RL is used **only** for the link/don't-link
correlation decision, never for attack typing.

The pipeline is built and evaluated in six stages: data acquisition and
verification; unified-schema preprocessing with campaign-level, leakage-free
train/test splitting; a supervised Random Forest attack-identification
classifier; a Deep Q-Network (DQN) correlation agent trained on same-
distribution data; a headline evaluation on a genuinely external dataset the
model never saw during training or tuning; and a systematic accounting of
threats to validity. An optional secondary cross-check on a second,
independent network-traffic dataset (UNSW-NB15) is also reported.

The central, and most important, empirical finding is a real cross-dataset
generalization gap: the DQN correlator reaches a strong same-distribution
validation F1 of 0.873 on held-out CICIDS2017 campaigns, but its pairwise-
link F1 on the external DARPA2000 evaluation set (0.251) falls below a naive
rule-based time-window baseline (0.375), even though the DQN leads the
baseline on Adjusted Rand Index (0.081 vs. 0.037). This disagreement between
metrics, and the DQN's underperformance relative to a trivial baseline on
genuinely unseen data, is reported and analyzed in full rather than
obscured, in direct service of this thesis's governing principle: that
results must be **defensible**, not merely impressive. A self-directed,
full-codebase correctness review conducted mid-project found and fixed three
real implementation bugs — including one that had silently degraded the
DQN's state representation for every DARPA2000 event pair — and the affected
stages were retrained and re-evaluated rather than having the bugs quietly
patched around already-reported numbers. This process, and its outcome, are
treated as part of the thesis's contribution: a worked example of how to
build and honestly evaluate an RL-based security system without reproducing
the data-leakage failure modes that are common in this literature.

# Chapter 1 — Introduction

## 1.1 Motivation

A single multi-stage cyber attack — reconnaissance, exploitation, lateral
movement, and eventual impact — typically produces many individual alerts
scattered across a network intrusion detection system, a web application
firewall, and host-based endpoint sensors. Security analysts are left to
manually determine which of these scattered alerts belong to the same
campaign, a process that does not scale with modern alert volumes: SOC
staffing is finite, alert volumes are not, and analyst burnout from
sustained high-volume triage is itself a documented operational problem,
not only a data-quality one (Sundaramurthy et al., 2015). *Alert
correlation* — grouping raw alerts into coherent attack campaigns — is
therefore a necessary layer between raw sensor output and actionable
incident response, sitting logically downstream of detection and upstream
of triage.

Rule-based correlation (fixed time windows, shared-host heuristics) is
brittle: it either over-links (grouping unrelated events that merely occur
close together) or under-links (missing campaigns whose stages are spread
over a longer, attacker-controlled timescale). Crucially, a fixed rule's
threshold — a 30-minute window, a shared-IP requirement — is a single
number or condition chosen once, in advance, and applied uniformly
regardless of what the surrounding evidence actually looks like for a
given pair of alerts. A *learned* correlation decision is attractive
precisely because it can, in principle, weigh several imperfect signals
together and adapt its sensitivity to context, rather than committing to
one hand-picked threshold for every case. Framing the decision
specifically as a *sequential* one — where the choice to link one alert
changes what the "right" decision looks like for the next alert under
consideration, because the campaign's working hypothesis has just grown —
is what motivates reinforcement learning in particular, as opposed to a
single-shot pairwise classifier: correlation, done incrementally over a
stream of candidate events, has genuine multi-step structure, and
Section 2.4–2.5 and Section 3.5 develop this connection formally. This
thesis investigates whether that theoretical advantage actually
materializes in practice, and, just as importantly, whether it survives
contact with data the learned model was never trained on — the two
questions Section 1.2 makes precise.

## 1.2 Problem Statement

Given a stream of already-collected, already-identified security alerts,
can a reinforcement-learning agent learn a better link/don't-link policy
for grouping alerts into attack campaigns than a simple rule-based
baseline — and does that advantage, if any, hold up when the agent is
evaluated on a dataset it has never seen, generated by an environment
and attacker it was never trained against? This thesis operationalizes
that question as three more specific, individually answerable ones,
mirrored directly in how Chapters 5 and 6 are organized:

1. **Same-distribution performance.** Can a DQN correlator, trained on one
   dataset's campaigns, learn a policy that outperforms a naive rule-based
   baseline when evaluated on held-out campaigns from that *same*
   dataset's distribution? (Section 5.3.)
2. **External generalization.** Does that same trained policy retain any
   advantage when evaluated, unmodified, on a second, structurally
   different dataset it never saw during training or tuning — the
   condition that actually matters for real deployment, where an
   attacker's traffic will never look exactly like the training
   distribution? (Section 5.4, Section 6.1.)
3. **Defensibility of the result, whatever it turns out to be.** Can this
   entire investigation be conducted, and its results reported, in a way
   that would survive a skeptical, adversarial re-examination of the
   codebase and the numbers — rather than one that merely produces a
   favorable-looking headline figure? (Section 1.4, Section 6.2, Section
   6.5.)

The answer to the first question, on its own, would not be a satisfying
thesis result: a strong same-distribution number is cheap to produce and,
as Section 1.4 and Section 2.2 both discuss, is exactly the kind of result
this literature has learned to distrust on its own. This thesis's central
contribution lives specifically in the combination of the second and
third questions — whether the advantage generalizes, and whether the
process used to find that out is itself trustworthy.

## 1.3 Scope Commitments

Two constraints were fixed before any code was written and were never
violated over the course of the project:

1. **This is a post-hoc, forensic correlation system, not a real-time
   detector.** Every stage of the pipeline operates on a bounded, offline
   batch of already-collected events. No component was designed, tuned, or
   measured for real-time throughput or decision latency, and no result in
   this thesis should be read as evidence of real-time viability.

2. **Reinforcement learning is used only for the correlation decision.**
   Attack *identification* — determining what technique a given alert most
   likely represents — is solved with an entirely separate, conventional
   supervised classifier (Chapter 4, Stage 3). The RL agent (Stage 4) never
   sees raw traffic and never assigns an attack label; its only decision is
   whether two already-identified events belong to the same campaign.

These commitments were restated and checked explicitly against the final
implementation in Chapter 7 (Threats to Validity).

## 1.4 Governing Principle: Defensible, Not Perfect

Machine-learning-based intrusion detection research has a well-documented
susceptibility to data leakage — features or splits that let a model
"succeed" by exploiting an artifact of how the dataset was generated (a
fixed experimental IP address, a simulation-configuration side effect)
rather than a genuine attack signature (see, e.g., Arp et al. 2022 on
common pitfalls in ML-based security evaluation). Suspiciously perfect
results in this literature are, more often than not, a symptom of exactly
this problem rather than a genuine breakthrough.

This thesis adopts a single governing rule, which outranks every other
design or presentation preference: **a discipline of treating near-perfect
results as bugs to be investigated, not outcomes to be reported.** Every
stage of this project that produced a suspiciously strong number was
stopped and investigated before being accepted — concrete examples include
the removal of a fixed-attacker-IP shortcut feature from the Stage 3
classifier, individual confusion-matrix and feature-importance
investigations of two near-perfect CICIDS2017 classes, and an explicit
before/after ablation of a suspected TTL-based leakage column in the
UNSW-NB15 cross-check (Chapter 5). Where this discipline surfaced an
inconvenient result — most importantly, the DQN underperforming a naive
baseline on external data — that result is reported as the thesis's
headline finding rather than downplayed.

## 1.5 Contributions

1. A leakage-disciplined, unified event schema and campaign-level,
   leakage-free train/test splitting methodology shared across two
   structurally different datasets (DARPA2000 and CICIDS2017), with the
   no-leakage guarantee enforced in code rather than only asserted in
   prose.
2. A two-model architecture that keeps attack identification and
   attack-campaign correlation as separate learning problems, with the
   correlation agent trained against the identifier's *actual* imperfect
   predictions rather than idealized ground truth.
3. A DQN-based correlation agent with a genuine multi-step episode
   structure, evaluated first under same-distribution validation and then,
   critically, under a genuinely external, held-out evaluation on
   DARPA2000 — never read by any training or tuning code.
4. An honest, fully reported cross-dataset generalization gap, including
   two metrics (Adjusted Rand Index and pairwise F1) that disagree on which
   method performs better, both reported rather than one selected for a
   cleaner narrative.
5. A documented, self-directed full-codebase correctness review that found
   and fixed three real implementation bugs after initial results had
   already been produced, with the affected stages retrained and
   re-evaluated and the pre-fix results kept (not deleted) for comparison.
6. An optional secondary cross-check of the identification methodology on
   a second, independent dataset (UNSW-NB15), including an empirically
   tested (not assumed) leakage hypothesis.

## 1.6 Thesis Structure

Chapter 2 reviews background on alert correlation, kill-chain models, and
reinforcement learning, and justifies the choice of DQN over alternative
algorithms for this specific problem. Chapter 3 describes the overall
system design — the architectural decisions made in direct response to
Chapter 2's review, ahead of any implementation. Chapter 4 details the
concrete methodology and implementation of each of the project's stages,
including the exact hyperparameters and procedures needed to reproduce the
results. Chapter 5 reports those results stage by stage, including the
per-class and per-run detail behind each headline number. Chapter 6
discusses the two findings this thesis treats as central — the
cross-dataset generalization gap (Section 1.2's second question) and the
correctness-review episode (Section 1.2's third question) — and what they
mean together for the thesis's overall contribution. Chapter 7 is a
systematic threats-to-validity accounting, stating plainly what was and
was not established. Chapter 8 concludes and outlines specific,
evidence-motivated future work.

# Chapter 2 — Background and Related Work

*Note to the author: this chapter's narrower, less universally-known
citations (Section 2.2, 2.6 in particular) should be independently verified
against the original sources — exact venues, years, and page numbers were
not re-checked line-by-line while assembling this document. Where a claim
rests on a specific paper's findings rather than general, well-established
knowledge, the citation is marked and should be confirmed before
submission.*

## 2.1 Alert Fatigue and the Case for Automated Correlation

The practical motivation for alert correlation research is a volume
problem, not a novelty problem. A single network of moderate size can
generate thousands of low-level intrusion detection and firewall alerts
per day, the large majority of which are either benign noise, duplicates
of the same underlying event, or fragments of a single incident reported
once per sensor per packet. Security operations center (SOC) analysts
facing this volume are well documented to experience "alert fatigue" —
a state in which real, actionable alerts are missed or delayed because
they are indistinguishable, at the point of triage, from the surrounding
noise, and sustained exposure to this volume is itself linked to analyst
burnout and attrition (Sundaramurthy et al., 2015), which compounds the
original volume problem by shrinking the team available to handle it.
Correlating raw alerts into campaign-level groupings before they
reach an analyst is one of the standard mitigations proposed for this
problem: it reduces the *number* of things a human must reason about,
ideally without discarding the *information* contained in the raw alerts.
This project's Stage 2 burst-aggregation step (collapsing tens of
thousands of near-duplicate flood packets into one-second aggregate
events, Section 4.2) is a small, concrete instance of this same problem
at the sensor level, before campaign-level correlation is even attempted.

## 2.2 Approaches to Alert Correlation

Prior alert-correlation approaches can be organized into four broad,
overlapping families.

**Rule-based and similarity-based correlation.** The simplest and most
widely deployed approach in production SOC tooling: alerts are grouped by
fixed heuristics such as a shared source or destination IP within a fixed
time window, or a similarity score over shared alert attributes. This
approach is interpretable and cheap to compute — this project's Stage 5
baseline is exactly this family — but it is brittle against attacker-
controlled timing (an attacker who spaces out a campaign's stages beyond
the fixed window defeats it entirely) and against topology variation
(pivoting through an intermediate host breaks a pure shared-IP rule).

**Statistical and probabilistic correlation.** Early academic work (e.g.,
Valdes and Skinner's probabilistic alert-correlation approach, and related
statistical similarity-scoring methods from the early 2000s intrusion-
detection literature) models correlation as a similarity or likelihood
score over alert attribute distributions rather than a hard rule,
producing soft correlation strengths instead of a binary link/don't-link
decision. This generalizes rule-based correlation but still typically
relies on hand-specified similarity functions and attribute weights.

**Causal and graph-based correlation.** A second line of work (e.g., Ning,
Cui, and Reeves' work on constructing attack scenarios by matching an
alert's declared "prerequisites" against a preceding alert's
"consequences") builds correlation on top of an explicit causal or
prerequisite model of how one attack stage enables the next — closer in
spirit to reconstructing a kill chain than to a similarity metric.
Reconstructing the DARPA2000 kill-chain progression in this project's
Stage 5 evaluation (Section 5.4) is conceptually adjacent to this family,
though this project's DQN and rule-based baseline both operate on
similarity/temporal features rather than an explicit prerequisite graph.

**Machine-learning-based correlation.** More recent work applies
supervised or unsupervised learning directly to the correlation problem —
clustering alerts (e.g., root-cause-analysis-oriented alarm clustering
approaches such as Julisch's work on clustering intrusion-detection
alarms) or, more recently, applying deep learning to alert-sequence data.
This family is the most flexible but also the most exposed to the data-
leakage failure mode discussed in Section 1.4: an ML-based correlator
evaluated only on same-distribution held-out data can appear to solve
correlation while actually having learned a dataset-specific shortcut
(e.g., a fixed attacker IP, a simulation artifact) that a rule-based
system never had the flexibility to exploit in the first place. This risk
is precisely why this thesis insists on an external, structurally
different evaluation dataset (Section 4.6) rather than treating strong
same-distribution validation numbers as sufficient evidence on their own.

This project's DQN-based correlator sits within the machine-learning
family but is deliberately restricted to a small, interpretable, seven-
dimensional feature vector (Section 3.5) rather than raw alert content, in
an attempt to reduce — though, as Chapter 6 shows, not eliminate — this
family's characteristic risk of learning dataset-specific shortcuts rather
than genuine correlation signal.

## 2.3 Kill-Chain and Attack-Stage Models

This project's notion of a "multi-stage" campaign is grounded in the
family of kill-chain and attack-lifecycle models used throughout the
intrusion-detection and threat-intelligence literature — most influentially
the Lockheed Martin Cyber Kill Chain (Hutchins, Cloppert, and Amin, 2011),
which decomposes an intrusion into a fixed sequence of stages
(reconnaissance, weaponization, delivery, exploitation, installation,
command-and-control, and actions on objectives), and the more granular,
technique-level MITRE ATT&CK framework, which catalogs specific adversary
techniques organized under broadly analogous tactic categories. This
project's `kill_chain_stage` schema field (Section 3.2) is a coarsened
version of this family of models, chosen to be granular enough to give the
correlation agent a genuine stage-progression signal (Section 3.5) while
remaining mappable onto both DARPA2000's own documented phase structure
and CICIDS2017's flat per-technique labels — the two source datasets do
not share a stage taxonomy natively, and reconciling them onto one shared
scale was itself one of Stage 2's concrete tasks (Section 4.2).

## 2.4 Reinforcement Learning Foundations

Reinforcement learning formalizes sequential decision-making as a Markov
Decision Process (MDP; Sutton and Barto, 2018): an agent observes a
state, takes an action, receives a reward, and transitions to a next
state, with the goal of learning a policy that maximizes expected
cumulative (discounted) reward. Q-learning (Watkins and Dayan, 1992) is
the foundational value-based
algorithm in this family: it learns an action-value function, Q(s, a),
estimating the expected return of taking action *a* in state *s* and
acting optimally thereafter, without requiring a model of the environment's
transition dynamics. Deep Q-Networks (DQN; Mnih et al., 2015) extend
tabular Q-learning to large or continuous state spaces by approximating
Q(s, a) with a neural network, and introduced two mechanisms specifically
to stabilize this combination: an experience replay buffer, which
decorrelates the sequential training samples a naive online update would
otherwise produce, and a separate, periodically-synchronized target
network, which prevents the bootstrapped Q-value target from chasing a
rapidly-shifting estimate of itself. Subsequent work identified and
addressed specific weaknesses in vanilla DQN — most notably Double DQN
(Van Hasselt, Guez, and Silver, 2016), which addresses DQN's tendency to
systematically overestimate Q-values by decoupling action selection from
action evaluation, and prioritized experience replay (Schaul, Quan,
Antonoglou, and Silver, 2016), which samples high-error transitions more
often than uniform replay does. This project's
implementation (Section 4.4) uses standard uniform replay and single-
network action selection rather than these refinements; this is a
deliberate scope limitation, not an oversight, and is revisited in
Section 7.4.

Policy-gradient methods take a different approach, directly parameterizing
and optimizing a stochastic policy rather than a value function. Trust
Region Policy Optimization (TRPO; Schulman et al., 2015) constrains each
policy update to a trust region around the current policy to guarantee
stable improvement, at significant computational cost per update; Proximal
Policy Optimization (PPO; Schulman et al., 2017) approximates the same
stabilizing effect far more cheaply via a clipped surrogate objective, and
has become one of the most widely used general-purpose deep-RL algorithms,
particularly for continuous-control and large-action-space problems.

## 2.5 DQN vs. PPO for This Problem

Section 2.4's algorithm families differ principally in the action spaces
and update stability tradeoffs they were designed around. PPO's clipped-
surrogate mechanism earns its complexity in continuous or very large
discrete action spaces, where naive value-based methods struggle to
represent or search over the action set efficiently. The correlation
decision in this project is a **binary, discrete** action — link or
don't-link — which is precisely the small, discrete action space DQN was
designed for, and does not require any of the machinery PPO adds value
for. DQN was selected as the better-fitting, simpler algorithm for this
specific action space, not as a default or a convenience choice; a
policy-gradient method remains a reasonable alternative worth testing in
future work (Section 8.2), but nothing about this problem's action space
argues for one over DQN.

The DQN implementation in this project includes the algorithm's core
stabilizing components described in Section 2.4: an experience replay
buffer, a target network synchronized periodically with the policy
network, epsilon-greedy exploration with decay, and a discount factor
applied over genuine multi-step episodes (Section 2.2 above; formalized as
an MDP in Section 3.5). It does not include Double DQN's decoupled action
selection or prioritized replay (Section 2.4), a disclosed scope
limitation. It also substitutes `scikit-learn`'s `MLPRegressor` for a
standard deep-learning framework, a disclosed environment-driven
engineering decision described in Section 4.4 and revisited as a
limitation in Section 7.4.

## 2.6 Reinforcement Learning Applied to Cybersecurity

Beyond alert correlation specifically, reinforcement learning has been
applied across several cybersecurity subareas, surveyed broadly by Nguyen
and Reddi (2021). Adaptive intrusion-*response* and mitigation-selection
systems learn which defensive action to take given an observed attack
state — a *response* decision, made after detection, rather than the
correlation decision this thesis addresses. Autonomous penetration-testing
and red-teaming agents (e.g., Schwartz and Kurniawati, 2019) learn
attack-*path* selection within a network model — an *offensive* planning
problem, structurally closer to classical RL benchmarks (an agent
navigating toward a goal state through a graph of hosts) than to
correlating already-observed alerts. RL-based intrusion *detection* (as
opposed to correlation) has also been explored directly — for example,
Lopez-Martin, Carro, and Sanchez-Esguevillas (2020) apply a DQN to network
intrusion detection framed as a classification problem — exactly the kind
of RL-for-*detection* application this thesis's scope commitments
(Section 1.3) deliberately rule out for its own identification stage. This broader
literature motivates RL as a viable tool in the security domain generally,
but the great majority of it targets a different problem than this
thesis: either the *detection*/*response* decision (this project
deliberately excludes RL from detection and response, restricting it to
the narrower correlation decision, Section 1.3) or an *offensive*
planning problem (attack-path selection), rather than the *defensive,
post-hoc correlation* problem this thesis addresses. This project's
specific combination — RL restricted to a binary correlation decision,
evaluated with a genuinely external, structurally different test set — is
comparatively underrepresented in this literature relative to RL-based
detection work, which more commonly reports same-distribution validation
results only, a gap this thesis's Chapter 6 finding speaks directly to.

## 2.7 Datasets

**DARPA2000** (MIT Lincoln Laboratory) is used as this project's external,
held-out evaluation set. It comprises two labeled multi-stage intrusion
scenarios, LLDOS 1.0 and LLDOS 2.0.2, and remains, as of this writing, one
of the few datasets with genuine, dataset-native multi-phase campaign
ground truth designed specifically to exercise correlation research — a
property this project's own Stage 2 work confirmed newer flow-centric
datasets do not have natively (Chapter 4). Its age and known limitations
(Section 7.3) are a standard, acknowledged tradeoff in the correlation
literature, not unique to this project: newer datasets with realistic
traffic volumes have generally not been constructed with genuine
multi-phase campaign ground truth as a design goal, which is itself an
observation this project's Stage 2 work makes concrete (Section 2.7,
CICIDS2017 below).

**CICIDS2017** (Sharafaldin, Lashkari, and Ghorbani, 2018) is used as the
primary training and same-distribution validation dataset for both the
Stage 3 identifier and the Stage 4 correlator. It is a large, modern,
labeled network-traffic dataset built specifically to address well-known
shortcomings of older intrusion-detection datasets (limited traffic
diversity, unrealistic background traffic, missing modern attack types),
and has become one of the most widely used benchmarks in recent network-
intrusion-detection ML research. It ships no native multi-stage campaign
labels, however — every attack type is captured as a separate, single-
technique file — which is precisely why Stage 2 had to derive campaign
structure rather than read it directly (Section 4.2), and is the dataset
property most directly responsible for this project's insistence on a
second, genuinely campaign-labeled dataset (DARPA2000) for the headline
evaluation.

**UNSW-NB15** (Moustafa and Slay, 2015) is used as an optional, secondary
cross-check dataset for the identification methodology only (Chapter 4,
Stage 3b). It was constructed specifically to be a harder, more modern
alternative to older benchmarks, with a hybrid of real modern and
synthetically generated attack traffic across nine attack categories, and
is widely acknowledged in the classification literature as a harder,
noisier classification problem than CICIDS2017 — consistent with, and
part of the motivation for treating as unsurprising, this project's own
Stage 3b finding of substantially lower classification performance on it
(Section 5.2).

## 2.8 Positioning This Thesis Relative to Prior Work

Relative to the correlation literature summarized in Section 2.2, this
thesis's contribution is not a new correlation algorithm family in the
abstract, but a specific, disciplined combination of an existing one (DQN,
Section 2.4–2.5) with an evaluation methodology — campaign-level leakage-
free splitting enforced in code, and headline evaluation on a genuinely
external, structurally different dataset — that is comparatively rare in
the ML-based alert-correlation and RL-for-cybersecurity literature
(Section 2.6), where same-distribution validation is more commonly
reported as the primary or only result. Section 6.1's central finding — a
strong same-distribution result that does not transfer externally — is
offered as direct empirical evidence for why that evaluation methodology
matters, not only as a caveat about this particular model.

# Chapter 3 — System Design

## 3.1 Design Requirements Derived from Chapter 2

Five requirements, each traceable to a specific point in Chapter 2's
review, drove the system-design decisions in this chapter. First, Section
2.2's observation that ML-based correlation is the family most exposed to
data leakage argues for leakage prevention being a first-class, code-
enforced property of the pipeline rather than an evaluation afterthought
(Section 3.4). Second, Section 2.3's kill-chain framing argues for an
explicit, shared stage taxonomy that both source datasets can be mapped
onto, so that stage-progression information is available to the
correlator regardless of which dataset an event came from (Section 3.2).
Third, Section 2.4-2.5's action-space argument for DQN over policy-gradient
methods argues for a correlation decision framed as a genuine, discrete,
multi-step MDP rather than a one-shot pairwise classification problem
(Section 3.5). Fourth, Section 2.8's positioning of this thesis around
external evaluation argues for a system architecture that can run
unmodified against a structurally different second dataset, which in turn
requires the shared schema described next. Fifth, Section 2.2's
observation that a rule-based baseline is the natural comparison point for
any learned correlator argues for isolating that comparison to the
decision function alone — which requires the learned agent and the
baseline to sit inside one shared reconstruction procedure rather than
each having its own bespoke pipeline (Section 3.6), so that a difference
in results can be attributed to *what* decides, not to incidental
differences in *how* decisions get turned into clusters.

## 3.2 Two-Model Architecture

The pipeline deliberately separates two learning problems that are often
conflated in prior alert-correlation work (Section 2.2):

- **Identification** (Stage 3): given a single alert's features, predict
  which attack technique it most likely represents, out of a fixed
  taxonomy. Solved with a supervised Random Forest classifier. No RL
  involvement whatsoever.
- **Correlation** (Stage 4/5): given a candidate pair of already-identified
  alerts (one being the current campaign's "anchor," one a candidate to
  link), decide whether they belong to the same campaign. Solved with the
  DQN agent described in Chapter 2. The correlator consumes the
  identifier's *actual* predicted labels — including its real
  misclassifications — not ground-truth labels, so that Stage 4's training
  signal reflects a realistic downstream operating condition rather than an
  idealized one.

Keeping these as two separately trained, separately evaluated models,
rather than one end-to-end system, has a direct evaluation benefit
beyond the scope commitment in Section 1.3: it lets Chapter 6 attribute
weaknesses to a specific stage where possible (e.g., noting that Stage
4's CICIDS2017 training absorbs Stage 3's actual identification errors,
Section 7.4) rather than reporting one opaque combined error rate that
conflates two different kinds of mistakes.

## 3.3 Unified Event Schema

Both DARPA2000 and CICIDS2017 are converted into a single shared event
schema (`src/schema.py`): timestamp, source/destination IP and port,
service, sensor tier (network / web / host), campaign identifier,
kill-chain stage, attack type, and a same-campaign link indicator. This
shared schema is what allows one correlation environment and one DQN agent
implementation to operate over both datasets unmodified, and is a
prerequisite for the leakage-free split machinery described next. It plays
the same normalizing role in this project that common event schemas
(e.g., the vendor-neutral Common Event Format, or a SIEM vendor's common
information model) play in production SOC tooling — allowing downstream
logic to be written once against a stable field set rather than once per
sensor vendor or dataset — narrowed here to exactly the fields the
correlation task needs (Section 2.3's kill-chain stage among them) rather
than the much larger field sets those production schemas typically carry.

The `kill_chain_stage` field is populated differently for each source, a
distinction stated explicitly rather than glossed over: DARPA2000's value
is derived from the dataset's own documented phase structure (genuine
ground truth, Section 2.7), while CICIDS2017's value is either a
Stage-3-predicted label (train/validation time, Section 3.2) or, in the
one place ground truth is used for CICIDS2017, a direct mapping from its
native single-technique label. This asymmetry — one dataset supplying
ground-truth stage information, the other supplying a mix of predicted and
mapped-from-native-label values — is a real methodological wrinkle,
revisited directly in Section 6.1's discussion of Stage 5's idealized
DARPA2000 identification input.

The original proposal envisioned a genuine three-tier telemetry design
(network/firewall, web/WAF, host/endpoint). In the delivered system, only
the network tier is actually populated with real sensor data end to end;
this gap, and why it exists, is discussed explicitly in Section 7.2.

## 3.4 Leakage-Free, Campaign-Level Splitting

Section 2.2 identified data leakage as the ML-correlation literature's
characteristic risk, and Section 1.4 makes avoiding it this thesis's
governing principle; this section is where that principle is translated
into a specific, code-enforced mechanism rather than an evaluation-time
promise. `src/leakage.py` provides two functions that are the concrete
backing for every "no leakage" claim made about this project's results:
`assert_no_campaign_leakage`, which verifies that no campaign identifier
appears on both sides of a train/test split, and
`assert_dataset_is_test_only`, which verifies that a designated test-only
data source (DARPA2000) never appears in a training dataframe. These
assertions run inline in the actual training and evaluation scripts — they
are not documentation of an intended property, but a property the code
will refuse to proceed past if violated. Splitting is deliberately done at
the *campaign* level, not the individual-event level: a naive event-level
random split would place some of a campaign's events in training and
others in test, letting the model implicitly learn campaign-specific
identifiers (a shared timestamp cluster, a specific host pair) rather than
generalizable correlation signal — precisely the shortcut-learning failure
mode Section 2.2 warns is endemic to this literature, and the same failure
mode that Stage 3's fixed-attacker-IP feature (Section 4.3) was found to
be exploiting before it was removed.

## 3.5 Correlation Environment: MDP Formulation and Reward Design

`src/correlation_env.py` implements the RL environment underlying the
DQN agent described in Section 2.4–2.5, formalized as the following MDP:

- **State.** A seven-dimensional feature vector comparing a candidate
  event against the current campaign anchor: temporal proximity (time
  elapsed since the anchor), shared-host and shared-service indicators,
  and a small set of features encoding the anchor's and candidate's
  relative kill-chain stage progression (Section 2.3, Section 3.3).
- **Action.** Binary: link the candidate event to the current campaign, or
  reject it (Section 2.5).
- **Transition.** Accepting a link updates the episode's anchor to the
  newly linked event, changing the state features available for the next
  decision — the genuine, non-cosmetic multi-step structure argued for in
  Section 2.2 and Section 3.1. Rejecting a candidate leaves the anchor
  unchanged and advances to the next candidate in the episode's event
  pool.
- **Episode.** Built from a single campaign's true events plus sampled
  same-time-window distractor events (Section 4.2), capped at 60 events
  per episode (Section 7.7).
- **Reward.** A correct link and a correct rejection are both rewarded; an
  incorrect link (false positive) is penalized; a missed link (false
  negative) is penalized more heavily than a false positive, reflecting
  the higher real-world cost of a security analyst missing part of a
  campaign versus investigating one spurious link. The specific false-
  negative weighting was tuned once during Stage 4 (Section 4.4) after
  observing a degenerate policy at an initial setting, and this single,
  disclosed adjustment is revisited as a limitation in Section 7.4.
- **Discount factor.** Applied over the genuine multi-step episode
  structure above, consistent with the standard MDP formulation in
  Section 2.4 rather than treated as a free hyperparameter divorced from
  the problem's actual sequential structure.

Restricting the state to this compact, seven-dimensional, structurally
interpretable feature vector — rather than, say, raw alert text or the
full unified-schema record — is itself a design choice made in direct
response to Section 2.2's shortcut-learning risk: a smaller, hand-chosen
feature set is easier to audit for leakage than a large, opaque one, at
the cost of expressiveness the agent might otherwise have used
productively. Whether this tradeoff was well-judged is left open, and
Section 6.1's discussion of the DQN's generalization gap should be read
with this restricted state representation as one candidate contributing
factor.

## 3.6 Campaign Reconstruction: A Shared, Decision-Function-Agnostic Procedure

Section 3.5's MDP describes how the DQN makes one pairwise link/don't-link
decision. Turning a stream of such decisions into actual campaign clusters
— and, separately, turning the rule-based baseline's decisions into
clusters for comparison — is a distinct design problem, addressed by
`src/campaign_reconstruction.py`. The requirement identified in Section
3.1 is that this project's headline comparison (Section 5.4) needs to
isolate *what decides* — the trained DQN's policy versus the fixed
shared-IP-and-time-window rule — from *how a sequence of pairwise
decisions becomes a partition of events into clusters*, since conflating
the two would leave any observed difference in results ambiguous between
"the decision function is better" and "the surrounding algorithm happens
to suit one decision function more than the other."

The design response is a single reconstruction procedure, parameterized
only by a decision function, used identically for the trained DQN's
greedy policy and for the rule-based baseline: events are processed in
chronological order, each new event is compared against every currently
open cluster (one recently active enough to still be a plausible match)
via the supplied decision function, and the event joins the most recently
active cluster that votes to link, or starts a new cluster if none do.
Both the DQN and the baseline are then just different implementations of
the one function this procedure needs — `decide(anchor_event,
candidate_event) -> bool` — plugged into the same surrounding logic. This
is a deliberate architectural choice, not an incidental implementation
detail: it is what makes the Section 5.4/Section 6.1 comparison between
the DQN and the baseline a comparison of decision quality specifically,
rather than a comparison confounded by two different clustering
algorithms built around two different decision functions. The procedure's
concrete parameters (the "open" window, the baseline's own decision rule)
are given in Section 4.6.

# Chapter 4 — Methodology

## 4.1 Stage 1 — Data Acquisition and Verification

DARPA2000 was acquired via a scripted download and verified (checksums,
file presence) via `scripts/stage1_download_darpa2000.sh` and
`scripts/stage1_verify_darpa2000.py` — the only one of the three datasets
with no manual step anywhere in its acquisition.

CICIDS2017 required a manual initial download step because its original
CIC mirror now redirects all direct file requests back to its info page;
the dataset's current maintainer, York University's BCCC lab, gates
access behind a request form (institutional email and a stated research
purpose), documented step by step in `docs/stage1_manual_downloads.md`
alongside a noted fallback (a third-party Kaggle mirror of the same
CICFlowMeter-generated CSVs) that was deliberately not used for this
project's headline correlation ground truth, since it is a re-upload
rather than the canonical source. Once the request-form link was
obtained, extraction and organization into `data/raw/cicids2017/` were
scripted. An earlier download attempt produced a corrupted archive after
mixing download tools across resumed sessions; this was resolved by
re-downloading the entire archive in a single, uninterrupted connection
rather than attempting to patch or resume the corrupted file.

UNSW-NB15 — scoped from the outset as an optional, secondary dataset —
was gated behind a browser-only OneDrive/SharePoint folder with no
scriptable, anonymous download path at all; it was obtained later in the
project by downloading the relevant CSV parts through a browser session
and transferring them onto this machine via `scp` (Stage 3b, Section
4.5), the one dataset-acquisition step in this project with no scripted
component whatsoever.

## 4.2 Stage 2 — Unified Schema, Campaign Derivation, Leak-Free Split

**DARPA2000.** Campaign membership was derived from the data itself, not
assumed: an internal host is treated as a genuine campaign member only if
it appears in the scenario's phase-4 (installation/compromise) records —
i.e., it was actually compromised, not merely probed. Two real bugs
surfaced and were fixed while building this logic. First, an initial
version derived campaign hosts from phases 4 *and* 5 combined; phase 5 is
the DDoS-launch phase, which uses spoofed source IPs by design, and this
pulled thousands of one-off forged addresses into the host set. This was
fixed by deriving hosts from phase 4 only. Second, even after that fix, the
attacker's own external IP address remained eligible to satisfy the
membership check on either side of a packet, making the check for the
LLDOS 1.0 scenario nearly vacuous (every reconnaissance probe involves the
attacker's IP, whether or not the probed host was ever compromised). This
was fixed by restricting host-matching to the simulated network's internal
address space, so membership reflects whether the *target* was actually
compromised. After both fixes, LLDOS 1.0 yields a genuine, non-degenerate
split of 150 true campaign-linked events against 134 same-time-window
distractor events (dead-end probes against hosts that were never
compromised); LLDOS 2.0.2 was confirmed, by direct inspection of the raw
data, to have zero within-scenario distractors as a real, documented
property of that scenario's design (it only probes the host it goes on to
compromise), not a parsing artifact — with the practical consequence that
LLDOS 2.0.2 alone cannot supply negative examples for evaluation and must
be pooled with LLDOS 1.0 (Section 4.6). The DDoS flood itself
(approximately 67,000 near-duplicate spoofed packets) was collapsed into
one-second aggregate events, reflecting realistic SOC alert granularity
rather than one alert per raw packet. The final DARPA2000 output is 316
unified events, explicitly reserved as test-only data.

**CICIDS2017.** CICIDS2017 ships no native multi-stage campaign labels,
and the signal that worked for DARPA2000 (shared host) does not
discriminate here, since every attack type in the dataset was generated by
the same fixed attacker VM against the same fixed victim host. Campaigns
were instead derived via time-gap clustering: attack-type blocks occurring
within 60 minutes of each other on the same collection day were merged
into a single campaign. This derived grouping was cross-checked for face
validity against the officially published CICIDS2017 attack schedule
(Sharafaldin et al., 2018) and matched it exactly, producing six
campaigns spanning four collection days. This is explicitly a
**constructed** grouping, not native ground truth, and is treated as such
throughout this thesis. The resulting campaigns were split four for
training and two for validation at the campaign level (leakage-free, verified
in code), benign distractor events were sampled at five times each day's
attack-event count (capped for tractability), and the same flood-
aggregation policy used for DARPA2000 was applied. The final CICIDS2017
output is 58,057 unified events (13,720 true campaign-linked, 44,337
distractor). Each CICIDS2017 event also carries a `predicted_attack_type`
and `predicted_kill_chain_stage` scored by Stage 3's trained classifier at
parse time — real predictions, not ground truth — which is what lets
Stage 4 train against the identifier's actual error profile (Section 3.1).

## 4.3 Stage 3 — Supervised Attack Identification

A Random Forest classifier (`n_estimators=150`, `max_depth=20`,
`min_samples_leaf=3`, `class_weight='balanced'`, fixed random seed) was
trained on CICIDS2017's flow-level features to predict one of 14
attack-type classes (13 attack categories plus Benign), using a
chronological per-file split — each source file's first 70% of timestamps
for training, last 30% for testing — to avoid mixing time-adjacent flow
records between train and test, since floods (DoS Hulk, port scans, DDoS)
produce tens of thousands of near-identical consecutive rows that a random
row split would otherwise leak across the boundary. An early version of
the feature set included the source and destination IP address and port,
which produced suspiciously strong performance; investigation confirmed
this was a shortcut exploiting CICIDS2017's fixed attacker/victim IP
pairing (the attacker IP appears in every attack type's flows and in none
of 50,000 sampled benign rows) rather than a genuine per-technique signal,
and all four fields were removed from the feature set. Class imbalance was
handled with `class_weight='balanced'` rather than resampling.

Two individual classes that scored near-perfect metrics after that fix
(DDoS_LOIT, Port_Scan) were separately investigated via confusion matrices
and feature importances before being accepted as genuine rather than
leaked (results in Section 5.1). This investigation produced a
methodological lesson that shaped how Stage 3b's own near-perfect class
was later checked (Section 4.5): an initial audit pass flagged near-perfect
classes by F1 score alone, which caught DDoS_LOIT but missed Port_Scan,
whose F1 (0.97) was pulled down by moderate precision even though its
recall was 0.999 — essentially perfect and just as worth auditing for
leakage as a high F1 score would have been. A second, row-normalized-recall
pass over the confusion matrix was added specifically to catch this case,
and was applied to every subsequent stage's near-perfect-class checks in
this project.

## 4.4 Stage 4 — DQN Correlation Agent Training

The DQN agent (`src/dqn_agent.py`) implements Section 3.5's MDP with the
following concrete choices. The Q-function is a two-hidden-layer
multilayer perceptron (32 and 16 units, ReLU activations, Adam optimizer,
learning rate 1e-3), implemented via `scikit-learn`'s `MLPRegressor` with
`partial_fit` used to perform one gradient step per training call rather
than a framework-native training loop (Section 2.5, revisited as a
limitation in Section 7.4); the network has one output per action —
Q(s, don't-link) and Q(s, link) — trained by leaving the untaken action's
regression target equal to the network's own current prediction for that
action and overwriting only the taken action's target with the real TD
target, so gradients only push on the action actually selected. A separate
target network, synced from the policy network every 200 training steps by
copying its weights directly, supplies the TD targets and is never itself
updated by gradient descent, which is what prevents the "moving target"
instability of plain online Q-learning. Experience replay uses a
20,000-transition fixed-size buffer with random-sampled minibatches of 64,
decorrelating consecutive transitions from the same episode. The discount
factor is gamma = 0.9. Exploration follows an epsilon-greedy schedule,
linearly decayed from 1.0 to 0.05 over the first 2,000 of the run's 3,000
total training episodes, then held at 0.05 for the remainder.

Concretely, the seven state features (Section 3.5) compare a candidate
event against the current campaign anchor: (1) time elapsed since the
anchor, clipped to one hour and scaled to [0, 1]; (2) shared source IP;
(3) shared destination IP; (4) any shared IP across the anchor's and
candidate's source/destination pair (catching a pivot, where the
candidate's source equals the anchor's destination); (5) shared service;
(6) matching `predicted_attack_type`; and (7) the difference between the
candidate's and anchor's ordinal `predicted_kill_chain_stage` position
(recon=0, exploit=1, install=2, action=3), capturing whether the candidate
plausibly escalates the kill chain. The reward is +1.0 for a correct link,
-1.0 for an incorrect link (false positive), +0.1 for a correct rejection
(true negative — deliberately not equal to the link reward, so the agent
is not tempted to default to rejecting on this majority-negative task),
and, for a missed true link (false negative), -2.5 as reported (originally
-5.0; Section 5.3 reports both configurations' full results). Each episode
is built from one CICIDS2017 campaign's true events, subsampled and capped
at 60 events, chronologically interleaved with same-day distractor events
pulled from a 30-minute window around the campaign; the campaign's first
true event is seeded as the starting anchor, since this project's stated
scope is deciding whether two events belong together, not discovering
where a campaign starts (Section 1.3).

The agent was trained on the four training campaigns from Stage 2's
CICIDS2017 split, with the two validation campaigns used for periodic
greedy-policy evaluation (epsilon = 0) every 100 episodes during training,
never for gradient updates. An initial reward configuration weighting
false negatives at 5x the link reward (-5.0) produced a degenerate
"almost-always-link" policy; this was diagnosed from the training curves
and the false-negative weight was reduced to 2.5x (-2.5), a single
disclosed adjustment (revisited as a limitation in Section 7.4).
`assert_dataset_is_test_only` and `assert_no_campaign_leakage` both run at
the start of this training script, re-verifying — not merely trusting
Stage 2's split — that no DARPA2000-derived row is present in the training
data and that no campaign identifier appears on both sides of the
train/validation boundary.

## 4.5 Stage 3b — UNSW-NB15 Cross-Check (Optional, Secondary)

UNSW-NB15 was scoped from the outset as an optional secondary check of
whether the Stage 3 *approach* — not the specific trained model, which is
architecturally inapplicable across datasets with non-overlapping feature
schemas — generalizes to a second, independent network-traffic dataset. An
independent Random Forest, using the same hyperparameters, class
weighting, and leakage discipline as Stage 3 (`n_estimators=150`,
`max_depth=20`, `min_samples_leaf=3`, `class_weight='balanced'`), was
trained and evaluated entirely on UNSW-NB15's own canonical train/test
split (`UNSW_NB15_training-set.csv` / `testing-set.csv`, 175,341 training
rows, 82,332 test rows) — not a split this project constructed. Before
training, the raw `sttl`/`dttl` (TTL) columns were checked directly against
class labels, because UNSW-NB15's literature documents a known TTL
artifact concern; the marginal distributions were indeed starkly different
between benign and attack traffic (Normal traffic overwhelmingly
`sttl=31`, most attack categories 97–100% `sttl=254`). Rather than
assuming this constituted leakage, two models were trained on the same
split and compared: a "naive" model retaining all 42 usable feature
columns including `sttl`/`dttl`, and an "honest" model using the same 42
minus those two columns (40 features), with the honest model's numbers
used as Stage 3b's headline result regardless of which way the ablation
came out.

## 4.6 Stage 5 — Headline External Evaluation on DARPA2000

Stage 5 is the project's headline result: the trained DQN agent (Stage 4,
trained and tuned exclusively on CICIDS2017) is used, unmodified, to
reconstruct campaigns on DARPA2000 — genuinely external data that no
training or tuning code ever read, enforced in code via
`assert_dataset_is_test_only`. Because LLDOS 2.0.2 has no within-scenario
distractors of its own (Section 4.2), the two DARPA2000 scenarios are
pooled for this evaluation so the correlator has real negative examples to
reject; pooling requires no artificial time-shifting, since the two
scenarios are naturally over a month apart (March 7 and April 16, 2000)
and any reasonable time-windowed method keeps them separate on its own.
Because Stage 3's identifier cannot run on DARPA2000 at all (its
122-column CICIDS2017 flow-statistics feature schema has no equivalent in
DARPA2000's parsed tcpdump-session records), DARPA2000's own ground-truth-
derived attack type and kill-chain stage are used as the DQN's
identification input for this evaluation — an explicitly disclosed
idealized condition that CICIDS2017 training never had, discussed further
in Section 6.3 and Chapter 7.

Campaigns are reconstructed with one shared online procedure
(`src/campaign_reconstruction.py`), applied identically to the trained DQN
and to the baseline so that the comparison isolates the decision function
itself rather than the surrounding algorithm: events are processed in
chronological order; for each new event, every currently "open" cluster —
one whose most recently added event occurred within the last hour, the
same one-hour window used as Stage 4's time-feature cap (Section 4.4) — is
compared against it via the decision function, most-recently-active
cluster first, and the event joins the first cluster that votes "link,"
or starts a new one if none do. This deliberately differs from Stage 4's
training-time procedure, which only ever compared a candidate against a
single evolving anchor for tractability; Stage 5's reconstruction compares
each new event against every open cluster, closer to how a production
correlation engine would actually operate online. The rule-based baseline
uses the same reconstruction procedure with a fixed decision function
(link if and only if the two events share an IP address and fall within
30 minutes of each other) in place of the trained DQN's greedy policy. The
DQN's and baseline's reconstructed campaigns are compared against
DARPA2000's ground-truth campaign labels using two metrics: Adjusted Rand
Index (ARI) and pairwise precision/recall/F1, both reported because, as
described in Chapter 6, they disagree on this data.

## 4.7 Stage 6 — Threats to Validity

Stage 6 is not a modeling stage but a systematic, evidence-traced
accounting of every limitation surfaced across Stages 1–5, restated in
full in Chapter 7 of this thesis.

## 4.8 The Correctness Review

After Stages 1–5 produced an initial set of results, the full codebase was
independently re-reviewed for correctness, empirically testing suspicious
mechanisms rather than only reading code. Three real bugs were found and
fixed; both affected stages (4 and 5) were retrained and re-evaluated from
the fix, and the pre-fix results were kept on disk rather than deleted. A
second, follow-up review pass specifically checked whether the fix itself
had introduced any new problems; it found none, and additionally verified
event-ID uniqueness across both processed datasets and empirically
confirmed the DQN's target-network synchronization mechanism behaves
correctly. The full mechanics of all three bugs and both review passes are
detailed in Section 6.2, since the review's findings materially changed
this thesis's headline numbers and are treated as a first-class part of
the results, not an appendix note.

# Chapter 5 — Results

This chapter reports the outcome of each stage in the order it was run,
including per-class detail and the specific investigative steps taken to
rule out inflated or leaked results, not only aggregate headline numbers.
Interpretation of what these results mean for the project's central
claims — the cross-dataset generalization gap and the correctness review —
is deferred to Chapter 6; this chapter's job is to state, precisely and
with evidence, what happened at each stage.

## 5.1 Stage 3: Supervised Identification (CICIDS2017)

Trained and evaluated on 751,810 flow records across 14 classes (13 attack
types plus Benign), using a chronological per-source-file 70/30 split
(751,810 total rows train+test; 225,552 rows in the reported test set).
Two leakage defenses were applied and verified, not assumed: source/
destination IP and port were excluded from the feature set after directly
confirming that CICIDS2017's fixed attacker IP (172.16.0.1) appears in
every attack type's flows but in none of 50,000 sampled benign rows —
leaving IP in would let the model key on attacker identity rather than
flow behavior; and the train/test split was taken chronologically within
each source file rather than as a random row split, since floods (DoS
Hulk, port scans, DDoS) produce tens of thousands of near-identical
consecutive rows that a random split would otherwise leak across the
train/test boundary as near-duplicates.

| Metric | Value |
|---|---|
| Accuracy | 85% |
| Weighted F1 | 0.86 |
| Macro F1 | 0.65 |

The gap between weighted and macro F1 reflects real class-imbalance cost,
visible directly in the per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.49 | 0.91 | 0.64 | 30,001 |
| Botnet_ARES | 1.00 | 0.25 | 0.41 | 1,653 |
| DDoS_LOIT | 0.99 | 1.00 | 0.99 | 28,720 |
| DoS_GoldenEye | 0.93 | 0.86 | 0.89 | 2,510 |
| DoS_Hulk | 1.00 | 0.76 | 0.87 | 104,773 |
| DoS_Slowhttptest | 0.69 | 0.62 | 0.65 | 2,058 |
| DoS_Slowloris | 0.97 | 0.41 | 0.57 | 1,554 |
| FTP-Patator | 1.00 | 0.48 | 0.64 | 2,860 |
| Heartbleed | 0.01 | 0.50 | 0.01 | 4 |
| Port_Scan | 0.94 | 1.00 | 0.97 | 48,397 |
| SSH-Patator | 0.93 | 0.96 | 0.94 | 1,785 |
| Web_Brute_Force | 0.67 | 0.28 | 0.39 | 821 |
| Web_SQL_Injection | 1.00 | 0.75 | 0.86 | 8 |
| Web_XSS | 0.31 | 0.28 | 0.30 | 408 |

**Near-perfect classes were individually investigated, not accepted at
face value.** DDoS_LOIT (F1 = 0.994) resolves to 28,698/28,720 correct,
with the 22 errors all falling to Benign and no suspicious clean-boundary
pattern; its top feature importances are dominated by inter-packet-
arrival-time statistics, a well-documented, genuine DDoS flood signature
(near-zero, highly uniform packet timing) rather than a residual identity
leak, since IP/port had already been excluded. Port_Scan (recall = 0.999
on n = 48,397, though its F1 of 0.97 is pulled down by precision = 0.944,
which meant it was not initially flagged by an F1-threshold check and was
only caught by a second, row-normalized-recall pass over the confusion
matrix) shows a clean recall story — only 62 of 48,397 true port-scan
flows missed — alongside a real precision cost: 1,813 Benign and 1,071
Botnet_ARES flows are misclassified as Port_Scan, because short probe-like
connections (background benign traffic, botnet C2 beaconing) superficially
resemble a port scan's flow-statistic signature. Both are judged legitimate
easy classes rather than leakage artifacts. The methodological lesson
carried forward from this check: an F1-based near-perfect filter alone is
insufficient, since a class can have suspiciously perfect recall while its
F1 stays moderate if precision happens to be lower — precision and recall
need to be checked separately, not only their harmonic mean.

**Honest weak spots, reported rather than hidden behind the aggregate
accuracy figure:**

- **Benign precision = 0.49 despite recall = 0.91.** The classifier
  over-predicts Benign, which means a meaningful fraction of real attack
  flows are misclassified as benign — false negatives from a security
  standpoint, and the single most consequential weakness in this table
  given what Stage 3's predictions feed into downstream.
- **Web_XSS (F1 = 0.30) and Web_Brute_Force (F1 = 0.39).** Web-layer
  attacks are difficult to fingerprint from flow statistics alone, since
  this feature set has no HTTP payload visibility — an expected
  limitation of the feature representation, not a training defect.
- **Web_SQL_Injection (F1 = 0.857) and Heartbleed (F1 = 0.01) rest on only
  8 and 4 test examples respectively.** Both are flagged here explicitly
  as statistically untrustworthy at this support level and should not be
  read as solid per-class numbers.

Stage 4's DQN correlator consumes this classifier's *predicted*
attack_type as part of its state representation (Section 3.5) — never
ground truth — so Benign's low precision and Web_XSS/Web_Brute_Force's
weak F1 mean the correlator's input labels are sometimes genuinely wrong
during both training and evaluation. This is treated as a realistic
feature of the two-stage design (a real SOC's upstream identifier is never
perfect either), and its downstream consequences are revisited in Chapter
7.

## 5.2 Stage 3b: UNSW-NB15 Cross-Check

This stage is optional and secondary (Section 4.5): it does not apply
Stage 3's trained model to a second dataset — UNSW-NB15's 49-column schema
shares no columns with CICIDS2017's 122, making that architecturally
impossible, exactly as for DARPA2000 — but instead trains and evaluates an
independent Random Forest, using the same methodology and leakage
discipline as Stage 3, entirely on UNSW-NB15's own data and its own
canonical train/test split (175,341 / 82,332 rows). It checks whether the
*approach* generalizes, not whether one specific model does.

**A leakage hypothesis that was tested, not assumed.** UNSW-NB15's `sttl`/
`dttl` (TTL) columns are documented in the literature as a possible
leakage concern, and the raw marginal distribution looked exactly like the
kind of shortcut Stage 3 had to guard against: Normal traffic is
overwhelmingly `sttl=31`, while most attack categories sit at 97–100%
`sttl=254` (Shellcode 100%, Fuzzers 99.6%, Generic 99.4%, Reconnaissance
99.5%) — a simulation-configuration artifact (attack traffic generated
from VMs with a different default TTL than the background-traffic
generator), not a learned attack behavior. Rather than assuming this
mattered, both a "naive" model (all 42 features, including sttl/dttl) and
an "honest" model (40 features, sttl/dttl excluded) were trained:

| | Naive (with TTL) | Honest (without TTL) | Delta |
|---|---|---|---|
| Accuracy | 0.6805 | 0.6795 | +0.0010 |
| Macro F1 | 0.4940 | 0.4814 | +0.0126 |
| Weighted F1 | 0.7328 | 0.7315 | +0.0013 |

Removing the suspected leakage columns barely changed performance —
reported precisely because it is the less dramatic, less satisfying
outcome of the two possible ablation results. The likely explanation is
that the other 40 features (byte counts, packet timing, service/state
patterns) already carry enough overlapping signal that the model does not
need `sttl`/`dttl` specifically to reach a similar decision boundary. The
**honest model's numbers are used as this stage's headline regardless**,
since excluding a column with no real attack-behavior justification is the
correct methodological call independent of whether it moves the number.

**Headline (honest model, UNSW-NB15's own held-out test set, 10 classes):**

| Metric | Value |
|---|---|
| Accuracy | 68.0% |
| Macro F1 | 0.481 |
| Weighted F1 | 0.732 |

Substantially more modest than Stage 3's CICIDS2017 numbers — the expected,
honest shape of the same methodology meeting a different, independently
acknowledged harder classification problem (10 classes here vs. 14 there),
not a failure of the approach. Per-class detail for the honest model:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Generic | 1.00 | 0.96 | 0.98 | 18,871 |
| Normal | 0.99 | 0.60 | 0.75 | 37,000 |
| Reconnaissance | 0.86 | 0.84 | 0.85 | 3,496 |
| Exploits | 0.76 | 0.63 | 0.69 | 11,132 |
| Worms | 0.55 | 0.55 | 0.55 | 44 |
| Fuzzers | 0.25 | 0.68 | 0.36 | 6,062 |
| Shellcode | 0.20 | 0.91 | 0.33 | 378 |
| DoS | 0.31 | 0.19 | 0.23 | 4,089 |
| Backdoor | 0.04 | 0.35 | 0.07 | 583 |
| Analysis | 0.01 | 0.02 | 0.01 | 677 |

`Generic` (F1 = 0.981, n = 18,871) was individually investigated given its
near-perfect score, exactly as DDoS_LOIT and Port_Scan were in Stage 3: its
confusion-matrix breakdown (18,179/18,871 correct, only 5 total false
positives from other classes) shows near-perfect precision with errors
concentrated as false negatives into semantically adjacent categories
(Exploits, Fuzzers, DoS), and its top feature importances are dominated by
`sbytes`/`smean` (payload byte-size statistics) — UNSW-NB15's "Generic"
category specifically covers a block-cipher-attack technique with a
documented, consistent payload-size signature, a genuine, non-leaky
pattern rather than an artifact. Several minority classes show weak
precision despite reasonable recall — Analysis (precision 0.01), Backdoor
(0.04), Shellcode (0.20) — a real cost of `class_weight='balanced'`
pulling false positives in from the much larger classes to boost rare-class
recall, mirroring the same precision/recall tradeoff documented for
CICIDS2017's Benign class in Section 5.1.

This stage is genuinely secondary: it does not change any Stage 3/4/5
result, and Stage 3's actual trained model was never touched. Its value is
narrow — it shows the leakage-aware supervised-classification approach
used in Stage 3 produces sensible, non-inflated numbers on a second,
independent network dataset, and that a plausible leakage hypothesis was
checked with an actual ablation rather than assumed true from the
literature or ignored because it was inconvenient to test.

## 5.3 Stage 4: DQN Correlator Training (CICIDS2017, Same-Distribution)

Three training configurations are reported in full, reflecting the
project's disclosed reward-tuning history and the later correctness-review
fix (Section 4.8) — not only the final, best-performing configuration.

| Run | Configuration | Mean Val. F1 | Mean Val. Precision | Mean Val. Recall | Best Checkpoint F1 |
|---|---|---|---|---|---|
| Run 1 | FN penalty = 5x link reward | 0.838 (std 0.031) | 0.730 (std 0.041) | 0.984 (std 0.020) | 0.886 (ep. 1200) |
| Run 2 | FN penalty = 2.5x link reward | 0.889 (std 0.034) | 0.808 (std 0.048) | 0.989 (std 0.023) | 0.946 (ep. 1200) |
| Run 3 (corrected, current) | FN penalty = 2.5x, post state-feature fix | 0.873 (std 0.027) | 0.793 | 0.972 | — |

**Run 1's failure was diagnosed, not simply discarded.** Its reward used a
5x asymmetric penalty for missed links (matching the original design
rationale that missing a real campaign link is worse than one bad
grouping). The resulting validation curve shows precision oscillating
noisily with no improving trend while recall stays pinned near 1.0
throughout — the agent converged toward an almost-always-link policy.
Training-loss inspection rules out an inert network: TD loss dropped from
2.46 to 1.41 over training, so the network was genuinely learning to fit
its own bootstrapped targets, but the resulting policy never discovered a
sharper discriminator. The likely mechanism is that with the
false-negative penalty dominating the reward scale, "link when uncertain"
is a locally rational strategy, compounded by limited state diversity —
only 4 training campaigns, one of which (Heartbleed-derived) has just 12
events.

**Reducing the FN-penalty asymmetry to 2.5x (Run 2) produced a real,
across-the-board improvement**, not a single lucky checkpoint: +0.051 mean
F1, +0.078 mean precision, at essentially unchanged (still near-perfect)
recall. This was one disclosed, one-time adjustment made in response to a
specific diagnosed failure mode, not repeated tuning toward a target
number, and both runs' full results remain on disk
(`data/processed/stage4_run1_fn5x/`, `data/processed/stage4_run2_fn2.5x/`)
for direct inspection.

**Run 2's training curve is not monotonic, and is reported as such.**
Performance peaked around episodes 1100–1300 (F1 up to 0.946), partially
regressed in the back third of training (episodes 2200–2900 dip as low as
F1 = 0.818), then recovered somewhat by the final checkpoint (F1 = 0.885).
The training script's own automated stability heuristic (comparing the
first third of checkpoints' mean F1 against the last third's) reports "no
clear improvement" for this run, because the first third's average happens
to be pulled up by the strong 1100–1300 window while the last third
absorbs the 2200–2900 dip — a coarse check that cannot distinguish "never
learned anything" from "learned, peaked, then partially drifted," which is
what the full checkpoint-by-checkpoint record shows actually happened.
Both the automated verdict and the fuller record are reported here rather
than relying on either alone. The saved policy is the full run's final
state, not a cherry-picked best checkpoint — a deliberate choice favoring
this project's commitment to not tuning toward favorable numbers over the
otherwise-legitimate practice of reporting a best-validation checkpoint.

**Run 3** retrains Run 2's exact reward configuration after the
correctness review's state-feature fix (Section 4.8, Section 6.2); its
CICIDS2017 validation numbers (mean F1 0.873 vs. Run 2's 0.889) are
essentially unchanged, within noise, consistent with that bug having had
no material effect on CICIDS2017 training specifically — its impact was
concentrated in Stage 5's DARPA2000 evaluation (Section 5.4). Run 3 is the
model held out in `data/processed/stage4/` and the one Stage 5 evaluates;
Run 3's precision/recall (0.793 / 0.972) on the two held-out validation
campaigns confirms recall remains consistently strong while precision
stays moderate, the same pattern carried forward into Section 5.4.

## 5.4 Stage 5: Headline External Evaluation (DARPA2000)

Pooled evaluation across LLDOS 1.0 and LLDOS 2.0.2 (316 unified events,
never read by any training or tuning code — enforced at runtime by
`assert_dataset_is_test_only`, not only asserted in prose), reconstructing
campaigns with a shared online procedure (Section 4.6) applied identically
to both the trained Stage 4 DQN (greedy policy) and a naive rule-based
baseline (link iff shared IP and within 30 minutes), so the comparison
isolates the decision function rather than the surrounding algorithm.
These are the corrected, post-fix numbers (Section 4.8); the pre-fix
figures are kept on disk in `data/processed/stage5_buggy_run/` for direct
comparison rather than being silently replaced.

| Metric | DQN | Rule-Based Baseline |
|---|---|---|
| Pairwise Precision | 0.317 | 0.251 |
| Pairwise Recall | 0.208 | 0.746 |
| Pairwise F1 | 0.251 | 0.375 |
| Adjusted Rand Index | 0.0806 | 0.0369 |

The correctness-review fix (Section 4.8) moved every one of the DQN's
pairwise numbers in the right direction relative to the pre-fix run
(precision 0.259→0.317, recall 0.160→0.208, F1 0.198→0.251) and flipped
the ARI comparison in the DQN's favor (0.0224→0.0806, vs. the baseline's
unchanged 0.0369). The DQN still does not outperform the naive baseline on
pairwise F1 even after the fix — reported here as the honest headline
finding, not reframed.

**The two metrics disagree about which method wins, and the mechanism
behind that disagreement was traced directly in the cluster output**
(`data/processed/stage5/darpa2000_reconstruction.csv`), not left as an
unexplained discrepancy. The rule-based baseline transitively over-links:
its "any shared IP within 30 minutes" rule chains almost everything into
one 262-event cluster (83% of all 316 events), because DARPA2000's fixed,
small IP space means many unrelated events share an address with
*something* nearby in time. This produces 26,063 false-positive pairs, and
inflates recall by brute-force over-connection rather than genuine
discrimination — exactly the pattern ARI's chance-correction penalizes.
The DQN, post-fix, forms fewer and more purposeful clusters (17, down from
22 pre-fix) with markedly better true-campaign concentration — e.g., one
38-event cluster is 89% LLDOS 1.0 members — which is what drives its ARI
lead, even though its raw pairwise recall remains well below the
baseline's indiscriminate approach.

**The DQN still under-links relative to the baseline, even post-fix.**
LLDOS 2.0.2 (32 events, confirmed in Stage 2 to have zero
within-scenario distractors — a low-ambiguity case with no risk of
incorrect merging) should trivially cluster as one group; the baseline
gets this exactly right (all 32 events, one cluster, completeness 1.00).
The DQN fragments it into 3 clusters post-fix (improved from 4 pre-fix,
but still not 1), missing the "exploit" stage in its best-matching
cluster.

Kill-chain completeness (fraction of a campaign's kill-chain stages
represented in the reconstructed cluster), unchanged by the fix:

| Scenario | DQN | Baseline |
|---|---|---|
| LLDOS 1.0 | 1.00 | 0.75 |
| LLDOS 2.0.2 | 0.75 | 1.00 |

Not a clean sweep either way. On LLDOS 1.0 specifically — the scenario
with real distractors to reject, and therefore the more realistic test of
the two — the DQN's best-matching cluster captures the full four-stage
kill-chain narrative despite its fragmentation elsewhere, while the
baseline misses the "exploit" stage entirely.

The DQN loses to the naive baseline on pairwise F1 but wins on ARI — a
genuine metric disagreement, not a reporting artifact, whose implications
for the project's central generalization-gap finding are discussed in
Section 6.1. This is the project's headline result and is reported as
such, even though it is not a clean win for the learned approach.

# Chapter 6 — Discussion

## 6.1 The Central Finding: A Real Cross-Dataset Generalization Gap

The single most important result of this project is the gap between
Stage 4's strong same-distribution validation performance (mean F1 ≈
0.87–0.89 on held-out CICIDS2017 campaigns) and Stage 5's external
DARPA2000 evaluation, where the DQN's pairwise-link F1 (0.251) falls below
a naive rule-based baseline (0.375). This is precisely the failure mode a
same-distribution validation split cannot reveal on its own, and precisely
why this project's insistence on a genuinely external, held-out test set
mattered: reporting only the ~0.87–0.89 validation number would have
reproduced exactly the kind of inflated, non-generalizing result this
thesis's governing principle (Section 1.4) exists to avoid.

Investigation traced part of the shortfall to the DQN under-linking events
even in a near-zero-ambiguity case (LLDOS 2.0.2, 32 events, no distractors)
that the naive baseline handled correctly by construction. The most
plausible mechanism is that a policy learned on CICIDS2017's specific
campaign structure — event volumes ranging from 12 to 4,153 per campaign,
a fixed two-host attacker/victim topology — does not transfer cleanly to
DARPA2000's structurally different event density and timing. Notably,
DARPA2000's identification input to the DQN was DARPA's own ground-truth-
derived labels rather than a trained classifier's predictions (Section 4.6)
— an idealized condition CICIDS2017 training never had — which, if
anything, should have made DARPA2000's correlation task *easier* for the
DQN than its CICIDS2017 training was. That the DQN still underperformed
under this more favorable condition makes the generalization-gap finding
more concerning, not less.

The two evaluation metrics used in Stage 5 disagree on which method is
better: ARI ranks the DQN ahead (0.081 vs. 0.037), while pairwise F1 ranks
the baseline ahead (0.375 vs. 0.251). This is a known behavior of ARI's
chance-correction when comparing partitions with very different
cluster-count and cluster-size profiles — exactly the situation here,
since DARPA2000's ground truth is dominated by many true singleton
distractor "clusters," while the baseline collapses the large majority of
DARPA2000's events into one giant cluster. The DQN forms fewer, more
concentrated clusters, which ARI rewards; the baseline's brute-force
over-linking still catches more raw true pairs, which pairwise recall (and
therefore F1) rewards. Both metrics are reported, deliberately, rather than
selecting whichever tells a cleaner story.

## 6.2 The Correctness Review as a Methodological Contribution

A self-directed, full-codebase correctness review was conducted after
Stages 1–5 had already produced an initial set of reported results. It
found and fixed three real bugs:

1. **(High severity)** `_ordinal_stage()` in the correlation environment
   looked up an event's predicted attack type in a CICIDS2017-only
   vocabulary dictionary. DARPA2000's aliased attack-type values share zero
   keys with that dictionary, which meant the seventh of the DQN's seven
   state features (stage progression) was silently constant at exactly 0.0
   for all 315 DARPA2000 event pairs used in Stage 5 — a bug that had
   directly affected the already-reported headline results. It was fixed by
   deriving a properly computed kill-chain-stage field for both datasets
   and reading it directly, rather than re-deriving it through a
   dataset-specific lookup. Stage 4 was retrained from scratch and Stage 5
   was fully re-evaluated on the fix; the corrected numbers changed
   materially (ARI flipped from the DQN trailing the baseline, 0.0224 vs.
   0.0369, to leading it, 0.0806 vs. 0.0369; pairwise F1 improved from
   0.198 to 0.251, though the baseline's 0.375 remained ahead either way).
2. **(Medium severity)** An operator-precedence bug in a training-stability
   report (`-n // 3` parses as `(-n) // 3` in Python, not the intended
   `-(n // 3)`). This did not affect the specific numbers already reported,
   since the run in question happened to have an evenly divisible episode
   count, but would have silently misreported the comparison for other
   configurations.
3. **(Low severity)** Distractor events sampled into an episode's event
   pool but positioned chronologically before the seeded campaign anchor
   were never reachable as comparison candidates, since the training loop's
   cursor only moves forward from the anchor. This quietly reduced negative-
   example exposure below what the sampling budget implied.

All three bugs were disclosed and fixed, not silently patched; the affected
stages were retrained and re-evaluated rather than leaving stale numbers
standing, and the pre-fix results were retained on disk for direct
comparison rather than deleted. A second, follow-up review pass then
specifically checked whether the fix itself had introduced any new
problems and whether anything else had been missed. It found no further
bugs, and went beyond re-reading code: it empirically confirmed zero
duplicate event IDs across both processed datasets (316 DARPA2000 rows,
58,057 CICIDS2017 rows — a collision here would have silently corrupted
several downstream dictionary-keyed computations), and it directly
exercised the DQN's target-network synchronization mechanism (a manual
implementation, since the project uses `scikit-learn`'s `MLPRegressor`
rather than a standard deep-learning framework), confirming that the
policy and target networks start identical, measurably diverge between
scheduled sync points, and exactly re-converge at each sync — i.e.,
confirming the target network genuinely functions as intended rather than
silently being always equal to, or always different from, the policy
network.

This episode is treated in this thesis as a limitation, not a strength
(Section 7.6): the results reported here were not correct on the first
attempt, and no number of correctness passes, however thorough, is a
guarantee against further undiscovered bugs, here or in any codebase. What
can be stated concretely is exactly what was checked and how — which is
the standard this thesis holds itself to throughout, rather than a general
assurance of correctness.

## 6.3 Error Propagation from Identification to Correlation

The two-stage design (Section 3.2) deliberately trains Stage 4 on Stage
3's actual predicted labels, not ground truth (Section 3.1), on the
reasoning that a real deployed system would face the same imperfect
upstream signal. Section 5.1 quantifies exactly how imperfect that signal
is on CICIDS2017: Benign precision of only 0.49 means a substantial share
of true attack flows enter Stage 4's training as mislabeled benign events,
and Web_XSS's F1 of 0.30 means that class's label is wrong more often than
not. Since the DQN's state vector is built from `predicted_attack_type`
(Section 3.5) — including the `stage_progression` feature whose
CICIDS2017-specific vocabulary lookup was itself the source of the
correctness review's highest-severity bug (Section 6.2) — a systematically
noisy identification signal could plausibly teach the DQN to rely on
state-transition patterns that reflect Stage 3's error structure as much
as genuine campaign structure, and there is no experiment in this thesis
that isolates how much of that risk actually materialized.

This confound is instructive precisely because of where it is, and is
not, present. It applies to Stage 4's CICIDS2017 training and to the
same-distribution validation numbers reported in Section 5.3, but it does
*not* apply to Stage 5's DARPA2000 evaluation, whose identification input
is DARPA's own ground-truth-derived labels rather than a trained
classifier's output (Section 4.6). That asymmetry cuts against the most
convenient explanation for Section 6.1's generalization gap: if the DQN's
DARPA2000 shortfall were mainly a matter of noisy training supervision on
CICIDS2017, removing that noise at evaluation time should have made
DARPA2000 easier, not merely equally hard — and Section 6.1 already notes
the DQN underperformed there regardless. The identification-error
propagation risk documented here is therefore a genuine, disclosed
weakness in how Stage 4's CICIDS2017 numbers should be read (Section 7.4),
but it does not, on its own, account for the cross-dataset result that
this thesis treats as its central finding. Disentangling the two effects
directly — retraining Stage 4 against CICIDS2017 ground-truth labels
rather than Stage 3's predictions, holding everything else fixed — is the
single most direct way to settle the question, and is named explicitly as
future work (Section 8.2, item 5) rather than left as an open, unaddressed
possibility.

## 6.4 What the Metric Disagreement Would Mean in Practice

Section 5.4's numbers can be translated into what each method would
actually hand an analyst. The rule-based baseline's high recall (0.746)
is achieved by transitively merging 83% of all 316 DARPA2000 events into
one 262-event cluster — it does find most true links, but it does so by
finding almost every link, true or not (26,063 false-positive pairs), which
defeats alert correlation's actual purpose (Section 2.1): an analyst
handed one near-undifferentiated 262-event blob has not had their triage
burden meaningfully reduced, even though the method "wins" on recall. The
DQN's smaller, more concentrated clusters (17, with individual clusters
reaching 89% purity) are more usable in exactly the sense Section 2.1
cares about, at the cost of missing roughly four in five true links
(recall 0.208) — a link the DQN never proposes is a connection between two
events an analyst may never think to examine together at all, arguably a
worse operational failure than a spurious link buried in an over-large
cluster, which at least keeps the two genuinely linked events in the same
group even if surrounded by noise. This is an interpretation, not a
finding this thesis's evaluation was designed to test directly — no user
study or analyst-workload measurement was conducted — but it is a
defensible reading of why pairwise recall specifically, not only the
combined F1, deserves attention when judging which of these two failure
modes is more costly in a real SOC.

The kill-chain-completeness metric (Section 5.4) offers a third, more
favorable lens on the same result: on LLDOS 1.0 — the scenario with real
distractor events to reject, and so the more realistic of the two — the
DQN's best-matching cluster reconstructs the full four-stage kill-chain
narrative (completeness 1.00) despite its lower raw pairwise recall,
while the baseline's over-linked approach misses the "exploit" stage
entirely (completeness 0.75) precisely because its indiscriminate
merging is not actually organized around any one coherent campaign
narrative. If an operational goal is "can an analyst find the story," not
only "does the tool recover the maximum number of individual true pairs,"
this specific result is a genuinely favorable data point for the DQN that
the headline pairwise-F1 comparison alone does not surface — a further,
concrete reason Section 5.4 reports multiple metrics rather than
collapsing the evaluation into one number.

## 6.5 Reassessing the Contribution: A Mixed Result as the Central Finding

Section 2.6 observes that the RL-for-cybersecurity literature more
commonly reports same-distribution validation as its primary or only
result, and Section 2.8 positions this thesis's methodological
contribution — campaign-level leakage-free splitting enforced in code, and a
headline evaluation on a genuinely external, structurally different
dataset — as a direct response to that gap. Read against that backdrop,
Section 6.1's finding is not a disappointing result that happens to be
what this project got; it is the specific kind of result that
methodology exists to be capable of producing, and a project that used
the same methodology but happened to find a clean DQN win would not have
demonstrated the methodology was doing any real work. A same-distribution
validation F1 of 0.87–0.89 (Section 5.3) is, by itself, indistinguishable
from a genuinely generalizing correlator or from an overfit one; only the
external DARPA2000 result (Section 5.4) — and the willingness to report it
even though it complicates the thesis's narrative — actually distinguishes
between those two possibilities.

The correctness review (Section 6.2) sits in the same relationship to this
thesis's credibility as the external-evaluation methodology does: both are
mechanisms for finding out whether an initially favorable-looking result
survives scrutiny, and in this project neither mechanism was decorative.
The review changed the headline DARPA2000 numbers in the DQN's favor on
one metric (ARI) and only partially closed the gap on another (pairwise
F1), rather than resolving the story cleanly either way — and that
partial, metric-dependent correction is itself a more credible outcome
than a review that had conveniently fixed everything. Taken together, a
project that (1) proactively found and disclosed three real bugs that
materially changed its own already-reported results and (2) still reports
a headline comparison where the proposed method does not outright win on
the field's most standard metric is offering a stronger form of evidence
for its own trustworthiness than a clean win would have, precisely because
an unqualified success is the harder result to distinguish from the kind
of inflated, non-generalizing claim Section 1.4 and Section 2.8 both
identify as a known failure mode in this literature.

# Chapter 7 — Threats to Validity and Limitations

This chapter states precisely what was implemented, what was
experimentally evaluated, and what remained only conceptually proposed,
consistent with this thesis's governing principle (Section 1.4).

## 7.1 Scope Commitments, Restated

Both binding commitments from Section 1.3 held throughout the project:
Stage 3 (identification) is an ordinary Random Forest with zero RL
involvement; Stage 4 (correlation) is the only stage where RL appears, and
it never classifies raw traffic; no stage of the pipeline runs inline, and
every stage processes bounded, offline batches of already-collected
events.

## 7.2 Data Limitations

**CICIDS2017's campaign labels are derived, not native ground truth.**
Unlike DARPA2000, whose phase structure is the dataset's own documented
design, CICIDS2017 ships no multi-stage campaign labels at all — every
attack type is a separate, single-technique capture. Stage 2 derived
campaigns via time-gap clustering (attack-type blocks within 60 minutes of
each other on the same collection day merged into one campaign, Section
4.2), cross-checked for face validity against the officially published
CICIDS2017 attack schedule (Sharafaldin et al., 2018) and found to match
it exactly, producing six campaigns across four collection days. That
exact match is reassuring but does not change what the label is: a
constructed grouping, not something CICIDS2017's creators labeled as
multi-stage, and every result trained or tuned against this split should
be read with that distinction in mind.

**No genuine web/WAF-tier alert data exists anywhere in this project.**
CICIDS2017's `web_sql_injection`/`web_xss` captures remain network-
flow-level (5-tuple and flow statistics) — verified directly against the
schema during Stage 2, not assumed — with no HTTP method, URL, or payload
field of any kind, so they are not genuine web/WAF-tier alerts despite
describing web-layer attacks. The originally planned DVWA plus
OWASP-CRS/ModSecurity supplementary source, which would have produced real
WAF-style alerts, was deferred early in the project as an explicit scoping
decision and never revisited. Concretely: in every result this thesis
reports, the project's "web tier" is still network-tier data — a real,
disclosed gap between the proposal's three-tier design and what was
actually evaluated.

**UNSW-NB15 was obtained outside this project's scripted pipeline, and
its cross-check has real limits.** It sat gated behind a browser-only
SharePoint folder with no scriptable access until it was downloaded
manually and transferred onto this machine — the one dataset in this
project not acquired through an automated, reproducible download step
(Section 4.1). Stage 3b's independent model shows the leakage-aware
*approach* used in Stage 3 generalizes reasonably to a second dataset
(Section 5.2), not that Stage 3's specific, deployed CICIDS2017 classifier
does — that classifier was never touched or re-validated against
UNSW-NB15 by this exercise.

**Benign/distractor sampling used a fixed heuristic, not an exhaustively
tested one.** CICIDS2017 benign rows were sampled at five times each
collection day's attack-row count, capped at 50,000 rows/day with a fixed
random seed (Section 4.2) — a deliberate, disclosed choice made for
computational tractability on the 2-core/4GB machine this project ran on,
not an oversight, but a different sampling rate or method could plausibly
shift the class-imbalance handling and downstream results somewhat, and
that sensitivity was not tested.

## 7.3 DARPA2000-Specific Limitations

**Age and tooling.** DARPA2000 is from March/April 2000 and predates
modern web attacks entirely; its documented attack tooling (the `sadmind`
RPC exploit in LLDOS 1.0, the `mstream` DDoS trojan in LLDOS 2.0.2)
reflects that era's network- and host-layer intrusion techniques, not
current ones. It remains the canonical labeled multi-stage correlation
benchmark specifically because of its genuine, dataset-native multi-phase
ground truth, a property this project's own Stage 2 work confirmed newer
flow-centric datasets do not have (Section 2.7) — a real, stated tradeoff,
not one avoided by using it anyway.

**Small N, and an uneven pair at that.** The entire external headline
evaluation rests on two campaigns from one simulated network and one
red team's tooling; statistical power at N = 2 campaigns is minimal, and
"the DQN generalizes worse than a naive baseline to DARPA2000" is a
narrower claim than "the DQN generalizes worse than a naive baseline to
external network intrusion data in general." The two campaigns are not
even a matched pair: LLDOS 1.0 is the noisier scenario, with 150 true
campaign-linked events set against 134 same-time-window distractor events
to reject (Section 4.2), while LLDOS 2.0.2 is confirmed, by direct
inspection of the raw data, to have zero within-scenario distractors at
all. This asymmetry is useful for exposing two different failure modes —
Section 5.4's kill-chain-completeness results differ across the two
scenarios in exactly the way this difference would predict — but it is not
a broad behavioral sample of attacker activity, and N = 2 is too small to
separate "this specific scenario's structure" from "DARPA2000 in general"
as an explanation for either campaign's result.

**No web/WAF tier represented at all.** DARPA2000 predates HTTP-based
attacks, so the headline evaluation exercises only the network and host
tiers of the originally proposed three-tier design, and even the host tier
is not used, since Stage 5's evaluation runs on network-derived events
only. Genuinely three-tier correlation was only ever exercised in concept
on the CICIDS2017/DVWA side of the project, and DVWA was never built
(Section 7.2).

**Idealized identification signal.** DARPA2000's identification input to
the DQN was DARPA's own ground-truth-derived `attack_type`/
`kill_chain_stage`, not a trained classifier's predictions (Section 4.6) —
structurally necessary, since Stage 3's 122-column CICIDS2017 flow-
statistics schema has no equivalent in DARPA's parsed tcpdump-session
records, but still an idealized condition CICIDS2017 training never had.
What this idealization does, and does not, imply for reading Section 6.1's
central finding is discussed in full in Section 6.3.

## 7.4 Model and Training-Methodology Limitations

**One disclosed reward-asymmetry adjustment.** The DQN's false-negative
penalty was changed once, from 5x to 2.5x the link reward, after
diagnosing that 5x produced a degenerate, almost-always-link policy
(Section 5.3) — a single, disclosed, diagnosis-driven change with both
runs' full results reported side by side, not repeated tuning toward a
target score. It is still worth naming as a departure from a purely
first-principles reward design validated before any training occurred:
the final reward weights were informed by one round of observed training
behavior, a common and defensible practice in RL, but not the same claim
as a reward scheme fixed by analysis alone.

**Limited training-campaign diversity.** The DQN trained on only four
CICIDS2017 campaigns (Stage 2's leakage-free split), one of which — the
Heartbleed-derived campaign — has just 12 events. This is a narrow
experience base for learning a generalizable Q-function, and is a
plausible contributor to the generalization gap discussed in Section 6.1.

**`scikit-learn`'s `MLPRegressor` in place of a standard deep-RL
framework.** No CPU PyTorch package was available through this
environment's package manager, and PyPI downloads of that size repeatedly
stalled under this environment's network conditions — the same
constraint Stage 3's setup hit before switching to apt (Section 4.3). The
implementation includes all of DQN's real components (experience replay,
a separate target network, epsilon-greedy decay with a genuine discount
factor over real multi-step episodes, Section 2.3), built on a small
two-hidden-layer network sized to the seven-dimensional state space in
use. This is a defensible engineering substitution for this project's
constraints, but a reader should know the underlying optimizer and
framework differ from most published DQN work, which could plausibly
affect training dynamics in ways not explored here.

**Stage 3's classification errors propagate into Stage 4's CICIDS2017
training by design.** The correlator trains on Stage 3's actual predicted
labels, not ground truth (Section 3.1) — the more realistic condition, but
one that also means any weakness in Stage 4's same-distribution results
could originate in Stage 3's identification errors (Benign precision 0.49,
Web_XSS F1 0.30, Section 5.1) rather than in Stage 4's own
correlation-learning design. No ablation isolating these two effects was
run; the mechanism and why it does not fully explain the DARPA2000 result
specifically is analyzed in Section 6.3, and separating the two effects
directly is named as future work (Section 8.2).

## 7.5 Evaluation-Methodology Limitations

**Adjusted Rand Index and pairwise F1 disagree on the Stage 5 results**
(Section 6.1), a known behavior of ARI's chance-correction when comparing
partitions with very different cluster-count and cluster-size profiles,
which is exactly the situation between DARPA2000's ground truth and the
rule-based baseline's near-total over-linking (Section 5.4). Pairwise
precision/recall/F1 was added and reported as the primary metric
specifically because of this disagreement, with ARI — the metric named in
the original proposal — kept and reported alongside it rather than
dropped. This is a genuine metric-choice limitation, not a hypothetical
one: the two metrics do not merely differ quantitatively here, they
disagree on which method is better, and a thesis defense should be
prepared to explain both readings rather than cite whichever is more
favorable.

**The operational reading offered in Section 6.4 — that missed links may
cost a SOC more than over-linked noise — was not itself tested.** No
analyst-workload measurement or user study was conducted to confirm which
of the two failure modes (the baseline's over-linking or the DQN's
under-linking) is actually more costly in practice; Section 6.4's argument
is a defensible interpretation of the pairwise and kill-chain-completeness
numbers, not an evaluated claim, and should be read as such.

**Only one genuinely external dataset (DARPA2000) was used.** A more
thorough validation of the generalization-gap finding would test against
at least one further independent, multi-stage-labeled external source;
none was available within this project's scope, and the search for one,
along with the reasoning for why DARPA2000 was the best available choice,
is documented at Stage 1.

## 7.6 The Correctness Review, Restated as a Limitation

The correctness-review episode (Section 6.2) is listed here deliberately
as a limitation, not a strength: the results in this thesis were not
correct on the first attempt. Two correctness passes, including one that
went as far as empirically testing the trickiest custom mechanism in the
codebase rather than only reading it, are not a guarantee against further
undiscovered bugs. What can be stated concretely is exactly what was
checked and how (Section 6.2).

## 7.7 Post-Hoc, Non-Real-Time Scope

Every design choice in this pipeline assumes offline, batch access to a
bounded window of already-collected events, and this shows up concretely
in the implementation, not only as a stated design principle: Stage 4's
training episodes process one campaign's events at a time, subsampled and
capped at 60 events per episode (Section 3.5); Stage 5's reconstruction
procedure compares each new event only against clusters updated within the
last hour (Section 4.6), a reasonable batch/forensic-analysis window but
not a latency or throughput guarantee suitable for inline traffic
inspection. No part of this system was measured for, or designed around,
real-time performance constraints (events-per-second throughput, decision
latency), and no result in this thesis should be read as evidence this
approach would perform acceptably as a real-time or inline system — this
was explicitly out of scope from the original proposal and remained so
throughout.

## 7.8 Project-Wide Summary

| Component | Status |
|---|---|
| Three-tier unified event schema | Implemented (network tier only actually populated; web/host tiers conceptual, Section 7.2) |
| DARPA2000 campaign parsing, leakage-free holdout | Implemented and evaluated |
| CICIDS2017 campaign derivation (time-gap clustering) | Implemented and evaluated; derived, not native, ground truth |
| Stage 3 supervised identifier | Implemented and evaluated on CICIDS2017 only; never re-validated against a second dataset |
| DVWA + OWASP-CRS/ModSecurity web-tier alerts | Never implemented (deferred at Stage 1, never revisited) |
| Stage 4 DQN correlator | Implemented and evaluated on CICIDS2017 (same-distribution validation) |
| Stage 5 DARPA2000 external evaluation | Implemented and evaluated; the project's real headline result |
| Real-time / inline operation | Never implemented, never claimed — explicitly out of scope throughout |
| Stage 3b UNSW-NB15 cross-check | Implemented and evaluated (independent model, dataset's own split, not Stage 3's actual classifier) |

# Chapter 8 — Conclusion and Future Work

## 8.1 Conclusion

This thesis set out to test whether a reinforcement-learning agent could
be used specifically and only for the alert-correlation decision within a
post-hoc, forensic multi-stage-attack-detection pipeline, while holding
attack identification as a separate supervised problem. The system was
built and evaluated end to end, with leakage-free, campaign-level splitting
enforced in code, and with a genuinely external held-out dataset used for
the headline evaluation rather than same-distribution validation alone.

Measured against the six contributions claimed in Section 1.5, the first
two — the shared, leakage-disciplined event schema and campaign-level
split (Section 3.4), and the two-model architecture that trains the
correlator against the identifier's real, imperfect predictions rather
than an oracle (Section 3.2) — were built and hold up under inspection:
the no-leakage guarantee is enforced in code and exercised by every
training script (Sections 4.2, 4.4), and Stage 4's use of Stage 3's actual
predictions, warts included, is what makes the correlator's evaluation
realistic rather than idealized (Section 5.1). The third and fourth — the
DQN correlator's dual same-distribution and external evaluation, and the
honest, fully reported cross-dataset generalization gap — are this
thesis's central empirical contribution, discussed below. The fifth, the
self-directed correctness review, is treated throughout this thesis as a
limitation on the results rather than a triumph over them (Sections 6.2,
7.6), even though the discipline of running it, disclosing what it found,
and re-evaluating rather than quietly patching is itself part of what this
thesis argues a defensible security-ML project should do (Section 6.5).
The sixth, the optional UNSW-NB15 cross-check, delivered exactly its
narrow, stated purpose: evidence that the identification methodology, not
any specific deployed model, transfers to a second dataset (Section 5.2).

The resulting evidence does not support a simple "RL improves alert
correlation" conclusion. The DQN correlator learns a strong policy on
same-distribution CICIDS2017 data (mean validation F1 ≈ 0.87), but that
policy's pairwise-link performance on genuinely external DARPA2000 data
falls below a naive rule-based baseline, even as it leads that baseline on
a different, chance-corrected metric. This is reported as the thesis's
central finding, not an inconvenient footnote, because it is precisely
the kind of result an external-evaluation methodology exists to surface,
and because a thesis that reported only the favorable same-distribution
number would have reproduced the exact failure mode — inflated, non-
generalizing results — that this project's governing principle was
designed to prevent (Section 1.4). The project also demonstrates, through
its self-disclosed correctness-review episode, that rigorous post-hoc
verification of a codebase's actual behavior — not just its results — is
a necessary and productive part of building a defensible ML security
system, not an optional afterthought.

Taken together, this thesis's value does not rest on the DQN having beaten
a naive baseline outright. It rests on having built a working, end-to-end,
leakage-disciplined pipeline; having evaluated it honestly against a
genuinely external test set specifically chosen because it could disagree
with the training-distribution numbers; having found, disclosed, and
corrected real bugs in its own implementation rather than reporting the
first numbers produced; and having reported a mixed, metric-dependent
result exactly as it was found. Section 6.5 argues this combination is
itself stronger evidence of the pipeline's trustworthiness than an
unqualified win would have been — a claim this conclusion does not soften,
even though it means the thesis's headline result is "the field's most
standard correlation metric does not favor the learned approach on
external data," rather than a cleaner story.

## 8.2 Future Work

The following are concrete, specific next steps motivated directly by this
thesis's own findings — not offered as excuses for the limitations
catalogued in Chapter 7.

**Broaden the DQN's training-campaign diversity.** Train across a wider,
more varied set of campaigns, potentially by synthesizing additional
campaign structures from CICIDS2017 at different event densities, to
deliberately expose the agent to more of the event-volume range Section
6.1 identifies as a likely driver of the generalization gap — the current
four-campaign training set, with one campaign as small as 12 events
(Section 7.4), is a narrow experience base to ask a general policy to
emerge from.

**Build the deferred web/WAF-tier data source.** Complete the originally
scoped DVWA plus OWASP-CRS/ModSecurity pipeline to obtain genuine
HTTP-layer alerts and re-run the full three-tier correlation design as
proposed (Section 7.2) — the current project's "web tier" is, in every
reported result, still network-flow data describing web-layer attacks,
and a genuine web/WAF signal could materially change both Stage 3's
Web_XSS/Web_Brute_Force weaknesses (Section 5.1) and Stage 4's downstream
correlation quality on web-attack campaigns.

**Re-validate Stage 3's actual deployed classifier against UNSW-NB15.**
Stage 3b showed the leakage-aware identification *approach* transfers to a
second dataset, but never touched Stage 3's specific, CICIDS2017-trained
model (Section 5.2). A feature-bridging approach — mapping both datasets
down to a small, shared feature subset — would let that specific model's
predictions, not just the methodology, be checked against independent
data.

**Obtain a second external, multi-stage-labeled correlation benchmark.**
DARPA2000 is the only genuinely external dataset used in this thesis
(Section 7.5), and N = 2 campaigns from one simulated network is too small
to separate "the DQN generalizes poorly to DARPA2000 specifically" from
"the DQN generalizes poorly to external network-intrusion data in
general" (Section 7.3). A second, independent benchmark with genuine
campaign ground truth — a scarce resource in this literature, per Section
2.7 — would be the most direct way to test which of those two claims is
actually true.

**Run the ablation isolating Stage 3's errors from Stage 4's own
weaknesses.** Retrain Stage 4 against CICIDS2017 ground-truth attack
labels instead of Stage 3's actual predictions, holding every other
setting fixed, to determine how much of Stage 4's same-distribution result
is attributable to noisy upstream identification versus the correlator's
own learning (Section 6.3). Section 6.3 argues this confound does not, on
its own, explain the DARPA2000 generalization gap, since DARPA2000's
identification input was already idealized — but confirming that directly,
rather than by inference, would close the single largest open question
this thesis leaves about its own CICIDS2017 numbers.

**Test a policy-gradient correlator (PPO) on the same MDP formulation.**
Section 2.5 selected DQN over PPO because the correlation decision is a
small, binary, discrete action space that does not require PPO's
machinery — a reasoned choice, not a default, but one that remains
untested against the alternative in practice. Training a PPO agent on
Section 3.5's identical state/reward formulation and evaluating it the
same way (same-distribution validation, then DARPA2000) would show whether
the generalization gap documented in Section 6.1 is a property of this
correlation problem and its available training data, or specific to DQN's
value-based learning dynamics.

**Extend the DQN with its standard stabilizing variants.** This
project's implementation deliberately excludes Double DQN's decoupled
action selection and prioritized experience replay (Section 2.5). Adding
either is a natural next experiment given Section 5.3's documented
training-curve instability (a mid-run peak around episodes 1100–1300
followed by a partial regression through episodes 2200–2900) — both
extensions exist specifically to address exactly this kind of
overestimation-driven instability in standard DQN training.

**Directly test the operational interpretation offered in Section 6.4.**
This thesis argues, from the pairwise and kill-chain-completeness numbers
alone, that a correlator's missed links may cost a SOC analyst more than
an over-linked cluster's noise — but that argument was never tested
against actual analyst behavior (Section 7.5). A small user study —
presenting analysts with the baseline's and the DQN's DARPA2000
reconstructions and measuring time-to-correct-campaign-identification or
similar — would turn Section 6.4's interpretation into an evaluated
claim.

# References

*Note to the author: the entries below are provided as a starting point
and should be verified and completed against the actual published sources
(full author lists, venue, page numbers, DOI) before submission — exact
bibliographic details were not independently re-verified while assembling
this document.*

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization. *Proceedings of the 4th International Conference on
  Information Systems Security and Privacy (ICISSP)*. (CICIDS2017 dataset.)
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set for
  Network Intrusion Detection Systems (UNSW-NB15 Network Data Set).
  *Military Communications and Information Systems Conference (MilCIS)*.
- MIT Lincoln Laboratory. DARPA Intrusion Detection Scenario Specific
  Datasets — 2000 DARPA Intrusion Detection Scenario Specific Data Sets
  (LLDOS 1.0, LLDOS 2.0.2). MIT Lincoln Laboratory Information Systems
  Technology Group.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control
  through deep reinforcement learning. *Nature*, 518(7540), 529–533.
  (Deep Q-Network.)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Arp, D., Quiring, E., Pendlebury, F., et al. (2022). Dos and Don'ts of
  Machine Learning in Computer Security. *31st USENIX Security Symposium*.
  (General discussion of data-leakage and evaluation pitfalls in ML-based
  security research, cited here as background for this thesis's leakage-
  discipline methodology, Section 1.4.)
- Hutchins, E. M., Cloppert, M. J., & Amin, R. M. (2011). Intelligence-
  Driven Computer Network Defense Informed by Analysis of Adversary
  Campaigns and Intrusion Kill Chains. *Proceedings of the 6th
  International Conference on Information Warfare and Security*. (Lockheed
  Martin Cyber Kill Chain, Section 2.3.)
- MITRE. ATT&CK — a globally accessible knowledge base of adversary
  tactics and techniques. `attack.mitre.org` (Section 2.3).
- Valdes, A., & Skinner, K. (2001). Probabilistic Alert Correlation.
  *Recent Advances in Intrusion Detection (RAID)*. (Statistical
  alert-correlation approach, Section 2.2 — verify exact venue/year.)
- Ning, P., Cui, Y., & Reeves, D. S. (2002). Constructing Attack Scenarios
  through Correlation of Intrusion Alerts. *Proceedings of the 9th ACM
  Conference on Computer and Communications Security (CCS)*. (Causal /
  prerequisite-consequence alert correlation, Section 2.2 — verify exact
  venue/year.)
- Julisch, K. (2003). Clustering Intrusion Detection Alarms to Support
  Root Cause Analysis. *ACM Transactions on Information and System
  Security*, 6(4), 443–471. (Alarm-clustering correlation, Section 2.2 —
  verify exact venue/year/pages.)
- Sundaramurthy, S. C., Case, J., Truong, T., Zomlot, L., & Hoffmann, M.
  (2015). A Human Capital Model for Mitigating Security Analyst Burnout.
  *Symposium on Usable Privacy and Security (SOUPS)*. (SOC alert-fatigue
  and analyst-burnout background, Section 1.1, Section 2.1 — verify exact
  venue/year.)
- Nguyen, T. T., & Reddi, V. J. (2021). Deep Reinforcement Learning for
  Cyber Security. *IEEE Transactions on Neural Networks and Learning
  Systems*. (Broad RL-for-cybersecurity survey, Section 2.6 — verify exact
  volume/pages.)
- Schwartz, J., & Kurniawati, H. (2019). Autonomous Penetration Testing
  using Reinforcement Learning. *arXiv:1905.05965*. (Offensive attack-path
  RL, Section 2.6 — verify exact venue.)
- Lopez-Martin, M., Carro, B., & Sanchez-Esguevillas, A. (2020).
  Application of Deep Reinforcement Learning to Intrusion Detection for
  Supervised Problems. *Expert Systems with Applications*, 141. (RL-based
  intrusion detection, Section 2.6 — verify exact volume/pages.)
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*,
  8(3–4), 279–292. (Section 2.4.)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press. (Standard RL/MDP reference, Section
  2.4, Section 3.5.)
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement
  Learning with Double Q-learning. *Proceedings of the AAAI Conference on
  Artificial Intelligence*. (Double DQN, Section 2.4.)
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized
  Experience Replay. *International Conference on Learning
  Representations (ICLR)*. (Section 2.4.)
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
  Trust Region Policy Optimization. *International Conference on Machine
  Learning (ICML)*. (TRPO, precursor to PPO, Section 2.4.)

# Appendix A — Repository and Reproduction

The full implementation, all per-stage results, and per-stage findings
documents referenced throughout this thesis are available in the project
repository (`README.md` for setup and the exact pipeline run order;
`docs/stage1_manual_downloads.md`, `docs/stage2_findings.md`,
`docs/stage3_findings.md`, `docs/stage3b_unsw_findings.md`,
`docs/stage4_findings.md`, `docs/stage5_findings.md`, and
`docs/stage6_limitations.md` for the full, unabridged per-stage write-ups
this document summarizes; `data/processed/` for all result artifacts,
including the pre-correctness-review results kept alongside the corrected
ones for direct comparison).
