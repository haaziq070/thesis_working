"""
Stage 4: the pairwise event-correlation environment.

This is the ONLY place in the whole project where reinforcement learning
happens. The RL agent's job is narrow and specific: given an "anchor" event
(the most recent event linked into a growing campaign chain) and a
"candidate" event (the next event under consideration, in chronological
order), decide LINK or DON'T-LINK. It is not a detector -- every event it
sees has already been through Stage 3's supervised classifier, and this
environment only ever sees CICIDS2017 training/validation campaigns.
DARPA2000 never appears here; it is reserved entirely for Stage 5.

State (7 features, built ONLY from Stage 3's *predicted* labels, never
ground truth -- the correlator has to work with what a real identifier
would actually hand it, including its mistakes):
  1. time_delta_norm       time since anchor, seconds, clipped to
                            [0, TIME_CAP_SECONDS] and scaled to [0, 1]
  2. shared_src_ip          1.0 if anchor.src_ip == candidate.src_ip
  3. shared_dst_ip          1.0 if anchor.dst_ip == candidate.dst_ip
  4. shared_any_ip          1.0 if {src,dst} sets intersect (catches pivots:
                            candidate's src == anchor's dst, etc.)
  5. same_service           1.0 if anchor.service == candidate.service
  6. same_predicted_type    1.0 if predicted_attack_type matches
  7. stage_progression      ordinal(candidate's predicted_kill_chain_stage)
                            minus ordinal(anchor's) -- captures "does this
                            plausibly escalate the chain". Reads a
                            `predicted_kill_chain_stage` column directly
                            (already-resolved into the shared
                            recon/exploit/install/action/unknown vocabulary
                            by whichever Stage-2 parser produced the event),
                            rather than re-deriving it from attack_type here
                            -- an earlier version looked attack_type up in
                            CICIDS_LABEL_TO_STAGE, which silently broke on
                            DARPA2000 (whose attack_type vocabulary --
                            recon/exploit/backdoor_install/ddos -- shares no
                            keys with that CICIDS-only dict), making this
                            feature always 0 for every DARPA pair. Confirmed
                            and fixed; see docs/stage5_findings.md.

Action space: Discrete(2) -- 0 = don't-link, 1 = link.

Reward (asymmetric -- justified in the design doc and repeated here):
  link & truly same campaign        -> +1.0   (true positive)
  link & NOT same campaign          -> -1.0   (false positive: an analyst
                                                reviews one wrong grouping)
  don't-link & truly same campaign  -> -5.0   (false negative: the campaign
                                                fragments and the missed
                                                stage goes undetected -- a
                                                materially worse outcome for
                                                a SOC than one bad grouping,
                                                which is why this penalty is
                                                5x the link reward)
  don't-link & NOT same campaign    -> +0.1   (true negative: small reward,
                                                deliberately not equal to the
                                                link reward so the agent
                                                isn't tempted to just always
                                                say no on this
                                                majority-negative task)

Episode structure: one episode = one campaign's events, chronologically
interleaved with same-day distractor (non-campaign) events pulled from a
time window around the campaign. The FIRST true event of the campaign is
seeded as the starting anchor -- this project's stated scope is deciding
whether two events belong together, not discovering where a campaign
starts, so seeding the first link is a deliberate, documented
simplification, not an oversight. When the agent links (correctly or not),
the anchor advances to the candidate, so a wrong link has a real, compounding
consequence on every future comparison in the episode -- this is what gives
the discount factor genuine work to do, rather than reducing the task to a
one-step contextual bandit where gamma would be meaningless.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

TIME_CAP_SECONDS = 3600.0
STAGE_ORDER = {"recon": 0, "exploit": 1, "install": 2, "action": 3, "unknown": -1}

REWARD_TRUE_POSITIVE = 1.0
REWARD_FALSE_POSITIVE = -1.0
# First training attempt used -5.0 (5x the link reward) and converged to a
# near-"always link" policy: recall pinned at ~0.97-1.0 throughout training
# while precision oscillated 0.67-0.80 with no improving trend, and TD loss
# decreased steadily even while the validation policy didn't improve -- i.e.
# the network was successfully learning to fit a reward signal that itself
# rationally favored linking whenever uncertain. Reduced to -2.5 (still
# asymmetric -- missing a real link is still treated as worse than a false
# one, per the original design justification -- just not so dominant that it
# swamps the state-based discrimination signal). This is a disclosed,
# one-time ablation, not repeated tuning toward a target score: both the
# -5.0 and -2.5 runs' results are reported in docs/stage4_findings.md.
REWARD_FALSE_NEGATIVE = -2.5
REWARD_TRUE_NEGATIVE = 0.1

MAX_EPISODE_LENGTH = 60      # cap candidates per episode for tractable training
DISTRACTOR_WINDOW_MINUTES = 30  # pull same-day distractors within this buffer of the campaign


def _ordinal_stage(predicted_kill_chain_stage):
    return STAGE_ORDER.get(predicted_kill_chain_stage, -1)


def _feature_vector(anchor, candidate):
    time_delta = (candidate["timestamp"] - anchor["timestamp"]).total_seconds()
    time_delta_norm = min(max(time_delta, 0.0), TIME_CAP_SECONDS) / TIME_CAP_SECONDS

    anchor_ips = {anchor["src_ip"], anchor["dst_ip"]}
    cand_ips = {candidate["src_ip"], candidate["dst_ip"]}

    return np.array([
        time_delta_norm,
        float(anchor["src_ip"] == candidate["src_ip"]),
        float(anchor["dst_ip"] == candidate["dst_ip"]),
        float(len(anchor_ips & cand_ips) > 0),
        float(anchor["service"] == candidate["service"]),
        float(anchor["predicted_attack_type"] == candidate["predicted_attack_type"]),
        float(_ordinal_stage(candidate["predicted_kill_chain_stage"]) - _ordinal_stage(anchor["predicted_kill_chain_stage"])),
    ], dtype=np.float32)


STATE_DIM = 7


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class CorrelationEnv:
    """One episode = one campaign. Call reset(campaign_id) then step(action)
    repeatedly until done."""

    def __init__(self, events_df, rng_seed=42):
        self.events_df = events_df
        self.rng = np.random.default_rng(rng_seed)
        self._episode_events = None
        self._anchor_idx = None
        self._cursor = None

    def build_episode_pool(self, campaign_id):
        """Chronological true-campaign events + same-day distractors within a
        time window around the campaign. Returns the event list used by reset().

        Both the true-event count AND the distractor count are capped so total
        episode length never exceeds MAX_EPISODE_LENGTH, regardless of how big
        the underlying campaign is (some campaigns, e.g. the merged Friday
        botnet+portscan+DDoS one, have thousands of raw events -- without this
        cap a single episode could run to thousands of steps, which would make
        training intractable and isn't what a bounded correlation window in a
        real SOC would look like anyway). True events are subsampled but kept
        in chronological order, so the anchor still sees a genuine, if sparser,
        campaign progression.
        """
        camp = self.events_df[self.events_df["campaign_id"] == campaign_id].sort_values("timestamp")
        if len(camp) < 2:
            return None
        day = camp["day"].iloc[0]
        start, end = camp["timestamp"].min(), camp["timestamp"].max()
        # window_start is deliberately NOT start - buffer: the anchor is always seeded at
        # the campaign's first event (== start), and step() only ever moves the cursor
        # forward from there, so any distractor sampled earlier than `start` would occupy
        # a pool slot but could never actually be compared against anything -- wasted
        # sampling budget, silently. Only the end-side buffer stays, since post-campaign
        # distractors remain reachable as the cursor advances through them.
        window_start = start
        window_end = end + pd.Timedelta(minutes=DISTRACTOR_WINDOW_MINUTES)

        distractor_pool = self.events_df[
            (self.events_df["campaign_id"] == "NONE")
            & (self.events_df["day"] == day)
            & (self.events_df["timestamp"] >= window_start)
            & (self.events_df["timestamp"] <= window_end)
        ]

        true_budget = min(len(camp), max(2, int(MAX_EPISODE_LENGTH * 0.6)))
        if len(camp) > true_budget:
            # keep the first event (seeds the anchor) always; subsample the rest
            first = camp.iloc[[0]]
            rest = camp.iloc[1:].sample(
                n=true_budget - 1, random_state=int(self.rng.integers(0, 1_000_000))
            )
            camp = pd.concat([first, rest]).sort_values("timestamp")

        distractor_budget = max(0, MAX_EPISODE_LENGTH - len(camp))
        if len(distractor_pool) > distractor_budget:
            distractor_pool = distractor_pool.sample(
                n=distractor_budget, random_state=int(self.rng.integers(0, 1_000_000))
            )

        pool = pd.concat([camp, distractor_pool], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        return pool

    def reset(self, campaign_id):
        pool = self.build_episode_pool(campaign_id)
        if pool is None:
            return None
        self._episode_events = pool
        self._campaign_id = campaign_id
        # seed the anchor at the first true-campaign event (see module docstring)
        true_positions = pool.index[pool["campaign_id"] == campaign_id].tolist()
        self._anchor_idx = true_positions[0]
        self._cursor = self._anchor_idx + 1
        if self._cursor >= len(pool):
            return None
        anchor = pool.iloc[self._anchor_idx]
        candidate = pool.iloc[self._cursor]
        return _feature_vector(anchor, candidate)

    def step(self, action):
        pool = self._episode_events
        anchor = pool.iloc[self._anchor_idx]
        candidate = pool.iloc[self._cursor]

        true_link = (candidate["campaign_id"] == self._campaign_id) and (self._campaign_id != "NONE")

        if action == 1 and true_link:
            reward = REWARD_TRUE_POSITIVE
        elif action == 1 and not true_link:
            reward = REWARD_FALSE_POSITIVE
        elif action == 0 and true_link:
            reward = REWARD_FALSE_NEGATIVE
        else:
            reward = REWARD_TRUE_NEGATIVE

        info = {
            "true_link": true_link,
            "action": action,
            "anchor_event_id": anchor["event_id"],
            "candidate_event_id": candidate["event_id"],
        }

        if action == 1:
            self._anchor_idx = self._cursor  # chain extends (or drifts, if this was wrong)
        self._cursor += 1

        done = self._cursor >= len(pool)
        next_state = None
        if not done:
            next_anchor = pool.iloc[self._anchor_idx]
            next_candidate = pool.iloc[self._cursor]
            next_state = _feature_vector(next_anchor, next_candidate)

        return StepResult(state=next_state, reward=reward, done=done, info=info)
