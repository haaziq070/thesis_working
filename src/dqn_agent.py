"""
Stage 4: the DQN agent itself.

Framework note (worth defending explicitly in the viva): this uses
scikit-learn's MLPRegressor as the Q-function approximator rather than
PyTorch/TensorFlow. That's a deliberate, documented choice, not a corner cut:
no CPU-only PyTorch package was available via apt on this machine, and a
fresh PyPI download of the ~200MB PyTorch wheel repeatedly stalled under the
network throttling observed throughout this project (documented in
requirements.txt and the Stage 3 setup notes) -- attempting it would have
re-run into the same multi-hour stall Stage 3's dependency install hit before
apt was used instead. Given the state space here is tiny (7 features, 2
actions), a small MLP is a small MLP regardless of framework: this
implementation has all the real DQN components --

  - a genuine neural network Q-function (a 2-hidden-layer MLP, not a lookup
    table or a linear model)
  - experience replay (a fixed-size buffer, random-sampled minibatches,
    decorrelating consecutive transitions)
  - a separate target network (periodically synced from the policy network,
    used only to compute TD targets, which is what stops the "moving target"
    instability of plain online Q-learning)
  - epsilon-greedy exploration with decay
  - a discount factor (gamma) that has genuine work to do here because the
    environment has real multi-step structure (a wrong link decision moves
    the anchor and changes every subsequent comparison in the episode)

MLPRegressor's partial_fit is used to do the incremental minibatch updates
DQN needs; it natively supports multi-output regression, so the network has
one output per action (Q(s, don't-link), Q(s, link)) exactly like a
standard DQN's two-headed output layer. The standard trick for training a
generic multi-output regressor as a Q-network is used: for a training
example (s, a, target), build the 2-dim regression target as [Q_policy(s,0),
Q_policy(s,1)] (i.e. leave the untaken action's target equal to whatever the
network currently predicts) and then overwrite index a with the real TD
target -- this way the gradient only pushes on the action that was actually
taken.
"""
import copy
import random
from collections import deque

import numpy as np
from sklearn.neural_network import MLPRegressor

ACTION_DIM = 2


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


def _make_network(state_dim, seed):
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,        # one partial_fit call = one gradient step, we control the loop
        warm_start=True,
        random_state=seed,
    )


class DQNAgent:
    def __init__(self, state_dim, gamma=0.9, lr=1e-3, buffer_capacity=20000,
                 batch_size=64, target_sync_every=200, seed=42):
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.rng = np.random.default_rng(seed)

        self.policy_net = _make_network(state_dim, seed)
        self.target_net = _make_network(state_dim, seed)
        self._initialize_networks(state_dim, seed)

        self.replay = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.train_steps = 0
        self.loss_history = []

    def _initialize_networks(self, state_dim, seed):
        # MLPRegressor needs one fit() call before partial_fit works (it sets up
        # the layer shapes from the first call). Fit on a small batch of noise
        # with zero targets so the initial Q-values start near zero, not on
        # whatever sklearn's default init would otherwise imply.
        rng = np.random.default_rng(seed)
        dummy_X = rng.normal(size=(8, state_dim)).astype(np.float32)
        dummy_y = np.zeros((8, ACTION_DIM), dtype=np.float32)
        self.policy_net.fit(dummy_X, dummy_y)
        self.target_net.fit(dummy_X, dummy_y)
        self._sync_target()

    def _sync_target(self):
        self.target_net.coefs_ = copy.deepcopy(self.policy_net.coefs_)
        self.target_net.intercepts_ = copy.deepcopy(self.policy_net.intercepts_)

    def q_values(self, state, net=None):
        net = net or self.policy_net
        return net.predict(state.reshape(1, -1))[0]

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, ACTION_DIM))
        q = self.q_values(state)
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        # store a zero vector as a placeholder for terminal next_states (never
        # used in the TD target computation when done=True, but the buffer
        # needs a fixed-shape array for numpy stacking)
        ns = next_state if next_state is not None else np.zeros(self.state_dim, dtype=np.float32)
        self.replay.push(state, action, reward, ns, done)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        q_current = self.policy_net.predict(states)              # (batch, 2)
        q_next_target = self.target_net.predict(next_states)     # (batch, 2)
        max_next_q = q_next_target.max(axis=1)

        targets = q_current.copy()
        td_target = rewards + (~dones) * self.gamma * max_next_q
        targets[np.arange(len(actions)), actions] = td_target

        loss_before = float(np.mean((q_current[np.arange(len(actions)), actions] - td_target) ** 2))
        self.policy_net.partial_fit(states, targets)
        self.loss_history.append(loss_before)

        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self._sync_target()

        return loss_before
