#!/usr/bin/env python
"""
SAC from Scratch (PyTorch)
=========================
Pure‑PyTorch reference implementation of **Soft Actor‑Critic** (Haarnoja et al., 2018).
No Stable‑Baselines3 or RLlib — everything (networks, replay buffer, updates,
alpha tuning, state‑dependent exploration) is implemented in **one file**.

Run examples
------------
    $ python sac_from_scratch.py --env Pendulum-v1
    $ python sac_from_scratch.py --env MountainCarContinuous-v0 --total_steps 300000 \
          --use_sde True --sde_sample_freq -1

Key features
------------
* **Twin Q‑networks** + target networks (SAC v2)
* **Gaussian policy** with re‑parameterisation trick
* Optional **state‑dependent exploration (gSDE)** with resampling frequency
* Optional **automatic entropy‑temperature tuning** (learns *log α*)
* Uniform **experience replay**
* Periodic evaluation with deterministic actions
* TensorBoard logging (reward, losses, α, etc.) under ./runs/<timestamp>
* All hyper‑parameters exposed as CLI flags (≈ 420 LOC total)
"""

from __future__ import annotations

import argparse
import os
import random
import time
from tqdm import tqdm
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# ──────────────────────────────────────────────────────────────────────────────
#  Hyper‑Parameters (overridden by CLI)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    env="Pendulum-v1",
    seed=0,
    total_steps=250_000,
    start_steps=10_000,           # random actions before SAC updates begin
    eval_interval=5_000,
    n_eval_episodes=10,
    train_freq=1,
    gradient_step=1,
    batch_size=256,
    replay_size=1_000_000,
    gamma=0.99,
    tau=0.005,                    # target network Polyak factor
    lr=3e-4,
    alpha=0.2,                    # fixed entropy temperature (ignored if automatic)
    automatic_entropy_tuning=True,
    target_entropy=None,          # default −|A|
    hidden_dim=64,
    # gSDE flags
    use_sde=0,
    sde_sample_freq=-1,           # −1 ⇒ resample noise only on env.reset()
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ──────────────────────────────────────────────────────────────────────────────
#  Utils
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
#  Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int], act_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int):
        max_mem = self.capacity if self.full else self.ptr
        indices = np.random.randint(0, max_mem, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[indices], device=self.device)
        acts = torch.as_tensor(self.acts_buf[indices], device=self.device)
        rews = torch.as_tensor(self.rews_buf[indices], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[indices], device=self.device)
        done = torch.as_tensor(self.done_buf[indices], device=self.device)
        return obs, acts, rews, next_obs, done

# ──────────────────────────────────────────────────────────────────────────────
#  Networks
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two‑layer MLP for actor and critics."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(state_dim, 2 * action_dim, hidden_dim)
        self.action_dim = action_dim

    def forward(self, state):
        mu_logsigma = self.net(state)
        mu, log_std = torch.chunk(mu_logsigma, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state, return_std = False):
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        x_t = normal.rsample()                         # reparameterisation
        y_t = torch.tanh(x_t)
        action = y_t
        # SAC log‑prob trick (appendix C)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu = torch.tanh(mu)
        if(return_std):
            return action, log_prob, mu, std
        else:
            return action, log_prob, mu


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

# ──────────────────────────────────────────────────────────────────────────────
#  SAC Agent (with optional gSDE)
# ──────────────────────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, cfg):
        state_dim = obs_space.shape[0]
        action_dim = act_space.shape[0]
        # print(obs_space.shape, act_space.shape)
        self.action_scale = torch.tensor((act_space.high - act_space.low) / 2.0, device=cfg.device)
        self.action_bias = torch.tensor((act_space.high + act_space.low) / 2.0, device=cfg.device)
        self.device = cfg.device

        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimisers
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)

        # Entropy temperature α
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = cfg.target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.alpha = cfg.alpha

        # gSDE flags
        self.use_sde = cfg.use_sde
        self.sde_sample_freq = cfg.sde_sample_freq
        self.sde_noise = None
        self.sde_steps = 0

        self.gamma = cfg.gamma
        self.tau = cfg.tau

    # ────────────────────────── helper for env resets ────────────────────────
    def reset_sde(self):
        """Call on env.reset() so that ε is redrawn at episode boundaries."""
        self.sde_noise = None
        self.sde_steps = 0

    # ─────────────────────────────── action select ───────────────────────────
    def select_action(self, state: np.ndarray, evaluate: bool = False, return_mu_std = False):
        state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)

        if evaluate or not self.use_sde:
            # Evaluation or vanilla SAC exploration path
            with torch.no_grad():
                if evaluate:
                    mu, _ = self.policy.forward(state_t)
                    action = torch.tanh(mu)
                else:
                    a, _, mu, std = self.policy.sample(state_t, return_std = True)
                    action = a
        else:
            # gSDE exploration path (deterministic ε for several steps)
            with torch.no_grad():
                mu, std = self.policy.forward(state_t)
                if (self.sde_noise is None or
                    (self.sde_sample_freq > 0 and self.sde_steps % self.sde_sample_freq == 0)):
                    self.sde_noise = torch.randn_like(mu)
                z = mu + std * self.sde_noise                      # state‑dependent!
                action = torch.tanh(z)
                self.sde_steps += 1

        action = action[0] * self.action_scale + self.action_bias
        return action.cpu().numpy()

    # ───────────────────────────── training step ─────────────────────────────
    def update(self, replay: ReplayBuffer, batch_size: int):
        obs, acts, rews, next_obs, done = replay.sample(batch_size)

        # Normalise stored actions back to [−1,1]
        acts_n = (acts - self.action_bias) / (self.action_scale + 1e-6)
        acts_n = torch.clamp(acts_n, -1.0, 1.0)

        # ------------------------- critic update -----------------------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            next_action_scaled = next_action * self.action_scale + self.action_bias
            target_q1 = self.q1_target(next_obs, next_action_scaled)
            target_q2 = self.q2_target(next_obs, next_action_scaled)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
            target = rews + (1 - done) * self.gamma * (target_q - alpha * next_log_prob)

        current_q1 = self.q1(obs, acts)
        current_q2 = self.q2(obs, acts)
        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)

        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()

        # --------------------------- actor update ----------------------------
        new_action, log_prob, _ = self.policy.sample(obs)
        new_action_scaled = new_action * self.action_scale + self.action_bias
        q1_pi = self.q1(obs, new_action_scaled)
        q2_pi = self.q2(obs, new_action_scaled)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
        policy_loss = (alpha * log_prob - min_q_pi).mean()

        self.policy_optim.zero_grad(); policy_loss.backward(); self.policy_optim.step()

        # ---------------------- temperature (α) update -----------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()
            alpha_value = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.); alpha_value = self.alpha

        # -------------------------- target update ----------------------------
        with torch.no_grad():
            for t_p, p in zip(self.q1_target.parameters(), self.q1.parameters()):
                t_p.data.mul_(1 - self.tau); t_p.data.add_(self.tau * p.data)
            for t_p, p in zip(self.q2_target.parameters(), self.q2.parameters()):
                t_p.data.mul_(1 - self.tau); t_p.data.add_(self.tau * p.data)

        return dict(q1_loss=q1_loss.item(), q2_loss=q2_loss.item(),
                    policy_loss=policy_loss.item(), alpha=alpha_value)

# ──────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(agent: SACAgent, env: gym.Env, n_episodes: int):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        agent.reset_sde()                       # fresh ε per eval episode
        done = truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    return float(np.mean(returns))


def run(cfg):
    env = gym.make(cfg.env)
    env.action_space.seed(cfg.seed)

    set_seed(cfg.seed)

    agent = SACAgent(env.observation_space, env.action_space, cfg)
    replay = ReplayBuffer(cfg.replay_size, env.observation_space.shape,
                          env.action_space.shape[0], cfg.device)

    run_name = f"SAC_{cfg.env}_{int(time.time())}"
    writer = SummaryWriter(os.path.join("runs", run_name))

    obs, _ = env.reset(seed=cfg.seed)
    agent.reset_sde()
    episode_return, episode_length = 0.0, 0

    for global_step in tqdm(range(1, cfg.total_steps + 1)):
        if global_step < cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, done, truncated, _ = env.step(action)
        terminal = done or truncated
        replay.add(obs, action, reward, next_obs, float(terminal))

        obs = next_obs
        episode_return += reward
        episode_length += 1

        if terminal:
            writer.add_scalar("charts/episode_return", episode_return, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            obs, _ = env.reset()
            agent.reset_sde()
            episode_return, episode_length = 0.0, 0

        # SAC update
        if global_step >= cfg.start_steps:
            if(global_step % cfg.train_freq == 0):
                for _ in range(cfg.gradient_step):
                    metrics = agent.update(replay, cfg.batch_size)
            if global_step % 1000 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"losses/{k}", v, global_step)

        # Periodic evaluation
        if global_step % cfg.eval_interval == 0:
            eval_ret = evaluate(agent, env, cfg.n_eval_episodes)
            writer.add_scalar("charts/eval_return", eval_ret, global_step)
            print(f"Step {global_step:>7}: eval_return = {eval_ret:.1f}")

    env.close(); writer.close()
    print("Training complete — logs in ./runs/")

# ──────────────────────────────────────────────────────────────────────────────
#  Entry point (CLI)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Soft Actor‑Critic (PyTorch) — from scratch + gSDE")
    for k, v in DEFAULTS.items():
        arg_type = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=arg_type, default=v)
    cfg = parser.parse_args()

    # Merge with defaults to ensure missing keys are filled
    from types import SimpleNamespace
    cfg = SimpleNamespace(**{**DEFAULTS, **vars(cfg)})

    print("Running with config:", cfg)
    run(cfg)
