"""
Implementation of PPO (clip version) using library such as PyTorch, NumPy, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym.spaces import Discrete
import gym
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorCritic(nn.Module):
    """MLP policy and value function, with customized activation functions."""
    def __init__(self, state_dim, action_dim, hidden_dim=64, activation="ReLU", use_sde=False):
        super(ActorCritic, self).__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation function {activation} not supported")

        self.use_sde = use_sde
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.actor_mean)

        if use_sde:
            # For SDE, we predict the log standard deviation from the state
            self.actor_logstd = nn.Linear(hidden_dim, action_dim)
        else:
            # For non-SDE, we use a learnable parameter for log_std
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mean = self.actor_mean(x)
        if self.use_sde:
            log_std = self.actor_logstd(x)
        else:
            log_std = self.actor_logstd.expand_as(mean)
        return mean, log_std, self.critic(x)

class PPO:
    """PPO algorithm."""
    def __init__(self, state_dim, action_dim, hidden_dim=64, activation="ReLU", deterministic_actions=False, 
                 pi_lr=0.001, vf_lr=0.001, train_pi_iters=80, train_v_iters=80, gamma=0.99, lam=0.97, 
                 eps_clip=0.2, steps_per_epoch=4000, n_epochs=100, max_ep_len=1000, seed=0, use_sde=False,
                 entropy_coef=0.01, max_grad_norm=0.5, normalize_rewards=True):
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.max_ep_len = max_ep_len
        self.use_sde = use_sde
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_rewards = normalize_rewards

        # Reward normalization parameters
        self.ret_rms = RunningMeanStd()
        self.cliprew = 10.0

        # Initialize actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, activation, use_sde).to(device)
        self.deterministic = deterministic_actions

        # Initialize optimizers
        self.pi_optimizer = optim.Adam(self.actor_critic.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.actor_critic.parameters(), lr=vf_lr)

    def get_action(self, state, is_discrete):
        """Get action from the policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            mean, log_std, _ = self.actor_critic(state)
            
            if is_discrete:
                # For discrete action spaces, use softmax to get action probabilities
                probs = torch.softmax(mean, dim=-1)
                if self.deterministic:
                    # Take the action with highest probability
                    action = torch.argmax(probs).item()
                else:
                    # Sample from the categorical distribution
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()
                return action
            else:
                # For continuous action spaces
                std = torch.exp(log_std)
                if self.deterministic:
                    action = mean
                else:
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                return action.cpu().numpy()

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation."""
        # Convert to torch tensors
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae_lam

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns, is_discrete):
        """Update policy and value function."""
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        if is_discrete:
            actions = torch.LongTensor(actions).to(device)
        else:
            actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy update
        for _ in range(self.train_pi_iters):
            mean, log_std, _ = self.actor_critic(states)
            if is_discrete:
                probs = torch.softmax(mean, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            # Add entropy bonus to encourage exploration
            pi_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy)

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            # Clip policy gradients
            # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.pi_optimizer.step()

        # Value function update
        for _ in range(self.train_v_iters):
            _, _, values = self.actor_critic(states)
            vf_loss = nn.MSELoss()(values.squeeze(), returns)

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            # Clip value function gradients
            # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.vf_optimizer.step()

        return pi_loss.item(), vf_loss.item()

    def normalize_reward(self, rewards):
        """Normalize rewards using running mean and std."""
        if self.normalize_rewards:
            self.ret_rms.update(rewards)
            rewards = rewards / (self.ret_rms.std + 1e-8)
            rewards = np.clip(rewards, -self.cliprew, self.cliprew)
        return rewards

    def train(self, env, reward_shaping=None, eval_interval=1000, eval_episodes=5, render=False):
        """
        Train the PPO agent.
        
        Args:
            env: Training environment
            reward_shaping: Optional reward shaping function
            eval_interval: Number of time steps between evaluations (default: 1000)
            eval_episodes: Number of episodes to evaluate
        """
        episode_rewards = []
        eval_rewards = []
        best_mean_reward = float('-inf')
        total_steps = 0
        next_eval_step = eval_interval
        
        # Check if environment has discrete action space
        is_discrete = isinstance(env.action_space, Discrete)
        print(f"Action space is {'discrete' if is_discrete else 'continuous'}")
        
        for epoch in range(self.n_epochs):
            state = env.reset()
            done = False
            ep_len = 0
            episode_reward = 0

            # Collect experience
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

            for step in range(self.steps_per_epoch):
                # Get action
                action = self.get_action(state, is_discrete)
                
                # Get value estimate
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    mean, log_std, value = self.actor_critic(state_tensor)
                    if is_discrete:
                        probs = torch.softmax(mean, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        log_prob = dist.log_prob(torch.tensor(action).to(device))
                    else:
                        std = torch.exp(log_std)
                        dist = torch.distributions.Normal(mean, std)
                        log_prob = dist.log_prob(torch.FloatTensor(action).to(device)).sum()

                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                reward_ = reward
                if reward_shaping is not None:
                    reward += reward_shaping(next_state)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)

                state = next_state
                episode_reward += reward_
                ep_len += 1
                total_steps += 1
                
                # Check if it's time for evaluation
                if total_steps >= next_eval_step:
                    mean_reward, std_reward, _ = self.evaluate(env, n_episodes=eval_episodes, render=render)
                    eval_rewards.append(mean_reward)
                    if render:
                        print(f"\nEvaluation at {total_steps} steps:")
                        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    
                        if mean_reward > best_mean_reward:
                            best_mean_reward = mean_reward
                            print("New best model!")
                    
                    next_eval_step = total_steps + eval_interval
                
                if done or ep_len == self.max_ep_len:
                    episode_rewards.append(episode_reward)
                    state = env.reset()
                    done = False
                    ep_len = 0
                    episode_reward = 0

            # Get final value estimate
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                _, _, next_value = self.actor_critic(state_tensor)
                next_value = next_value.item()

            # Normalize rewards
            rewards = self.normalize_reward(np.array(rewards))

            # Compute advantages and returns
            advantages, returns = self.compute_gae(
                rewards,
                np.array(values),
                next_value,
                np.array(dones)
            )

            # Update policy and value function
            pi_loss, vf_loss = self.update(
                np.array(states),
                np.array(actions),
                np.array(log_probs),
                advantages,
                returns,
                is_discrete
            )

            # Print training statistics
            print(f"\nEpoch {epoch+1}/{self.n_epochs}")
            print(f"Total Steps: {total_steps}")
            print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
            print(f"Policy Loss: {pi_loss:.4f}")
            print(f"Value Loss: {vf_loss:.4f}")
            print("------------------------")
            episode_rewards = []
            
        return eval_rewards

    def evaluate(self, env, n_episodes=10, render=False):
        """
        Evaluate the trained agent's performance over multiple episodes.
        
        Args:
            env: The environment to evaluate in
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            mean_reward: Average reward over all episodes
            std_reward: Standard deviation of rewards
            episode_lengths: List of episode lengths
        """
        episode_rewards = []
        episode_lengths = []
        is_discrete = isinstance(env.action_space, Discrete)
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < self.max_ep_len:
                if render:
                    env.render()
                    
                # Get action from policy
                action = self.get_action(state, is_discrete)
                if not is_discrete:
                    action = np.clip(action, -1, 1)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if render:
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Episode Reward: {episode_reward:.2f}")
                print(f"Episode Length: {episode_length}")
                print("------------------------")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward, episode_lengths

class RunningMeanStd:
    """Running mean and standard deviation calculator."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class NormalizedEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.obs_mean = np.zeros_like(env.observation_space.low)
        self.obs_std = np.ones_like(env.observation_space.low)
        self.epsilon = 1e-8  # To prevent division by zero

    def reset(self):
        obs = self.env.reset()
        return self.normalize(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.normalize(obs), reward, done, info

    def normalize(self, obs):
        # Assuming obs is a numpy array
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + self.epsilon)
        return normalized_obs


# def reward_shaping(s_):
#     r = 0.0

#     if s_[0] > 0.4:
#         r += 5.0 * (s_[0] + 0.4)
#     if s_[0] > -0.1:
#         r += 100.0 * s_[0]
#     if s_[0] < 0.7:
#         r += 5.0 * (-0.7 - s_[0])
#     if s_[0] < -0.3 and np.abs(s_[1]) > -0.02:
#         r += 4000.0 * (np.abs(s_[1]) - 0.02)

#     return r


# def reward_shaping(s_):
#     r = 0.0
#     position, velocity = s_[0], s_[1]

#     # Encourage forward progress
#     r += 10 * (position + 0.5)  # Normalize: position ∈ [-1.2, 0.6] → [0, ~17]

#     # Encourage higher velocity (to build momentum)
#     r += 5 * abs(velocity)


#     return r

