import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yaml, os, pickle, pdb

def get_activation(name):
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU
    }.get(name.lower(), nn.ReLU)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def size(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layers, activation="relu"):
        super().__init__()
        act_fn = get_activation(activation)
        layers = []

        last_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_fn())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))  # constrain to action bounds



# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation="relu"):
        super().__init__()
        act_fn = get_activation(activation)

        def build_q_net():
            layers = []
            last_dim = state_dim + action_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(act_fn())
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q_net()
        self.q2 = build_q_net()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)

        

class OUActionNoise:
    def __init__(self, mean, std_dev=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # Ornstein-Uhlenbeck formula
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)


# TD3 Agent
class TD3:
    def __init__(self, 
                state_dim, action_dim, max_action,
                actor_lr=3e-4, actor_hidden_layers=[400, 300], actor_activation="relu",
                critic_lr=3e-4, critic_hidden_layers=[400, 300], critic_activation="relu",
                replay_buffer_size=1_000_000):
        self.actor = Actor(state_dim, action_dim, max_action, actor_hidden_layers, actor_activation).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, actor_hidden_layers, actor_activation).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.max_action = max_action

        self.critic = Critic(state_dim, action_dim, critic_hidden_layers, critic_activation).to(device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden_layers, critic_activation).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=100, initial_buffers_size=10_000, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if self.replay_buffer.size() < initial_buffers_size:
            return

        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def load_td3_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten config for easier access
    actor_cfg = config['actor']
    critic_cfg = config['critic']
    target_cfg = config['target_network']
    buffer_cfg = config['replay_buffer']

    td3_params = {
        "env_name": config["env_name"],
        "seed": config["seed"], # TODO: set seed
        "total_steps": config["total_training_steps"],
        "start_timesteps": config["start_timesteps"],
        "batch_size": buffer_cfg["batch_size"],
        "replay_buffer_size": buffer_cfg["size"],
        "actor_lr": actor_cfg["learning_rate"],
        "actor_hidden_layers": actor_cfg["hidden_layers"], 
        "actor_activation": actor_cfg["activation"],
        "critic_lr": critic_cfg["learning_rate"],
        "critic_hidden_layers": critic_cfg["hidden_layers"],
        "critic_activation": critic_cfg["activation"],
        "tau": target_cfg["tau"],
        "policy_noise": target_cfg["policy_noise"],
        "noise_clip": target_cfg["noise_clip"],
        "exploration_noise": actor_cfg["noise_std"],
        "noise_type": actor_cfg["noise_type"],
        "gamma": config["discount_factor"],
        "policy_delay": config["policy_delay"],
    }

    return td3_params


def train_once(env_name, time_steps, 
             actor_lr=3e-4, actor_hidden_layers=[400, 300], actor_activation="relu",
             critic_lr=3e-4, critic_hidden_layers=[400, 300], critic_activation="relu",
             replay_buffer_size=1_000_000,
             exp_noise=0.1, noise_type="normal",
             batch_size=100, initial_buffers_size=10_000, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    # Create environment
    env = gym.make(env_name) 

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) 

    agent = TD3(state_dim, action_dim, max_action,
                actor_lr, actor_hidden_layers, actor_activation,
                critic_lr, critic_hidden_layers, critic_activation,
                replay_buffer_size)
    
    ou_noise = OUActionNoise(mean=np.zeros(action_dim), std_dev=exp_noise)

    episode_rewards = []
    record_rewards = np.zeros(int(time_steps/1000))

    for episode in range(10000):
        state = env.reset()[0]
        ou_noise.reset()
        total_reward = 0
        for t in range(1000):
            action = agent.select_action(state)
            if noise_type == "ou":
                noise = ou_noise()
                action = (action + noise).clip(-max_action, max_action)
            elif noise_type == "normal":
                action = (action + np.random.normal(0, exp_noise, size=action_dim)).clip(-max_action, max_action)
            else: assert False
            next_state, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward

            agent.train(batch_size, initial_buffers_size, gamma, tau, policy_noise, noise_clip, policy_freq)

            if (agent.total_it % 1000 == 0) and (agent.total_it/1000 < len(record_rewards)):
                rw_avg_step = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.
                record_rewards[int(agent.total_it/1000)] = rw_avg_step
                if agent.total_it > 0:
                    print(f"total_it: {agent.total_it}, rw_avg_step: {rw_avg_step}")
            
            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward}")

        if agent.total_it >= time_steps:
            break
    return record_rewards, episode_rewards


def plot(episode_rewards_all_reps, line_label, save_folder):
    T = np.min([len(x) for x in episode_rewards_all_reps])
    returns = np.array([_[:T] for _ in episode_rewards_all_reps]) # [N_rep, N_episode]
    mean_returns = np.mean(returns, axis=0)  # average over seeds
    std_returns = np.std(returns, axis=0)    # std over seeds
    # episodes = np.arange(0,1000*returns.shape[1],1000)
    episodes = np.arange(returns.shape[1])

    if line_label is None:
        line_label = 'Mean Return'

    if save_folder is not None:
        plt.figure(figsize=(3.5, 3))
    # plt.subplot(2, 1, 1)
    plt.plot(episodes, mean_returns, label=line_label)
    plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.3)
    plt.xlabel("Episode")
    # plt.xlabel("Time steps")
    plt.ylabel("Return")
    # plt.title(f"Training Performance: Mean ± Std across Seeds ({len(episode_rewards_all_reps)} seeds)")
    plt.grid(True)
    # plt.subplot(2, 1, 2)
    # plt.plot(episodes, mean_returns, label=line_label)
    # plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.3)
    # plt.ylim(-200, 0)
    # plt.xlabel("Episode")
    # plt.ylabel("Return")
    # plt.title(f"Training Performance: Mean ± Std across Seeds ({len(episode_rewards_all_reps)} seeds)")
    # plt.grid(True)
    plt.tight_layout()
    
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, f"TD3_{td3_params['env_name']}.png"))
        plt.close()


# Pendulum-v1
search_space = {
    "actor_lr": [3e-4, 3e-3, 3e-5, 3e-2],
    "critic_lr": [3e-4, 3e-3, 3e-5, 3e-2],
    "batch_size": [100, 1000, 10],
    "policy_noise": [0.2, 0.02, 0.5],
    "noise_clip": [0.5, 0.1, 1.0, 0.],
    "tau": [0.005, 0.05, 0.001, 0.5],
    "policy_delay": [2, 1, 5, 50],
}

# # MountainCarContinuous-v0
# search_space = {
#     "batch_size": [1000, 100, 256, ], #500, ],
#     "start_timesteps": [25_000, 1_000],
#     "noise_type": ["ou", "normal", ],
#     "exploration_noise": [0.5, 0.1, 0.01],
#     "policy_noise": [0.3, 0.1, 0.5, 0.01],
#     "tau": [0.005, 0.05, 0.001, 0.5],  
#     "policy_delay": [2, 1, 5, 50],
# }

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "Pendulum-v1"
# env_name = "MountainCarContinuous-v0"

# start sweeping
for key in search_space.keys():
    base_dir = f"./TD3_{env_name}_results/{key}"

    vs = search_space[key]
    episode_rewards_all_setups = {}
    for i, v in enumerate(vs):
        # Unique trial folder
        trial_name = f"trial_{i:03d}"
        trial_path = os.path.join(base_dir, trial_name)
        os.makedirs(trial_path, exist_ok=True)


        # Add constant settings
        td3_params = load_td3_config(f"td3_{env_name}_config.yaml")
        td3_params.update({key: v})
        print(td3_params)

        result_file = os.path.join(trial_path, "results.pkl")
        config_file = os.path.join(trial_path, "config.yaml")
        # Check if the result file already exists
        # If it exists, we can skip the training
        # and just load the results
        if os.path.exists(os.path.join(trial_path, "results.npy")):
            episode_rewards_all_reps = np.load(os.path.join(trial_path, "results.npy"), allow_pickle=True)
            with open(result_file, 'wb') as f:
                pickle.dump(episode_rewards_all_reps, f)
        if os.path.exists(result_file):
            # check whether the config is what we want
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            if config[key] != v:
                print(f"Warning: config mismatch for {trial_name}. Expected {key}: {v}, but found {config[key]}")
                import pdb; pdb.set_trace()
            else:
                # episode_rewards_all_reps = np.load(result_file, allow_pickle=True)
                with open(result_file, 'rb') as f:
                    episode_rewards_all_reps = pickle.load(f)
                episode_rewards_all_setups[v] = episode_rewards_all_reps
                continue
        else:
            # Save YAML config
            with open(config_file, 'w') as f:
                yaml.dump(td3_params, f)

            # Train TD3 agent
            episode_rewards_all_reps = []
            for repi in range(3):
                record_rewards, episode_rewards = train_once(td3_params['env_name'], td3_params['total_steps'],
                                            actor_lr=td3_params['actor_lr'], 
                                            actor_hidden_layers=td3_params['actor_hidden_layers'],
                                            actor_activation=td3_params['actor_activation'],
                                            critic_lr=td3_params['critic_lr'], 
                                            critic_hidden_layers=td3_params['critic_hidden_layers'],
                                            critic_activation=td3_params['critic_activation'],
                                            replay_buffer_size=td3_params['replay_buffer_size'],
                                            exp_noise=td3_params['exploration_noise'],
                                            noise_type=td3_params['noise_type'],
                                            batch_size=td3_params['batch_size'], 
                                            initial_buffers_size=td3_params['start_timesteps'],
                                            gamma=td3_params['gamma'], 
                                            tau=td3_params['tau'], 
                                            policy_noise=td3_params['policy_noise'], 
                                            noise_clip=td3_params['noise_clip'], 
                                            policy_freq=td3_params['policy_delay'])
                episode_rewards_all_reps.append(record_rewards)

                episode_rewards_all_setups[v] = episode_rewards_all_reps
                with open(result_file, 'wb') as f:
                    pickle.dump(episode_rewards_all_reps, f)
                # with open(os.path.join(trial_path, "results.npy"), "wb") as f:
                #     np.save(f, episode_rewards_all_reps)

                plot(episode_rewards_all_reps, None, trial_path)

    plt.figure(figsize=(3.5, 3))
    sorted_dict = dict(sorted(episode_rewards_all_setups.items(), key=lambda item: item[0]))
    for v, episode_rewards_all_reps in sorted_dict.items():
        plot(episode_rewards_all_reps, line_label=f"{v}", save_folder=None)
    # for i in range(2):
    #     plt.subplot(2, 1, i+1)
    plt.legend() #title=key)
    plt.suptitle(f"{key}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"TD3_{key}_sweep.pdf"))    