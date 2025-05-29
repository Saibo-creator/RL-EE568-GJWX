import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

# --------------------
# Configuration
# --------------------
ENV_NAME = "CartPole-v1" # "MountainCar-v0" # or 
TIMESTEPS = 100_000
SEEDS = [0, 1, 2]
RESULTS_DIR = "./results_dqn"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------
# Ablation Settings
# --------------------
ABLATION_MODES = {
    # "target_update": [500, 1000, 5000],
    "exploration_fraction": [0.1, 0.3, 0.5],
    # "buffer_size": [1000, 5000, 20000]
}


def make_env(seed):
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_dqn(config_name, seed, param_key, param_value):
    env = make_env(seed)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=seed,
        tensorboard_log=f"{RESULTS_DIR}/tensorboard/",
        **{param_key: param_value}
    )
    
    rewards = []
    eval_steps = []
    
    # Custom training loop with intermediate evaluations
    for step in range(0, TIMESTEPS + 1, 1000):
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        mean_r, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=5, deterministic=True)
        rewards.append(mean_r)
        eval_steps.append(step)

    # Save reward curve
    np.savez(f"{RESULTS_DIR}/{config_name}_seed{seed}_curve.npz",
             steps=np.array(eval_steps),
             rewards=np.array(rewards))
    
    model.save(f"{RESULTS_DIR}/{config_name}_seed{seed}")
    return model


def plot_reward_curves(param_key, values, title, save_path):
    plt.figure(figsize=(7, 4))
    
    for val in values:
        all_curves = []
        for seed in SEEDS:
            path = f"{RESULTS_DIR}/{param_key}_{val}_seed{seed}_curve.npz"
            if not os.path.exists(path):
                continue
            data = np.load(path)
            all_curves.append(data["rewards"])
            steps = data["steps"]  # same for all
        
        if not all_curves:
            continue

        mean_rewards = np.mean(all_curves, axis=0)
        std_rewards = np.std(all_curves, axis=0)

        plt.plot(steps, mean_rewards, label=f"{param_key} = {val}")
        plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(model, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes, deterministic=True)
    return mean_reward


def run_ablation(mode):
    param_key = {
        "target_update": "target_update_interval",
        "exploration_fraction": "exploration_fraction",
        "buffer_size": "buffer_size"
    }[mode]

    all_results = {}

    for value in ABLATION_MODES[mode]:
        config_name = f"{mode}_{value}"
        print(f"Running {config_name}...")
        rewards = []
        for seed in SEEDS:
            model = train_dqn(config_name, seed, param_key, value)
            reward = evaluate(model)
            rewards.append(reward)
        all_results[str(value)] = rewards
    return all_results


def plot_results(results, title, x_label, save_path):
    labels = results.keys()
    avg = [np.mean(results[k]) for k in labels]
    std = [np.std(results[k]) for k in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, avg, yerr=std, capsize=5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Average Reward (10 episodes)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    for mode in ABLATION_MODES.keys():
        results = run_ablation(mode)
        
        # Plot final bar comparison (existing)
        plot_results(
            results,
            title=f"Ablation on {mode} ({ENV_NAME})",
            x_label=mode,
            save_path=f"{RESULTS_DIR}/{ENV_NAME}_{mode}_bar.png"
        )

        # Plot reward curves (new!)
        plot_reward_curves(
            param_key=mode,
            values=ABLATION_MODES[mode],
            title=f"Reward Curves for {mode} Ablation ({ENV_NAME})",
            save_path=f"{RESULTS_DIR}/{ENV_NAME}_{mode}_curves.png"
        )