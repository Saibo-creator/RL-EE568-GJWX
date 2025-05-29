import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# --------------------
# Configuration
# --------------------
ENV_NAME = "CartPole-v1"
TIMESTEPS = 100_000
SEEDS = [0, 1, 2]
EVAL_FREQ = 1000
RESULTS_DIR = "./results_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

def make_env(seed):
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

def train_and_evaluate(algo_cls, algo_name):
    for seed in SEEDS:
        env = make_env(seed)
        model = algo_cls(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=seed,
            tensorboard_log=f"{RESULTS_DIR}/tensorboard/"
        )

        rewards = []
        steps = []

        for step in range(0, TIMESTEPS + 1, EVAL_FREQ):
            model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
            mean_r, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=5, deterministic=True)
            rewards.append(mean_r)
            steps.append(step)

        np.savez(f"{RESULTS_DIR}/{algo_name}_seed{seed}_curve.npz",
                 steps=np.array(steps),
                 rewards=np.array(rewards))

        model.save(f"{RESULTS_DIR}/{algo_name}_seed{seed}")

def plot_comparison():
    plt.figure(figsize=(8, 5))

    for algo in ["DQN", "PPO"]:
        all_curves = []
        for seed in SEEDS:
            path = f"{RESULTS_DIR}/{algo}_seed{seed}_curve.npz"
            data = np.load(path)
            all_curves.append(data["rewards"])
            steps = data["steps"]

        mean_rewards = np.mean(all_curves, axis=0)
        std_rewards = np.std(all_curves, axis=0)

        plt.plot(steps, mean_rewards, label=algo)
        plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.title(f"PPO vs DQN on {ENV_NAME}")
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ppo_vs_dqn_cartpole.png")
    plt.show()

if __name__ == "__main__":
    print("Training DQN...")
    train_and_evaluate(DQN, "DQN")

    print("Training PPO...")
    train_and_evaluate(PPO, "PPO")

    print("Plotting results...")
    plot_comparison()