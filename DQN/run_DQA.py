import gymnasium as gym
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def make_env(env_id):
    env = gym.make(env_id)
    env = Monitor(env)  # to log reward
    return env

def run_experiment(env_id="CartPole-v1", total_timesteps=100_000, n_seeds=3, save_path="./logs_dqn/"):
    rewards_summary = []
    all_episode_rewards = []
    os.makedirs(save_path, exist_ok=True)

    for seed in range(n_seeds):
        print(f"Training with seed {seed}")

        # 设置 Monitor 以记录 episode reward
        def make_monitored_env():
            env = gym.make(env_id)
            return Monitor(env)

        vec_env = DummyVecEnv([make_monitored_env])
        model = DQN("MlpPolicy", vec_env, verbose=0, seed=seed)
        model.learn(total_timesteps=total_timesteps)

        # 提取 reward（每个 Monitor 环境有 .episode_rewards）
        rewards = vec_env.get_attr("episode_rewards")[0]
        all_episode_rewards.append(rewards)

        # Evaluate after training
        eval_env = gym.make(env_id)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        rewards_summary.append((seed, mean_reward, std_reward))

        # Save model
        model.save(f"{save_path}/dqn_seed{seed}")

    df_summary = pd.DataFrame(rewards_summary, columns=["Seed", "Mean Reward", "Std Reward"])
    return df_summary, all_episode_rewards

def plot_results(df, title="DQN Performance", save_dir="plots", algo_name="DQN", env_name="CartPole"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{algo_name}_{env_name}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(8, 5))
    plt.errorbar(df["Seed"], df["Mean Reward"], yerr=df["Std Reward"], fmt='-o', capsize=5)
    plt.title(title)
    plt.xlabel("Random Seed")
    plt.ylabel("Mean Evaluation Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()



def plot_learning_curve(reward_lists, save_path="plots/dqn_learning_curve.png", title="DQN Learning Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    for i, rewards in enumerate(reward_lists):
        plt.plot(rewards, label=f"Seed {i}")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    df, reward_lists = run_experiment(env_id="CartPole-v1", total_timesteps=10_000, n_seeds=3)
    plot_results(df)
    plot_learning_curve(reward_lists)