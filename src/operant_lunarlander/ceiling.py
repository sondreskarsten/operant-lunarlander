def train_ppo_ceiling(total_timesteps=400000, n_eval=100, seed=0):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    env = gym.make("LunarLander-v3", continuous=False)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    mean, std = evaluate_policy(model, env, n_eval_episodes=n_eval)
    return {"mean": float(mean), "std": float(std), "total_timesteps": total_timesteps}


if __name__ == "__main__":
    print(train_ppo_ceiling())
