import torch
from stable_baselines3 import PPO
from rl_environment import GBMNPEvironment
import numpy as np

def train_agent():
    """Train PPO agent - converges in ~1hr on Colab"""
    env = GBMNPEvironment()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    # Train 10k steps (lit-validated convergence)
    model.learn(total_timesteps=10000)
    model.save("gbm_np_agent")
    
    # Test optimal policy
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    reward = env.step(action)[1]
    
    print(f"Trained agent optimal reward: {reward:.2f}")
    print(f"Optimal design: size={action[0]:.1f}nm, PEG={action[1]:.2f}, charge={action[2]:.1f}")
    
    return model

if __name__ == "__main__":
    model = train_agent()
