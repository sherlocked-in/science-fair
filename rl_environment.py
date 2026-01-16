import gymnasium as gym
import numpy as np
from physics_engine import survival_prediction, bbb_diffusion_efficiency

class GBMNPEvironment(gym.Env):
    """Reinforcement Learning for NP Design"""
    
    def __init__(self):
        super().__init__()
        # Actions: [size_nm, peg_density, charge]
        self.action_space = gym.spaces.Box(low=np.array([10, 0.0, -1]), 
                                         high=np.array([100, 1.0, 1]), shape=(3,))
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(4,))
        self.current_step = 0
        self.max_steps = 50
        
    def reset(self, seed=None):
        self.current_step = 0
        obs = np.array([50, 0.1, 1, 0.1])  # [size, peg, charge, hypoxia]
        return obs, {}
    
    def step(self, action):
        size, peg, charge = action
        ph = 6.5
        hypoxia = 0.1
        
        days, reward = survival_prediction(size, charge, fus=True, 
                                         tumor_ph=ph, hypoxia=hypoxia)
        
        # Penalty for excess PEG (Nance 2014 opsonization)
        peg_penalty = abs(peg - 0.1) * 2.0
        
        final_reward = reward - peg_penalty
        self.current_step += 1
        
        obs = np.array([size, peg, charge, hypoxia])
        terminated = self.current_step >= self.max_steps
        return obs, final_reward, terminated, False, {}
