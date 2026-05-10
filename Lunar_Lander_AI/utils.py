import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import tensorflow as tf
import json
import os
import time
from reward_shaping import LunarLanderRewardShaping 
from config import *

def save_resume_data(filepath, timesteps, episodes):
    """
    Saves the current training state to a JSON file located at the specified filepath.
    """
    data = {
        "timesteps": timesteps,
        "episode_count": episodes,
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving resume data to {filepath}: {e}")

def load_resume_data(filepath):
    """
    Loads the last saved training state from the specified JSON file.
    Returns (timesteps, episode_count).
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            timesteps = data.get("timesteps", 0)
            episodes = data.get("episode_count", 0)
            return (timesteps, episodes)
        except Exception as e:
            print(f"Error loading resume data from {filepath}: {e}. Starting from scratch.")
    return (0, 0)

def make_env(env_id, seed, idx, **kwargs):
    """
    Creates a single environment instance with a unique seed and applies necessary wrappers.
    """
    def thunk():

        env = gym.make(env_id, **kwargs)
                
        env = LunarLanderRewardShaping(env) 
        
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        
        return env
    return thunk

def make_parallel_envs(env_id, num_envs, seed):
    """
    Creates multiple environments and wraps them in an AsyncVectorEnv.
    """
    env_fns = [make_env(env_id, seed, i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns)

def calculate_gae(rewards, values, terminated, truncated, next_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    """
    Calculates Generalized Advantage Estimation (GAE) and Returns (R) from rollout data.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lambda = 0
        
    for t in reversed(range(N_STEPS)):
        done_mask = 1.0 - (terminated[t] | truncated[t]).astype(np.float32)
        if t == N_STEPS - 1:
            next_non_terminal_value = next_value
            next_value_actual = values[t]
        else:
            next_non_terminal_value = values[t + 1]
            next_value_actual = values[t]
                
        delta = rewards[t] + gamma * next_non_terminal_value * done_mask - next_value_actual
                
        advantages[t] = delta + gamma * gae_lambda * done_mask * last_gae_lambda
        last_gae_lambda = advantages[t]
        
    returns = advantages + values
    return advantages, returns
