import gymnasium as gym
import numpy as np
import tensorflow as tf
import os
import time
from config import ENV_ID, SEED, SAVE_PATH, TOTAL_TIMESTEPS
from agent import PPOAgent
from reward_shaping import LunarLanderRewardShaping 

CHECKPOINT_SUBDIR = 'tf_checkpoints'
CHECKPOINT_ROOT = os.path.join(SAVE_PATH, CHECKPOINT_SUBDIR)

tf.config.run_functions_eagerly(True)

def run_trained_agent(episodes=10, target_step=None):
   
    print(f"--- Running Trained Agent on {ENV_ID} with Human Rendering ---")
    print(f"Checking for checkpoints in: {CHECKPOINT_ROOT}")

    try:
        env = gym.make(ENV_ID, render_mode="human")
        env = LunarLanderRewardShaping(env) 
    except Exception as e:
        print(f"ERROR: Could not create environment {ENV_ID} or apply wrapper. Details: {e}")
        return
        
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
     
    current_obs, info = env.reset(seed=SEED)

    agent = PPOAgent(obs_shape, action_size, TOTAL_TIMESTEPS) 

    checkpoint_to_load = None
        
    if target_step is not None:
        checkpoint_name_prefix = f'ckpt-{target_step}'
        potential_path = os.path.join(CHECKPOINT_ROOT, checkpoint_name_prefix) 
        
        if os.path.exists(f'{potential_path}.index'):
            checkpoint_to_load = potential_path
            print(f"\nAttempting to load specific checkpoint: T={target_step}")
        else:
            print(f"\nERROR: Specified checkpoint prefix '{checkpoint_name_prefix}' was not found in '{CHECKPOINT_ROOT}'.")
            print("Falling back to the latest available checkpoint.")
            checkpoint_to_load = agent.checkpoint_manager.latest_checkpoint
                
    else:
        checkpoint_to_load = agent.checkpoint_manager.latest_checkpoint

    if not checkpoint_to_load:
        print("\nERROR: Could not find any checkpoint in the designated save path.")
        env.close()
        return

    try:
        agent.checkpoint.restore(checkpoint_to_load).expect_partial()
        
        agent.obs_rms.mean = agent.rms_mean_var.numpy()
        agent.obs_rms.var = agent.rms_var_var.numpy()
        agent.obs_rms.count = agent.rms_count_var.numpy()
        
        checkpoint_name = os.path.basename(checkpoint_to_load)
        if 'ckpt-' in checkpoint_name:
            loaded_timesteps = int(checkpoint_name.split('-')[-1])
        else:
            loaded_timesteps = 0 

        print(f"\nSuccessfully loaded checkpoint trained to T={loaded_timesteps}")
    except Exception as e:
        print(f"\nERROR: Failed to restore checkpoint at {checkpoint_to_load}. Details: {e}")
        print("Suggestion: Check the consistency of your environment setup (wrapper, action size, file names).")
        env.close()
        return

    print(f"\nStarting {episodes} playback episodes...")
    total_rewards = []
    for i in range(episodes):
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            env.render()

            obs_to_agent = current_obs.reshape(1, *obs_shape)

            actions, _, _ = agent.select_action(obs_to_agent)

            action_to_step = actions[0]
            current_obs, reward, terminated, truncated, info = env.step(action_to_step)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            time.sleep(0.01)

        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward:7.2f}, Steps = {step_count}")
        
        current_obs, info = env.reset()

    env.close()
    if total_rewards:
        print("-" * 30)
        print(f"Average Reward over {episodes} episodes: {np.mean(total_rewards):7.2f}")
        print("-" * 30)

if __name__ == "__main__":

    run_trained_agent(episodes=15, target_step=None)
