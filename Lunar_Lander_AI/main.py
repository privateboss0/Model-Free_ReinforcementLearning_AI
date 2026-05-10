import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from config import *
from utils import make_parallel_envs, save_resume_data, load_resume_data
from agent import PPOAgent

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    rewards = rewards.astype(np.float32)
    values = values.astype(np.float32)
    dones = dones.astype(np.float32)
    next_value = next_value.astype(np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lambda = np.zeros_like(rewards[0], dtype=np.float32)         
    
    values = np.concatenate([values, next_value[None, :]], axis=0)     
    
    for t in reversed(range(N_STEPS)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        
        last_gae_lambda = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lambda
        advantages[t] = last_gae_lambda
        
    returns = advantages + values[:-1]
    return advantages, returns

def train():
    print(f"--- Running on {DEVICE} ---")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_DIR)

    vec_env = make_parallel_envs(ENV_ID, NUM_ENVS, SEED)
    obs_shape = vec_env.single_observation_space.shape
    action_size = vec_env.single_action_space.n

    ACTION_DTYPE = vec_env.action_space.dtype         
    print(f"Action space DTYPE retrieved: {ACTION_DTYPE}. Forcing agent output to this type.")
    
    agent = PPOAgent(obs_shape, action_size, TOTAL_TIMESTEPS)

    resume_path = os.path.join(os.path.dirname(SAVE_PATH) or '.', RESUME_FILE)
    initial_timesteps, initial_episode = load_resume_data(resume_path)
    current_timesteps = initial_timesteps
    current_episode = initial_episode
    obs, info = vec_env.reset(seed=SEED)

    all_episode_returns = [] 
    
    if current_timesteps == 0:
        print("Adapting observation normalizer for stability...")
        initial_observations = []
        current_obs = obs
        for _ in range(RMS_WARMUP_STEPS // NUM_ENVS): 
            action_array = vec_env.action_space.sample() 
            actions_to_step = np.ascontiguousarray(action_array.reshape(NUM_ENVS).astype(ACTION_DTYPE))
            current_obs, _, _, _, _ = vec_env.step(actions_to_step)
            initial_observations.append(current_obs)
                    
        initial_observations = np.array(initial_observations).reshape(-1, obs_shape[0])
        agent.adapt_normalization(initial_observations)
        
        obs, info = vec_env.reset(seed=SEED)
        print("Normalizer adapted. Starting training.")
            
    elif initial_timesteps > 0:
        try:
            agent.load_model(SAVE_PATH, initial_timesteps)
            print(f"Resumed training from timestep: {initial_timesteps}")
        except Exception as e:
            print(f"Error loading model weights at {initial_timesteps}: {e}. Starting from scratch.")
            current_timesteps = 0
            current_episode = 0
            obs, info = vec_env.reset(seed=SEED)

    start_time = time.time()
        
    base_timesteps = current_timesteps
    
    while current_timesteps < TOTAL_TIMESTEPS:
        rollout_data = {'observations': [], 'actions': [], 'log_probs': [],
                        'rewards': [], 'values': [], 'dones': [], 'old_values': []}
                
        episode_returns_list = [] 
        
        for step in range(N_STEPS):
            actions, values, log_probs = agent.select_action(obs)
                        
            actions_to_step = np.ascontiguousarray(actions.reshape(NUM_ENVS).astype(ACTION_DTYPE))
                        
            new_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions_to_step)
            dones = np.logical_or(terminateds, truncateds)

            # --- DEBUG PRINT ---
            #if np.any(dones):
                #print(f"DEBUG STEP: Done signal received! Raw infos keys: {infos.keys()}")
                #if 'final_info' in infos:
                    #print(f"DEBUG STEP: final_info key IS present.")
                #elif 'episode' in infos:
                     #print(f"DEBUG STEP: 'final_info' key IS MISSING, but raw 'episode' key IS present. Attempting fallback extraction.")
                #else:
                    #print("DEBUG STEP: 'final_info' key IS MISSING from infos dictionary.")
            # --- END DEBUG PRINT ---
                        
            rollout_data['observations'].append(np.array(obs).copy())
            rollout_data['actions'].append(actions_to_step.copy()) 
            rollout_data['log_probs'].append(np.array(log_probs).copy())
            rollout_data['values'].append(np.array(values).copy())
            rollout_data['old_values'].append(np.array(values).copy())
            rollout_data['rewards'].append(np.array(rewards).copy())
            rollout_data['dones'].append(np.array(dones).copy())
                        
            obs = new_obs
                        
            log_step = base_timesteps + ((step + 1) * NUM_ENVS)
            
            if 'final_info' in infos:
                env_infos_to_process = infos['final_info']
            
            elif 'episode' in infos:
                env_infos_to_process = [{'episode': {'r': r, 'l': l}} 
                                         for r, l in zip(infos['episode']['r'], infos['episode']['l'])]
            else:
                env_infos_to_process = []
                
            for env_info in env_infos_to_process:
                if env_info is not None and isinstance(env_info, dict):
                    if 'episode' in env_info and isinstance(env_info['episode'], dict):
                        episode_stats = env_info['episode']

                        if 'r' in episode_stats and 'l' in episode_stats:
                            episode_return = float(episode_stats['r'])
                            episode_length = int(episode_stats['l'])

                            episode_returns_list.append(episode_return) 
                            all_episode_returns.append(episode_return) 
                            current_episode += 1
                            
                            with summary_writer.as_default():
                                tf.summary.scalar("rollout/episode_return_single", episode_return, step=log_step)
                                tf.summary.scalar("rollout/episode_length_single", episode_length, step=log_step)

        current_timesteps += N_STEPS * NUM_ENVS
        base_timesteps = current_timesteps 

        rollout_data = {k: np.asarray(v) for k, v in rollout_data.items()}
        _, next_values, _ = agent.select_action(obs)
        rollout_data['advantages'], rollout_data['returns'] = compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones'],
            next_values,
            GAMMA,
            GAE_LAMBDA
        )
        ppo_batch = {}
        for k in ['observations', 'actions', 'log_probs', 'returns', 'advantages', 'old_values']:
            if rollout_data[k].ndim > 2:
                ppo_batch[k] = rollout_data[k].reshape((-1,) + rollout_data[k].shape[2:])
            else:
                ppo_batch[k] = rollout_data[k].reshape((-1,))
                        
        policy_loss, value_loss, entropy = agent.learn(ppo_batch)

        fps = current_timesteps / (time.time() - start_time)
        with summary_writer.as_default():
            tf.summary.scalar("train/policy_loss", policy_loss, step=current_timesteps)
            tf.summary.scalar("train/value_loss", value_loss, step=current_timesteps)
            tf.summary.scalar("train/entropy", entropy, step=current_timesteps)
            tf.summary.scalar("perf/timesteps_per_second", fps, step=current_timesteps)
                        
            if episode_returns_list:
                avg_rollout_return = np.mean(episode_returns_list)
                tf.summary.scalar("rollout/rollout_average_return_rollout", avg_rollout_return, step=current_timesteps)
            tf.summary.scalar("train/entropy_coef", ENTROPY_COEF, step=current_timesteps)
                            
            summary_writer.flush()
                        
        if episode_returns_list:
            avg_return = np.mean(episode_returns_list)
            print(f"T: {current_timesteps:<10} | Avg Return: {avg_return:7.2f} | Loss: {policy_loss:7.4f} | FPS: {fps:4.0f}")
        else:
            print(f"T: {current_timesteps:<10} | No episodes completed this rollout. | Loss: {policy_loss:7.4f} | FPS: {fps:4.0f}")

        if (current_timesteps - N_STEPS * NUM_ENVS) % CHECKPOINT_FREQ == 0:
            
            print(f"--- DIAGNOSTIC: Total recorded episodes: {len(all_episode_returns)} ---")

            if all_episode_returns:
                overall_avg_return = np.mean(all_episode_returns[-200:]) 
                with summary_writer.as_default():
                    tf.summary.scalar("rollout/overall_average_return_checkpoint", overall_avg_return, step=current_timesteps)
                print(f"--- CHECKPOINT: Overall Avg Reward (Last 200): {overall_avg_return:.2f} ---")
            
            agent.save_model(SAVE_PATH, current_timesteps)
            save_resume_data(resume_path, current_timesteps, current_episode)
            print(f"Checkpoint saved at timestep: {current_timesteps}")

    print(f"\nTarget reached ({TOTAL_TIMESTEPS} timesteps). Saving final model.")
    agent.save_model(SAVE_PATH, current_timesteps)
    save_resume_data(resume_path, current_timesteps, current_episode) 

    vec_env.close()        
    summary_writer.close()        
    print("Training complete.")

if __name__ == "__main__":
    train()
