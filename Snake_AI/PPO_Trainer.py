import gymnasium as gym
from Snake_EnvAndAgent import SnakeGameEnv
from PPO_Model import PPOAgent
import numpy as np
import time
import os
import json
from plot_utility_Trainer import plot_rewards, smooth_curve, init_live_plot, update_live_plot, save_live_plot_final

HYPERPARAMETERS = {
    'grid_size': 30,
    'actor_lr': 0.0003,
    'critic_lr': 0.0003,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'num_epochs_per_update': 10,
    'batch_size': 64,
    'num_steps_per_rollout': 2048,
    'total_timesteps': 7_000_000,
    'hidden_layer_sizes': [512, 512, 512],
    'save_interval_timesteps': 400000,
    'log_interval_episodes': 10,
    'render_training': False,
    'render_fps_limit': 10,
    'plot_smoothing_factor': 0.9,
    'live_plot_interval_episodes': 100
}

MODEL_SAVE_DIR = 'snake_ppo_models'
PLOT_SAVE_DIR = 'snake_ppo_plots'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
print(f"Model save directory created/checked: {os.path.abspath(MODEL_SAVE_DIR)}")
print(f"Plot save directory created/checked: {os.path.abspath(PLOT_SAVE_DIR)}")


def train_agent():
    print(f"Current working directory: {os.getcwd()}")
    print("Initializing environment and agent...")

    render_mode = 'human' if HYPERPARAMETERS['render_training'] else None
    env = SnakeGameEnv(render_mode=render_mode)
    
    if HYPERPARAMETERS['render_training'] and HYPERPARAMETERS['render_fps_limit'] > 0:
        env.metadata["render_fps"] = HYPERPARAMETERS['render_fps_limit']

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = PPOAgent(
        observation_space_shape=obs_shape,
        action_space_size=action_size,
        actor_lr=HYPERPARAMETERS['actor_lr'],
        critic_lr=HYPERPARAMETERS['critic_lr'],
        gamma=HYPERPARAMETERS['gamma'],
        gae_lambda=HYPERPARAMETERS['gae_lambda'],
        clip_epsilon=HYPERPARAMETERS['clip_epsilon'],
        num_epochs_per_update=HYPERPARAMETERS['num_epochs_per_update'],
        batch_size=HYPERPARAMETERS['batch_size'],
        hidden_layer_sizes=HYPERPARAMETERS['hidden_layer_sizes']
    )

    total_timesteps_trained = 0
    episode_count = 0
    all_episode_rewards = []
    plot_rewards_history = []
    last_saved_timesteps = 0 

    print("Starting training...")
    start_time = time.time()

    fig, ax, line = init_live_plot(PLOT_SAVE_DIR, filename="live_ppo_training_progress.png")

    while total_timesteps_trained < HYPERPARAMETERS['total_timesteps']:
        current_rollout_steps = 0
        
        while current_rollout_steps < HYPERPARAMETERS['num_steps_per_rollout'] and \
              total_timesteps_trained + current_rollout_steps < HYPERPARAMETERS['total_timesteps']:
            
            state, info = env.reset() 
            done = False
            current_episode_reward = 0
            steps_in_episode = 0

            while not done and current_rollout_steps < HYPERPARAMETERS['num_steps_per_rollout'] and \
                  total_timesteps_trained + current_rollout_steps < HYPERPARAMETERS['total_timesteps']:
                
                action, log_prob, value = agent.choose_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                current_episode_reward += reward
                
                agent.remember(state, action, reward, next_state, terminated, log_prob, value)

                state = next_state
                steps_in_episode += 1
                current_rollout_steps += 1 
                done = terminated or truncated

                if HYPERPARAMETERS['render_training'] and HYPERPARAMETERS['render_fps_limit'] == 0:
                    time.sleep(0.01)

                if done: 
                    episode_count += 1
                    all_episode_rewards.append(current_episode_reward)

                    if episode_count % HYPERPARAMETERS['log_interval_episodes'] == 0:
                        avg_reward_last_n_episodes = np.mean(all_episode_rewards[-HYPERPARAMETERS['log_interval_episodes']:]).round(2)
                        plot_rewards_history.append(avg_reward_last_n_episodes)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Timestep: {total_timesteps_trained + current_rollout_steps}/{HYPERPARAMETERS['total_timesteps']} | Episode: {episode_count} | Avg Reward (last {HYPERPARAMETERS['log_interval_episodes']}): {avg_reward_last_n_episodes} | Total Score (this ep): {info['score']} | Time: {elapsed_time:.2f}s")
                    
                    if episode_count % HYPERPARAMETERS['live_plot_interval_episodes'] == 0:
                        if len(plot_rewards_history) > 1:
                            episodes_for_plot = [i * HYPERPARAMETERS['log_interval_episodes'] for i in range(len(plot_rewards_history))]
                            smoothed_rewards = smooth_curve(plot_rewards_history, factor=HYPERPARAMETERS['plot_smoothing_factor'])
                            update_live_plot(fig, ax, line, episodes_for_plot, smoothed_rewards, 
                                             current_timestep=total_timesteps_trained + current_rollout_steps, 
                                             total_timesteps=HYPERPARAMETERS['total_timesteps'])

                    if HYPERPARAMETERS['render_training']:
                        time.sleep(0.5)
                    break 
            
        total_timesteps_trained += current_rollout_steps

        if len(agent.states) > 0:
            print(f"  --- Agent learning at Total Timestep {total_timesteps_trained} (collected {len(agent.states)} steps in rollout) ---")
            agent.learn()
        else:
            print(f"  --- No data collected in current rollout, skipping learning ---")

        if total_timesteps_trained >= HYPERPARAMETERS['save_interval_timesteps'] and \
           (total_timesteps_trained // HYPERPARAMETERS['save_interval_timesteps']) > \
           (last_saved_timesteps // HYPERPARAMETERS['save_interval_timesteps']):
            
            save_path_timesteps = (total_timesteps_trained // HYPERPARAMETERS['save_interval_timesteps']) * HYPERPARAMETERS['save_interval_timesteps']
            print(f"--- Triggering periodic save at calculated timestep: {save_path_timesteps} ---")
            agent.save_models(os.path.join(MODEL_SAVE_DIR, f"ppo_snake_{save_path_timesteps}"))
            last_saved_timesteps = save_path_timesteps

    print("\nTraining finished!")
    print(f"--- Triggering final save at total_timesteps: {total_timesteps_trained} ---")
    agent.save_models(os.path.join(MODEL_SAVE_DIR, "ppo_snake_final"))
    env.close()

    print("Generating final performance plot...")
    episodes_for_plot = [i * HYPERPARAMETERS['log_interval_episodes'] for i in range(len(plot_rewards_history))]
    smoothed_rewards = smooth_curve(plot_rewards_history, factor=HYPERPARAMETERS['plot_smoothing_factor'])
    update_live_plot(fig, ax, line, episodes_for_plot, smoothed_rewards, 
                     current_timestep=total_timesteps_trained, 
                     total_timesteps=HYPERPARAMETERS['total_timesteps'])
    save_live_plot_final(fig, ax)

    plot_rewards(smoothed_rewards, HYPERPARAMETERS['log_interval_episodes'], PLOT_SAVE_DIR, "ppo_training_progress_final.png", show_plot=False)

if __name__ == "__main__":
    train_agent()
