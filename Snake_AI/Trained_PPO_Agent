import gymnasium as gym
from Snake_EnvAndAgent import SnakeGameEnv
from PPO_Model import PPOAgent
import os
import time
import numpy as np
from plot_utility_Trained_Agent import init_live_plot, update_live_plot, save_live_plot_final, smooth_curve

PLAY_CONFIG = {
    'grid_size': 30,
    'model_path_prefix': 'snake_ppo_models/ppo_snake_final',
    'num_episodes_to_play': 100,
    'render_fps': 10,
    'live_plot_interval_episodes': 1,
    'plot_smoothing_factor': 0.8
}

PLOT_SAVE_DIR = 'snake_ppo_plots'
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


def play_agent():
    print("Initializing environment for playback...")
    env = SnakeGameEnv(render_mode='human')
    
    if PLAY_CONFIG['render_fps'] > 0:
        env.metadata["render_fps"] = PLAY_CONFIG['render_fps']

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = PPOAgent(
        observation_space_shape=obs_shape,
        action_space_size=action_size,
        actor_lr=3e-4, 
        critic_lr=3e-4,
        hidden_layer_sizes=[512, 512, 512]
    )

    print(f"loading models from: {PLAY_CONFIG['model_path_prefix']}")
    
    load_success = agent.load_models(PLAY_CONFIG['model_path_prefix'])
    
    if not load_success:
        print("\nFATAL ERROR: Failed to load trained models from disk. The agent CANNOT perform as trained. Exiting playback!.")
        env.close()
        return 

    print("--- Trained models loaded successfully. Loading playback. ---")

    print("Starting agent playback...")
    
    episode_rewards_playback = []
    
    fig, ax, line = init_live_plot(PLOT_SAVE_DIR, filename="live_playback_rewards_plot.png")
    ax.set_title('Live Playback Progress (Episode Rewards)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')

    for episode in range(PLAY_CONFIG['num_episodes_to_play']):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _, _ = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            done = terminated or truncated

        episode_rewards_playback.append(episode_reward)
        
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Score = {info['score']}, Steps = {steps}")
        
        if (episode + 1) % PLAY_CONFIG['live_plot_interval_episodes'] == 0:
            current_episodes = list(range(1, len(episode_rewards_playback) + 1))
            smoothed_rewards = smooth_curve(episode_rewards_playback, factor=PLAY_CONFIG['plot_smoothing_factor'])
            update_live_plot(fig, ax, line, current_episodes, smoothed_rewards, 
                             current_timestep=episode + 1,
                             total_timesteps=PLAY_CONFIG['num_episodes_to_play'])

        time.sleep(0.5)

    env.close()
    print("\nPlayback finished.")
    
    current_episodes = list(range(1, len(episode_rewards_playback) + 1))
    smoothed_rewards = smooth_curve(episode_rewards_playback, factor=PLAY_CONFIG['plot_smoothing_factor'])
    update_live_plot(fig, ax, line, current_episodes, smoothed_rewards, 
                     current_timestep=PLAY_CONFIG['num_episodes_to_play'], 
                     total_timesteps=PLAY_CONFIG['num_episodes_to_play'])
    save_live_plot_final(fig, ax)

    avg_playback_reward = np.mean(episode_rewards_playback)
    print(f"Average Reward over {PLAY_CONFIG['num_episodes_to_play']} playback episodes: {avg_playback_reward:.2f}")


if __name__ == "__main__":
    play_agent()
