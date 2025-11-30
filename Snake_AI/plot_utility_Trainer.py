import matplotlib.pyplot as plt
import numpy as np
import os
import time

def smooth_curve(points, factor=0.9):

    smoothed_points = []
    if points:
        smoothed_points.append(points[0])
        for i in range(1, len(points)):
            smoothed_points.append(smoothed_points[-1] * factor + points[i] * (1 - factor))
    return smoothed_points

def plot_rewards(rewards_history, log_interval, save_dir, filename="rewards_plot.png", show_plot=True):
   
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    episodes = [i * log_interval for i in range(1, len(rewards_history) + 1)]
    plt.plot(episodes, rewards_history, label='Average Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('PPO Training Progress (Average Reward per Episode)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {os.path.abspath(save_path)}")
    
    if show_plot:
        plt.show()

def init_live_plot(save_dir, filename="live_rewards_plot.png"):

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], label='Smoothed Average Reward')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.set_title('Live PPO Training Progress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    ax._save_path_final = os.path.join(save_dir, filename) 
    
    return fig, ax, line

def update_live_plot(fig, ax, line, episodes, smoothed_rewards, current_timestep=None, total_timesteps=None):
    """
    Updates the live plot with new data.
    """
    if not episodes or not smoothed_rewards:
        return

    line.set_data(episodes, smoothed_rewards)
    
    ax.set_xlim(0, max(episodes) * 1.05 if episodes else 1) 
    
    min_y = min(smoothed_rewards) * 0.9 if smoothed_rewards else -1
    max_y = max(smoothed_rewards) * 1.1 if smoothed_rewards else 1

    if abs(max_y - min_y) < 0.1: 
        min_y -= 0.05
        max_y += 0.05
    ax.set_ylim(min_y, max_y)

    if current_timestep is not None and total_timesteps is not None:
        ax.set_title(f'Live PPO Training Progress (Timestep: {current_timestep:,}/{total_timesteps:,})')

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

def save_live_plot_final(fig, ax):
    
    plt.ioff()
    save_path = getattr(ax, '_save_path_final', None)
    if save_path:
        plt.savefig(save_path)
        print(f"Final live plot saved to: {os.path.abspath(save_path)}")
    plt.close(fig)
    plt.show()
