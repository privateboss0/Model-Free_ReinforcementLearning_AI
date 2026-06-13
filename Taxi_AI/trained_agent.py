import gymnasium as gym
import tensorflow as tf
import numpy as np
import time
import os
from agent import PPOAgent
from config import Config

def evaluate(num_episodes=5):
    cfg = Config()
    
    #MANUAL PATH CONFIGURATION

    actor_path = "ppo_taxi/checkpoints/keras_models/actor_step_14200.keras" 

    if not os.path.exists(actor_path):
        print(f"❌ ERROR: Cannot find file at {os.path.abspath(actor_path)}")
        print("Check your folder structure or the spelling of the filename.")
        return

    env = gym.make(cfg.ENV_NAME, render_mode="human")
    agent = PPOAgent(cfg)
    
    try:
        agent.actor = tf.keras.models.load_model(actor_path)
        print(f"✅ Successfully loaded: {actor_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"--- Starting Episode {ep + 1} ---")
        
        while not done:
  
            state_input = np.array([[state]])
            
  
            probs = agent.actor(state_input, training=False)
            
            action = np.argmax(probs[0])
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            time.sleep(0.05)
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1} Finished | Reward: {total_reward} | Steps: {steps}")
    
    print("\n" + "="*40)
    print(f"EVALUATION COMPLETE")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Best Run:       {np.max(episode_rewards)}")
    print("="*40)
    
    env.close()

if __name__ == "__main__":
    evaluate(num_episodes=10)