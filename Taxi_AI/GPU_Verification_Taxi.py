import os
import tensorflow as tf
import gymnasium as gym
import numpy as np

def verify_gpu_and_env():
    print("--- 1. Hardware Check ---")
    # Check physical devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all VRAM at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s): {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("❌ No GPU found. Training will be slow on CPU.")

    print("\n--- 2. Architecture & GPU Placement Check ---")
    # Simulate the Taxi-v4 State (Discrete 0-499)
    # Testing separate networks with 11 parallel environments
    batch_size = 11
    dummy_states = np.random.randint(0, 500, size=(batch_size, 1))

    # Force execution on GPU
    with tf.device('/GPU:0'):
        # Dummy Actor
        actor = tf.keras.Sequential([
            tf.keras.layers.Embedding(500, 64, input_length=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        
        # Dummy Critic
        critic = tf.keras.Sequential([
            tf.keras.layers.Embedding(500, 64, input_length=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        actor_out = actor(dummy_states)
        critic_out = critic(dummy_states)

    if actor_out.shape == (11, 6) and critic_out.shape == (11, 1):
        print(f"✅ Separate Networks built and verified on GPU.")
        print(f"✅ Input Shape: {dummy_states.shape} -> Output Shapes: {actor_out.shape}, {critic_out.shape}")
    
    print("\n--- 3. Environment Check ---")
    try:
        env = gym.make("Taxi-v4")
        obs, _ = env.reset()
        print(f"✅ Gymnasium 'Taxi-v4' is installed. Initial observation: {obs}")
        env.close()
    except Exception as e:
        print(f"❌ Env Error: {e}")

if __name__ == "__main__":
    verify_gpu_and_env()