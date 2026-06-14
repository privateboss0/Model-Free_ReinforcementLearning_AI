# sanity_check.py
import sys
import os
import time
import tensorflow as tf
import gymnasium as gym
import numpy as np

def run_sanity_check():
    print("=" * 60)
    print("🔍 SYSTEM SANITY CHECK: INITIALIZING HARDWARE & ENVIRONMENT")
    print("=" * 60)

    # 1. System & Python Specs
    print(f"\n🐍 Python Version: {sys.version.split()[0]}")
    print(f"📦 TensorFlow Version: {tf.__version__}")
    print(f"📦 Gymnasium Version: {gym.__version__}")

    # 2. Nvidia GPU Verification
    print("\n--- [1/3] Checking Nvidia GPU Availability ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Success! Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   🔹 GPU [{i}]: {gpu.name}")
            try:
                # Test setting memory growth to ensure the driver is responsive
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   🔹 Memory growth successfully configured for GPU [{i}].")
            except RuntimeError as e:
                print(f"   ⚠️ Memory growth warning: {e}")
        
        # Core CUDA Math Ops Test
        print("\n⚡ Running Quick CUDA Graph Execution Matrix Test...")
        try:
            start_time = time.time()
            with tf.device('/GPU:0'):
                a = tf.random.normal([4000, 4000])
                b = tf.random.normal([4000, 4000])
                c = tf.matmul(a, b)

                _ = c.numpy()
            duration = time.time() - start_time
            print(f"✅ CUDA Test Passed! 4000x4000 Matrix multiplication took {duration:.4f} seconds.")
        except Exception as e:
            print(f"❌ CUDA Matrix Ops Failed! Issue with drivers or cuDNN Toolkit: {e}")
    else:
        print("❌ CRITICAL: No Nvidia GPUs detected by TensorFlow.")
        print("   Ensure NVIDIA Web Drivers, CUDA Toolkit, and cuDNN match your TF version requirements.")

  # 3. Asynchronous Vector Environment Test
    print("\n--- [2/3] Checking Asynchronous Vectorized Gym Environments ---")
    num_test_envs = 15
    env_name = "MountainCarContinuous-v0"
    print(f"🔄 Spawning {num_test_envs} asynchronous parallel workers for '{env_name}'...")
    
    try:
        start_env_time = time.time()
        envs = gym.vector.AsyncVectorEnv([
            lambda: gym.make(env_name) for _ in range(num_test_envs)
        ])
        
        states, info = envs.reset()
        print(f"✅ Subprocesses spawned successfully. Initial Batched State Shape: {states.shape} (Expected: ({num_test_envs}, 2))")
        
        random_actions = np.array([envs.action_space.sample() for _ in range(num_test_envs)], dtype=np.float32)
        
        # Reshape to ensure it perfectly matches (NUM_ENVS, ACTION_DIM) -> (15, 1)
        random_actions = random_actions.reshape(num_test_envs, -1)
        
        next_states, rewards, terminated, truncated, infos = envs.step(random_actions)
        
        print(f"✅ Step test successful.")
        print(f"   🔹 Next States Shape: {next_states.shape}")
        print(f"   🔹 Rewards Vector Shape: {rewards.shape}")
        
        envs.close()
        print(f"🧹 Cleaned up worker processes safely. Total environment test took {time.time() - start_env_time:.2f}s.")
    except Exception as e:
        print(f"❌ Multiprocessing Environment Failure: {e}")

    # 4. TensorFlow Probability Verification
    print("\n--- [3/3] Checking TensorFlow Probability Installation ---")
    try:
        import tensorflow_probability as tfp
        probs_mean = tf.constant([[0.0]])
        probs_std = tf.constant([[1.0]])
        dist = tfp.distributions.Normal(probs_mean, probs_std)
        sample = dist.sample()
        print(f"✅ TFP Distribution Layer Operational. Sample Action Scalar: {sample.numpy()[0][0]:.4f}")
    except Exception as e:
        print(f"❌ TensorFlow Probability verification failed: {e}")

    print("\n" + "=" * 60)
    print("🏁 SANITY CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_sanity_check()