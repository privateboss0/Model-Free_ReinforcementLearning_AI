import tensorflow as tf
import gymnasium as gym

# GPU Verification ---
print("--- Starting TensorFlow GPU Verification ---")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        
        print(len(gpus), "Physical GPUs found.")
        print(len(logical_gpus), "Logical GPUs configured (Memory Growth Enabled).")
        print("TensorFlow Version:", tf.__version__)

    except RuntimeError as e:
        print(f"Runtime Error during GPU setup: {e}")
else:
    print("No Physical GPUs found. TensorFlow will run on CPU.")

print("---------------------------------------------------")

# Environment Test
ENV_NAME = "LunarLander-v3"

try:
    print(f"Testing environment: {ENV_NAME}")
    
    WIND_POWER = 15.0
    TURBULENCE_POWER = 1.5
    
    env = gym.make(
        ENV_NAME,
        enable_wind=True,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER
    )
    
    print(f"Observation Space (State size): {env.observation_space.shape}")
    print(f"Action Space (Actions): {env.action_space}")

    # Small test run
    obs, info = env.reset()
    obs_next, reward, terminated, truncated, info = env.step(env.action_space.sample())

    print(f"\n{ENV_NAME} created and tested successfully.")
    
except Exception as e:
    print(f"\nError creating or testing environment '{ENV_NAME}': {e}")
    print("Please ensure 'gymnasium[box2d]' is installed.")

finally:
    if 'env' in locals() and env:
        env.close()

print("--- Verification Complete ---")
