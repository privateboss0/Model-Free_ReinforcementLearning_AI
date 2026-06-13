from datetime import datetime
import os
import tensorflow as tf

class Config:
    # Environment
    ENV_NAME = "Taxi-v4"
    NUM_ENVS = 11
    STATE_DIM = 500  # Discrete states
    ACTION_DIM = 6   # 6 discrete actions

    # Hyperparameters
    LEARNING_RATE = 3e-4
    GAE_LAMBDA = 0.95
    GAMMA = 0.99
    PPO_EPSILON = 0.2
    ENTROPY_BETA = 0.01
    ROLLOUT_STEPS = 256 # Added: needed for GAE and buffer logic
    UPDATE_EPOCHS = 10
    BATCH_SIZE = 64
    MAX_TIMESTEPS = 40_000_000
    SAVE_FREQ = 100 

    # Paths
    CHECKPOINT_DIR = "ppo_taxi/checkpoints"
    # FIXED: removed the extra .datetime
    LOG_DIR = os.path.join("taxilogs", f"ppo_taxi_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Ensure directories exist right at the start
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("taxilogs", exist_ok=True)

    # Hardware
    # Explicitly maps to your RTX 4060
    DEVICE = "/device:GPU:0" if tf.config.list_physical_devices('GPU') else "/device:CPU:0"