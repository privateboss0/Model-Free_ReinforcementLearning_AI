import tensorflow as tf
import os

ENV_ID = "LunarLander-v3"
SEED = 123
NUM_ENVS = 12

TOTAL_TIMESTEPS = 15_000_000
N_STEPS = 4096                
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 15
NUM_MINIBATCHES = 4
CLIP_RANGE = 0.1
LEARNING_RATE = 3e-4
RMS_WARMUP_STEPS = 5000 

-
VALUE_COEF = 0.5
ENTROPY_COEF = 0.1
MAX_GRAD_NORM = 0.5

DEVICE = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
if DEVICE == 'GPU':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
LOG_DIR = f"./Lunar_Lander_Discrete_logs/ppo_{ENV_ID.lower()}"


SAVE_PATH_ROOT = "./Lunar_Lander_Discrete_models"
SAVE_PATH = os.path.join(SAVE_PATH_ROOT, f"ppo_{ENV_ID.lower()}")
RESUME_FILE = f"ppo_{ENV_ID.lower()}_resume.json"
CHECKPOINT_FREQ = N_STEPS * NUM_ENVS * 21 
