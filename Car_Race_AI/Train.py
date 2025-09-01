import tensorflow as tf
import os
import json
from PPO_Model import PPOAgent
from datetime import datetime

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

agent_config = {
    "env_id": "CarRacing-v3",
    "num_envs": 21,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_epsilon": 0.2,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "ppo_epochs": 10,
    "minibatches": 4,
    "steps_per_batch": 1024,
    "num_stack_frames": 4,
    "resize_dim": (84, 84),
    "grayscale": True,
    "seed": 42,
    "log_dir": "./ppo_car_racing_logs",
    "entropy_coeff": 0.01,
    'save_interval_timesteps': 537600,
    'hidden_layer_sizes': [512, 512, 512]
}

RESUME_TRAINING_FLAG = True
RESUME_CONFIG_FILE = "resume_config.json"

if __name__ == "__main__":
    resume_from_timestep = 0
    resume_model_path = None
    run_log_directory = None

    if RESUME_TRAINING_FLAG and os.path.exists(RESUME_CONFIG_FILE):
        try:
            with open(RESUME_CONFIG_FILE, "r") as f:
                resume_info = json.load(f)
            resume_from_timestep = resume_info.get("last_global_timestep", 0)
            resume_model_path = resume_info.get("last_checkpoint_path", None)
            run_log_directory = resume_info.get("run_log_directory", None)
            print(f"Found resume config: Will attempt to resume from timestep {resume_from_timestep}")
            print(f"Loading model from: {resume_model_path}")
            print(f"Continuing logging in directory: {run_log_directory}")

            if not (resume_model_path and os.path.exists(resume_model_path)):
                print("WARNING: Resume model path invalid or not found. Starting a new run.")
                resume_from_timestep = 0
                resume_model_path = None
                run_log_directory = None
                os.remove(RESUME_CONFIG_FILE)
        except (IOError, json.JSONDecodeError) as e:
            print(f"WARNING: Failed to read or parse resume config file. Starting a new run. Error: {e}")
            resume_from_timestep = 0
            resume_model_path = None
            run_log_directory = None
            if os.path.exists(RESUME_CONFIG_FILE):
                os.remove(RESUME_CONFIG_FILE)
    
    if run_log_directory is None:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_directory = os.path.join(agent_config["log_dir"], current_time)
        print(f"No valid resume config found. Starting a new run in: {run_log_directory}")
    
    agent_config["log_dir"] = run_log_directory 

    print("Initializing PPO Agent...")
    agent = PPOAgent(**agent_config)

    total_timesteps = 30_000_000
    try:
        agent.train(total_timesteps,
                    resume_from_timestep=resume_from_timestep,
                    resume_model_path=resume_model_path,
                    run_log_dir=run_log_directory)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state for resume...")
        print("State likely saved by periodic checkpointing. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Attempting to save current state for resume before exiting...")
    finally:
        print(f"\nTraining session ended. TensorBoard logs available at: tensorboard --logdir {agent.train_log_dir}")
        print("To view logs, run the above command in your terminal.")