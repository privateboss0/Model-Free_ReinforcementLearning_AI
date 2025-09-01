import tensorflow as tf
import os
from PPO_Model import PPOAgent

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

if __name__ == "__main__":
    print("Initializing PPO Agent for evaluation...")
    
    agent = PPOAgent(**agent_config)

    root_log_dir = "./ppo_car_racing_logs"

    latest_log_run_dir = None
    if os.path.exists(root_log_dir):
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if all_runs:
    
            latest_log_run_dir = max(all_runs, key=os.path.getmtime)
            print(f"Found latest training run directory: {latest_log_run_dir}")
        else:
            print(f"No training run directories found in {root_log_dir}.")
    else:
        print(f"Log directory {root_log_dir} does not exist. Cannot find trained model.")


    model_to_load = None
    if latest_log_run_dir:

        final_model_path = os.path.join(latest_log_run_dir, "final_model.weights.h5")
        if os.path.exists(final_model_path):
            model_to_load = final_model_path
        else:
            print(f"Final model weights not found in {latest_log_run_dir}. Checking checkpoints...")
            
            checkpoint_dir = os.path.join(latest_log_run_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5")]
                if all_checkpoints:
                    model_to_load = max(all_checkpoints, key=os.path.getmtime)
                    print(f"Loading latest checkpoint: {model_to_load}")
                else:
                    print("No checkpoints found.")
            else:
                print("Checkpoints directory does not exist.")


    if model_to_load:
        
        print("\n--- Evaluation ---")
        agent.evaluate(num_episodes=20, render=True, model_path=model_to_load)
    else:
        print("No trained model found to evaluate. Please train an agent first.")
#L

