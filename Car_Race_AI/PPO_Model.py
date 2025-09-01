#O
#E
import gymnasium as gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras import optimizers
from keras.optimizers import Adam
import tensorflow_probability as tfp
import os
from datetime import datetime
import json
from RaceCar import ActorCritic
from Environment_Wrapper import CarRacingEnvWrapper

tfd = tfp.distributions

class PPOAgent:
    def __init__(self, env_id="CarRacing-v3", num_envs=21,
                 gamma=0.99, lam=0.95, clip_epsilon=0.2,
                 actor_lr=3e-4, critic_lr=3e-4,
                 ppo_epochs=10, minibatches=4,
                 steps_per_batch=1024,
                 num_stack_frames=4, resize_dim=(84, 84), grayscale=True,
                 seed=42, log_dir="./ppo_logs",
                 entropy_coeff=0.01,
                 save_interval_timesteps=537600, 
                 hidden_layer_sizes=[512, 512, 512]):
        
        self.env_id = env_id
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatches = minibatches
        self.steps_per_batch = steps_per_batch
        self.num_stack_frames = num_stack_frames
        self.resize_dim = resize_dim
        self.grayscale = grayscale
        self.seed = seed
        self.log_dir = log_dir
        self.entropy_coeff = entropy_coeff
        self.save_interval_timesteps = save_interval_timesteps
        self.hidden_layer_sizes = hidden_layer_sizes

        self.envs = self._make_vec_envs()
        self.action_dim = self.envs.single_action_space.shape[0]
        self.observation_shape = self.envs.single_observation_space.shape

        self.model = ActorCritic(self.action_dim, 
                                 self.num_stack_frames, 
                                 self.resize_dim[1], 
                                 self.resize_dim[0],
                                 hidden_layer_sizes=self.hidden_layer_sizes)
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

        self.train_log_dir = None
        self.summary_writer = None

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        dummy_input = np.zeros((1, *self.observation_shape), dtype=np.uint8)
        _ = self.model(dummy_input)

    def _make_env(self):
        env = gym.make(self.env_id, render_mode="rgb_array", continuous=True)
        env = CarRacingEnvWrapper(env, num_stack_frames=self.num_stack_frames,
                                  grayscale=self.grayscale, resize_dim=self.resize_dim)
        return env

    def _make_vec_envs(self):
        return gym.vector.SyncVectorEnv([self._make_env for _ in range(self.num_envs)])

    def _compute_returns_and_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.steps_per_batch)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + values[:-1]

        return advantages, returns

    @tf.function
    def _train_step(self, observations, actions, old_log_probs, advantages, returns):
        with tf.GradientTape() as tape:
            action_distribution, value_pred = self.model(observations)
            value_pred = tf.squeeze(value_pred, axis=-1)

            critic_loss = tf.reduce_mean(tf.square(returns - value_pred))

            log_prob = action_distribution.log_prob(actions)
            ratio = tf.exp(log_prob - old_log_probs)

            ratio = tf.where(tf.math.is_nan(ratio), 1.0, ratio)
            ratio = tf.where(tf.math.is_inf(ratio), tf.sign(ratio) * 1e5, ratio)
            
            pg_loss1 = ratio * advantages
            pg_loss2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))

            entropy = tf.reduce_mean(action_distribution.entropy())
            entropy_loss = -self.entropy_coeff * entropy
            
            total_loss = actor_loss + critic_loss + entropy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return actor_loss, critic_loss, entropy, total_loss

    def train(self, total_timesteps, resume_from_timestep=0, resume_model_path=None, run_log_dir=None):
        global_timestep = resume_from_timestep
        ep_rewards = deque(maxlen=100)
        ep_modified_rewards = deque(maxlen=100)
        
        resume_json_path = "resume_config.json"
        
        if run_log_dir is None:
            if os.path.exists(resume_json_path):
                with open(resume_json_path, "r") as f:
                    resume_info = json.load(f)
                    self.train_log_dir = resume_info.get("run_log_directory")
            
            if self.train_log_dir is None:
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.train_log_dir = os.path.join(self.log_dir, current_time)
        else:
            self.train_log_dir = run_log_dir

        os.makedirs(os.path.join(self.train_log_dir, "checkpoints"), exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        if resume_model_path and os.path.exists(resume_model_path):
            self.model.load_weights(resume_model_path)
            print(f"Resuming training from timestep {global_timestep} and loaded model from: {resume_model_path}")
        elif resume_from_timestep > 0:
            print(f"WARNING: Attempting to resume from timestep {global_timestep} but no valid model path provided or found. Starting fresh.")
        else:
            print("Starting new training run.")

        obs, _ = self.envs.reset(seed=self.seed)

        print(f"Target total timesteps: {total_timesteps}")
        print(f"Current global timestep: {global_timestep}")

        resume_save_frequency = self.steps_per_batch * self.num_envs * 5 
        
        while global_timestep < total_timesteps:
            batch_observations = np.zeros((self.steps_per_batch, self.num_envs, *self.observation_shape), dtype=np.uint8)
            batch_actions = np.zeros((self.steps_per_batch, self.num_envs, self.action_dim), dtype=np.float32)
            batch_rewards = np.zeros((self.steps_per_batch, self.num_envs), dtype=np.float32)
            batch_dones = np.zeros((self.steps_per_batch, self.num_envs), dtype=bool)
            batch_values = np.zeros((self.steps_per_batch, self.num_envs), dtype=np.float32)
            batch_log_probs = np.zeros((self.steps_per_batch, self.num_envs), dtype=np.float32)
            batch_original_rewards = np.zeros((self.steps_per_batch, self.num_envs), dtype=np.float32)

            for i in range(self.steps_per_batch):
                tf_obs = tf.convert_to_tensor(obs, dtype=tf.uint8)
                action_dist, value = self.model(tf_obs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                action_np = action.numpy()
                value_np = tf.squeeze(value).numpy()
                log_prob_np = log_prob.numpy()

                next_obs, reward, terminated, truncated, info = self.envs.step(action_np)
                done = terminated | truncated

                batch_observations[i] = obs
                batch_actions[i] = action_np
                batch_rewards[i, :] = reward
                batch_dones[i] = done
                batch_values[i] = value_np
                batch_log_probs[i] = log_prob_np
                
                if not isinstance(info, list):
                    info = [info]

                for inf_idx, (inf, d) in enumerate(zip(info, done)):
                    original_reward_value = inf.get("original_reward", reward[inf_idx])
                    if isinstance(original_reward_value, (list, np.ndarray)):
                        batch_original_rewards[i, inf_idx] = original_reward_value[0]
                    else:
                        batch_original_rewards[i, inf_idx] = original_reward_value
                    
                    if d:
                        if isinstance(inf, dict):
                            final_info = inf.get("final_info")
                            if isinstance(final_info, dict) and "episode" in final_info:
                                ep_rewards.append(final_info["episode"]["r"])
                
                obs = next_obs
                                
                global_timestep += self.num_envs
            
            _, last_values_np = self.model(tf.convert_to_tensor(obs, dtype=tf.uint8))
            last_values_np = tf.squeeze(last_values_np).numpy()
            
            batch_values = np.concatenate((batch_values, last_values_np[np.newaxis, :]), axis=0)

            advantages, returns = self._compute_returns_and_advantages(
                batch_rewards, batch_values, batch_dones
            )
            
            flat_observations = batch_observations.reshape((-1, *self.observation_shape))
            flat_actions = batch_actions.reshape((-1, self.action_dim))
            flat_old_log_probs = batch_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)

            flat_advantages = (flat_advantages - np.mean(flat_advantages)) / (np.std(flat_advantages) + 1e-8)

            batch_original_total_reward = np.sum(batch_original_rewards)
            batch_modified_total_reward = np.sum(batch_rewards)

            batch_indices = np.arange(self.steps_per_batch * self.num_envs)
            for _ in range(self.ppo_epochs):
                np.random.shuffle(batch_indices)
                for start_idx in range(0, len(batch_indices), len(batch_indices) // self.minibatches):
                    end_idx = start_idx + len(batch_indices) // self.minibatches
                    minibatch_indices = batch_indices[start_idx:end_idx]

                    mb_obs = tf.constant(flat_observations[minibatch_indices], dtype=tf.uint8)
                    mb_actions = tf.constant(flat_actions[minibatch_indices], dtype=tf.float32)
                    mb_old_log_probs = tf.constant(flat_old_log_probs[minibatch_indices], dtype=tf.float32)
                    mb_advantages = tf.constant(flat_advantages[minibatch_indices], dtype=tf.float32)
                    mb_returns = tf.constant(flat_returns[minibatch_indices], dtype=tf.float32)

                    actor_loss, critic_loss, entropy, total_loss = self._train_step(
                        mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns
                    )
            
            with self.summary_writer.as_default():
                tf.summary.scalar("charts/total_timesteps", global_timestep, step=global_timestep)
                tf.summary.scalar("losses/actor_loss", actor_loss, step=global_timestep)
                tf.summary.scalar("losses/critic_loss", critic_loss, step=global_timestep)
                tf.summary.scalar("losses/entropy", entropy, step=global_timestep)
                tf.summary.scalar("losses/total_loss", total_loss, step=global_timestep)
                
                if ep_rewards: 
                    tf.summary.scalar("charts/avg_episode_reward_original_env", np.mean(ep_rewards), step=global_timestep)
                    tf.summary.scalar("charts/min_episode_reward_original_env", np.min(ep_rewards), step=global_timestep)
                    tf.summary.scalar("charts/max_episode_reward_original_env", np.max(ep_rewards), step=global_timestep)
                
                tf.summary.scalar("charts/batch_total_reward_original", batch_original_total_reward, step=global_timestep)
                tf.summary.scalar("charts/batch_total_reward_modified", batch_modified_total_reward, step=global_timestep)

                with tf.name_scope('actor_std'):
                    tf.summary.histogram('log_std', self.model.actor_log_std, step=global_timestep)
                    tf.summary.scalar('std_mean_overall', tf.reduce_mean(tf.exp(self.model.actor_log_std)), step=global_timestep)

            if global_timestep % (self.steps_per_batch * self.num_envs * 5) == 0:
                if ep_rewards:
                    avg_orig_reward_str = f"{np.mean(ep_rewards):.2f}"
                else:
                    avg_orig_reward_str = 'N/A'
                
                #print(f"Timestep: {global_timestep}, Avg Original Env Reward (100 eps): {avg_orig_reward_str}")
                print(f"Timestep: {global_timestep}, Number of episodes in Timestep: (Check tensorboard..lol)") #{avg_orig_reward_str}")

                current_checkpoint_path = os.path.join(self.train_log_dir, "checkpoints", f"Actor-Critic_at_{global_timestep}.weights.h5")
                self.model.save_weights(current_checkpoint_path)

                resume_info = {
                    "last_global_timestep": global_timestep,
                    "last_checkpoint_path": current_checkpoint_path,
                    "run_log_directory": self.train_log_dir 
                }
                with open(resume_json_path, "w") as f:
                    json.dump(resume_info, f, indent=4)
                print(f"Resume info saved to {resume_json_path}")


        print("Training finished.")
        self.envs.close()
        self.summary_writer.close()
        
        final_model_path = os.path.join(self.train_log_dir, "Actor-Critic_final_model.weights.h5")
        self.model.save_weights(final_model_path)
        
        if os.path.exists(resume_json_path):
            os.remove(resume_json_path)
            print(f"Removed {resume_json_path} as training completed successfully.")

    def evaluate(self, num_episodes=5, render=True, model_path=None):
        eval_env = gym.make(self.env_id, render_mode="human" if render else "rgb_array", continuous=True)
        eval_env = CarRacingEnvWrapper(eval_env, num_stack_frames=self.num_stack_frames,
                                       grayscale=self.grayscale, resize_dim=self.resize_dim)

        if model_path:
            dummy_input = np.zeros((1, *self.observation_shape), dtype=np.uint8)
            _ = self.model(dummy_input)
            self.model.load_weights(model_path)
            print(f"Loaded model from {model_path}")

        episode_rewards = []
        episode_original_rewards = []
        
        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            total_original_reward = 0
            while not done:
                tf_obs = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.uint8)
                action_dist, _ = self.model(tf_obs)
                action = action_dist.mean().numpy().flatten()
                
                action[0] = np.clip(action[0], -1.0, 1.0)
                action[1] = np.clip(action[1], 0.0, 1.0)
                action[2] = np.clip(action[2], 0.0, 1.0)

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_original_reward += info.get('original_reward', reward)

            episode_rewards.append(total_reward)
            episode_original_rewards.append(total_original_reward)
            print(f"Episode {ep+1} finished. Modified Reward: {total_reward:.2f}, Original Env Reward: {total_original_reward:.2f}")
        eval_env.close()
        print(f"Average modified evaluation reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
        print(f"Average original environment reward over {num_episodes} episodes: {np.mean(episode_original_rewards):.2f}")
        return episode_rewards, episode_original_rewards

