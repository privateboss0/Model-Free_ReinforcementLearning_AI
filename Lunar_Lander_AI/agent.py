import tensorflow as tf
from keras.layers import Dense, Normalization, Input
from keras.models import Model
import tensorflow_probability as tfp
import numpy as np
import os
from config import *

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

class PPOAgent:
    def __init__(self, obs_shape, action_size, total_timesteps):
        self.obs_shape = obs_shape
        self.action_size = action_size
        
        self.policy = self._build_policy_model(obs_shape, action_size)
        self.value = self._build_value_model(obs_shape)
        
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=total_timesteps * PPO_EPOCHS / (N_STEPS * NUM_ENVS),
            end_learning_rate=0.0
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-5)
        
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        
        self.rms_mean_var = tf.Variable(self.obs_rms.mean, dtype=tf.float32, name="obs_rms_mean")
        self.rms_var_var = tf.Variable(self.obs_rms.var, dtype=tf.float32, name="obs_rms_var")
        self.rms_count_var = tf.Variable(self.obs_rms.count, dtype=tf.float32, name="obs_rms_count")

        self.checkpoint = tf.train.Checkpoint(
            policy=self.policy, 
            value=self.value, 
            optimizer=self.optimizer,
            obs_rms_mean=self.rms_mean_var,
            obs_rms_var=self.rms_var_var,
            obs_rms_count=self.rms_count_var
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, os.path.join(SAVE_PATH, 'tf_checkpoints'), max_to_keep=1000
        )

    def _build_policy_model(self, obs_shape, action_size):
        inputs = tf.keras.Input(shape=obs_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        logits = Dense(action_size)(x) 
        
        return tf.keras.Model(inputs=inputs, outputs=logits, name="policy_model")

    def _build_value_model(self, obs_shape):
        inputs = tf.keras.Input(shape=obs_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        value = Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=value, name="value_model")

    def adapt_normalization(self, initial_observations):
        """Updates the observation normalizer using initial data and saves state to checkpoint variables."""
        self.obs_rms.update(initial_observations)
        # Update checkpoint variables for persistence
        self.rms_mean_var.assign(self.obs_rms.mean)
        self.rms_var_var.assign(self.obs_rms.var)
        self.rms_count_var.assign(self.obs_rms.count)

    def normalize_obs(self, obs):
        """
        Applies normalization. This runs in Eager mode (NumPy input) or Graph mode (Tensor input).
        The source of RMS parameters is selected based on the input type.
        """
        is_tensor = tf.is_tensor(obs)
        
        rms_mean = self.rms_mean_var if is_tensor else self.obs_rms.mean
        rms_var = self.rms_var_var if is_tensor else self.obs_rms.var
        
        if not is_tensor:
            obs = obs.astype(np.float32)
        
        normalized_obs = (obs - rms_mean) / tf.sqrt(rms_var + 1e-8)
        
        normalized_obs = tf.clip_by_value(normalized_obs, -10.0, 10.0)
        
        if not is_tensor:
            return normalized_obs.numpy()
        
        return normalized_obs


    def select_action(self, obs):
        """Selects action, computes value, and log_prob for the given observation in Eager Mode."""
        
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        normalized_obs = self.normalize_obs(obs_tensor)
        
        logits = self.policy(normalized_obs, training=False)
        values = self.value(normalized_obs, training=False)
        
        distribution = tfp.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        
        log_probs = distribution.log_prob(actions)
        
       
        actions_np = actions.numpy().astype(np.int64) 
        values_np = values.numpy().flatten()
        log_probs_np = log_probs.numpy()
        
        return actions_np, values_np, log_probs_np

    @tf.function
    def learn_step(self, obs, actions, old_log_probs, returns, advantages, old_values):
        """Performs a single PPO optimization step."""
        with tf.GradientTape() as tape:
            
            logits = self.policy(obs, training=True)
            values = self.value(obs, training=True)
            
            distribution = tfp.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            ratio = tf.exp(log_probs - old_log_probs)
            
            values_clipped = old_values + tf.clip_by_value(values - old_values, -CLIP_RANGE, CLIP_RANGE)
            value_loss1 = tf.square(returns - values)
            value_loss2 = tf.square(returns - values_clipped)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))
            
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
            
            entropy = tf.reduce_mean(distribution.entropy())
            
            total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        
        grads = tape.gradient(total_loss, self.policy.trainable_variables + self.value.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables + self.value.trainable_variables))
        
        return -policy_loss, value_loss, entropy

    def learn(self, ppo_batch):
        """PPO update loop with data preparation and mini-batching."""
        
        self.obs_rms.update(ppo_batch['observations'])
        self.rms_mean_var.assign(self.obs_rms.mean)
        self.rms_var_var.assign(self.obs_rms.var)
        self.rms_count_var.assign(self.obs_rms.count)
        
        obs = self.normalize_obs(ppo_batch['observations'])

        actions = ppo_batch['actions']
        old_log_probs = ppo_batch['log_probs']
        returns = ppo_batch['returns']
        advantages = ppo_batch['advantages']
        old_values = ppo_batch['old_values']

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=np.int64) 
        old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns.flatten(), dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages.flatten(), dtype=tf.float32)
        old_values_tensor = tf.convert_to_tensor(old_values.flatten(), dtype=tf.float32)

        batch_size = obs_tensor.shape[0]
        minibatch_size = batch_size // NUM_MINIBATCHES
        
        policy_losses, value_losses, entropies = [], [], []

        for epoch in range(PPO_EPOCHS):

            indices = tf.range(batch_size)
            shuffled_indices = tf.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = shuffled_indices[start:end]
                
                mb_obs = tf.gather(obs_tensor, minibatch_indices)
                mb_actions = tf.gather(actions_tensor, minibatch_indices)
                mb_old_log_probs = tf.gather(old_log_probs_tensor, minibatch_indices)
                mb_returns = tf.gather(returns_tensor, minibatch_indices)
                mb_advantages = tf.gather(advantages_tensor, minibatch_indices)
                mb_old_values = tf.gather(old_values_tensor, minibatch_indices)
                
                p_loss, v_loss, entropy = self.learn_step(
                    mb_obs, mb_actions, mb_old_log_probs, mb_returns, mb_advantages, mb_old_values
                )
                policy_losses.append(p_loss.numpy())
                value_losses.append(v_loss.numpy())
                entropies.append(entropy.numpy())

        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)

    def save_model(self, save_dir, timesteps):
        """Saves the full checkpoint using the manager."""
        self.checkpoint_manager.save(checkpoint_number=timesteps)
        
    def load_model(self, save_dir, timesteps):
        """Loads the last successful checkpoint."""
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        
        if latest_checkpoint:
            print(f"Restoring checkpoint from {latest_checkpoint}...")

            self.checkpoint.restore(latest_checkpoint).expect_partial()
            

            self.obs_rms.mean = self.rms_mean_var.numpy()
            self.obs_rms.var = self.rms_var_var.numpy()
            self.obs_rms.count = self.rms_count_var.numpy()
            
            print("Model, Optimizer, and Normalizer restored successfully.")
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_manager.directory}")
