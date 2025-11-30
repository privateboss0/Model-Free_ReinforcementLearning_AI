import tensorflow as tf
import keras
from keras import layers, Model
import numpy as np
import tensorflow_probability as tfp
import os
import traceback 

tfd = tfp.distributions

@tf.keras.utils.register_keras_serializable()
class Actor(Model):
    def __init__(self, obs_shape, action_size, hidden_layer_sizes=[512, 512, 512], **kwargs):
        super().__init__(**kwargs)

        if len(obs_shape) > 1:
            self.flatten = layers.Flatten(input_shape=obs_shape)
        
            self.flatten(tf.zeros((1,) + obs_shape))
        else:
            self.flatten = None 
        
        self.dense_layers = []
        for size in hidden_layer_sizes:
            self.dense_layers.append(layers.Dense(size, activation='relu'))
        self.logits = layers.Dense(action_size)

        self._obs_shape = obs_shape
        self._action_size = action_size
        self._hidden_layer_sizes = hidden_layer_sizes

    def call(self, inputs):
    
        x = self.flatten(inputs) if self.flatten else inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.logits(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'obs_shape': self._obs_shape,
            'action_size': self._action_size,
            'hidden_layer_sizes': self._hidden_layer_sizes
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Critic(Model):
    def __init__(self, obs_shape, hidden_layer_sizes=[512, 512, 512], **kwargs):
        super().__init__(**kwargs)

        if len(obs_shape) > 1:
            self.flatten = layers.Flatten(input_shape=obs_shape)
            self.flatten(tf.zeros((1,) + obs_shape))
        else:
            self.flatten = None
        
        self.dense_layers = []
        for size in hidden_layer_sizes:
            self.dense_layers.append(layers.Dense(size, activation='relu'))
        self.value = layers.Dense(1)

        self._obs_shape = obs_shape
        self._hidden_layer_sizes = hidden_layer_sizes

    def call(self, inputs):
        x = self.flatten(inputs) if self.flatten else inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.value(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'obs_shape': self._obs_shape,
            'hidden_layer_sizes': self._hidden_layer_sizes
        })
        return config

class PPOAgent:
    def __init__(self, observation_space_shape, action_space_size,
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2,
                 num_epochs_per_update=10, batch_size=64,
                 hidden_layer_sizes=[512, 512, 512]):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.num_epochs_per_update = num_epochs_per_update
        self.batch_size = batch_size

        self.observation_space_shape = observation_space_shape 
        self.action_space_size = action_space_size 

        self.actor = Actor(observation_space_shape, action_space_size, hidden_layer_sizes=hidden_layer_sizes)
        self.critic = Critic(observation_space_shape, hidden_layer_sizes=hidden_layer_sizes)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

        dummy_obs = tf.zeros((1,) + observation_space_shape, dtype=tf.float32)
        self.actor(dummy_obs)
        self.critic(dummy_obs)

    def remember(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    @tf.function

    def _choose_action_tf(self, observation, action_mask):
        observation = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), 0)
        
        pi_logits = self.actor(observation)
        
        masked_logits = tf.where(action_mask, pi_logits, -1e9)
        
        value = self.critic(observation)
        
        distribution = tfd.Categorical(logits=masked_logits)
        
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action, log_prob, value

    def choose_action(self, observation, action_mask):
        action_tensor, log_prob_tensor, value_tensor = self._choose_action_tf(observation, action_mask)
        
        return action_tensor.numpy(), log_prob_tensor.numpy(), value_tensor.numpy()[0,0]

    def calculate_advantages_and_returns(self):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        last_next_state_value = self.critic(tf.expand_dims(tf.convert_to_tensor(self.next_states[-1], dtype=tf.float32), 0)).numpy()[0,0] if not dones[-1] else 0
        next_values = np.append(values[1:], last_next_state_value)
        
        advantages = []
        returns = []
        
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
          
            advantage = delta + self.gae_lambda * self.gamma * (1 - dones[t]) * last_advantage
            advantages.insert(0, advantage)

            returns.insert(0, advantage + values[t])
            last_advantage = advantage
            
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def learn(self):
        if not self.states:
            return

        states = tf.convert_to_tensor(np.array(self.states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.actions), dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(np.array(self.log_probs), dtype=tf.float32)
        
        advantages, returns = self.calculate_advantages_and_returns()
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, advantages, returns))
        dataset = dataset.shuffle(buffer_size=len(self.states)).batch(self.batch_size)

        for _ in range(self.num_epochs_per_update):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataset:
    
                with tf.GradientTape() as tape:
                    current_logits = self.actor(batch_states)
                    
                    new_distribution = tfd.Categorical(logits=current_logits) 
                    
                    new_log_probs = new_distribution.log_prob(batch_actions)
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                                        
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                                        
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                                
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    new_values = self.critic(batch_states)
                    critic_loss = tf.reduce_mean(tf.square(new_values - batch_returns))
                                
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def save_models(self, path):
        actor_save_path = f"{path}_actor.keras"
        critic_save_path = f"{path}_critic.keras"
        print(f"\n--- Attempting to save models ---")
        print(f"Target Actor path: {os.path.abspath(actor_save_path)}")
        print(f"Target Critic path: {os.path.abspath(critic_save_path)}")
        try:
            self.actor.save(actor_save_path)
            print(f"Actor model saved successfully to {os.path.abspath(actor_save_path)}")
        except Exception as e:
            print(f"ERROR: Failed to save Actor model to {os.path.abspath(actor_save_path)}")
            print(f"Reason: {e}")
            traceback.print_exc()
        try:
            self.critic.save(critic_save_path)
            print(f"Critic model saved successfully to {os.path.abspath(critic_save_path)}")
        except Exception as e:
            print(f"ERROR: Failed to save Critic model to {os.path.abspath(critic_save_path)}")
            print(f"Reason: {e}")
            traceback.print_exc()
        print(f"--- Models save process completed ---\n")

    def load_models(self, path):
    
        actor_load_path = f"{path}_actor.keras"
        critic_load_path = f"{path}_critic.keras"
        actor_loaded_ok = False
        critic_loaded_ok = False

        custom_objects = {
            'Actor': Actor,
            'Critic': Critic
        }
        
        try:

            self.actor = tf.keras.models.load_model(actor_load_path, custom_objects=custom_objects)
            actor_loaded_ok = True
            print(f"Actor model loaded from: {os.path.abspath(actor_load_path)}")
        except Exception as e:
            print(f"ERROR: Failed to load Actor model from {os.path.abspath(actor_load_path)}")
            print(f"Reason: {e}")
            traceback.print_exc()

        try:
            self.critic = tf.keras.models.load_model(critic_load_path, custom_objects=custom_objects)
            critic_loaded_ok = True
            print(f"Critic model loaded from: {os.path.abspath(critic_load_path)}")
        except Exception as e:
            print(f"ERROR: Failed to load Critic model from {os.path.abspath(critic_load_path)}")
            print(f"Reason: {e}")
            traceback.print_exc()

        if actor_loaded_ok and critic_loaded_ok:
            print(f"All PPO models loaded successfully from '{path}'.")
            return True
        else:
            print(f"Warning: One or both models failed to load. The agent will use untrained models.")
            return False
