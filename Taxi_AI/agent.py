import tensorflow as tf
from keras import layers
import os

class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.model_dir = os.path.join(self.cfg.CHECKPOINT_DIR, "keras_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Build SEPARATE networks as per design architecture
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
        
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0),
            actor=self.actor,
            critic=self.critic,
            actor_opt=self.actor_opt,
            critic_opt=self.critic_opt
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=cfg.CHECKPOINT_DIR, max_to_keep=5
        )

    def build_actor(self):
        """The Policy Network: States -> Probability Distribution over Actions[cite: 1]"""
        inputs = layers.Input(shape=(1,))

        x = layers.Embedding(self.cfg.STATE_DIM, 64)(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.cfg.ACTION_DIM, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name="Actor_Network")

    def build_critic(self):
        """The Value Network: States -> Scalar Value V(s)[cite: 1]"""
        inputs = layers.Input(shape=(1,))
        x = layers.Embedding(self.cfg.STATE_DIM, 64)(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation=None)(x)
        return tf.keras.Model(inputs, outputs, name="Critic_Network")

    @tf.function
    def get_action_and_value(self, state):
        """Used during experience collection on the GPU[cite: 1]"""
        probs = self.actor(state)
        value = self.critic(state)
        
        # Log-probs for PPO ratio calculation[cite: 1]
        action = tf.random.categorical(tf.math.log(probs + 1e-10), 1)
        action = tf.squeeze(action, axis=-1)
        
        action_mask = tf.one_hot(action, self.cfg.ACTION_DIM)
        log_prob = tf.math.log(tf.reduce_sum(probs * action_mask, axis=1) + 1e-10)
        
        return action, log_prob, tf.squeeze(value)

    def save(self, step, final=False):
        """Persists models in both Checkpoint and Native Keras formats[cite: 1]"""
        # Save Checkpoint (for Resume Logic)[cite: 1]
        self.ckpt.step.assign(step)
        save_path = self.ckpt_manager.save()
        print(f"--- Checkpoint saved at step {step}: {save_path} ---[cite: 1]")
        
        # Save .keras models (Latest Native Format)[cite: 1]
        suffix = "FINAL" if final else f"step_{step}"
        
        actor_path = os.path.join(self.model_dir, f"actor_{suffix}.keras")
        critic_path = os.path.join(self.model_dir, f"critic_{suffix}.keras")
        
        # Using the new native format[cite: 1]
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        print(f"--- Native Keras Models saved as {suffix} to {self.model_dir} ---[cite: 1]")

    def load(self):
        """Resumes training from the latest checkpoint if it exists[cite: 1]"""
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"--- Restored from {self.ckpt_manager.latest_checkpoint} ---[cite: 1]")
            return int(self.ckpt.step.numpy())
        print("--- No checkpoint found. Starting from scratch. ---[cite: 1]")
        return 0