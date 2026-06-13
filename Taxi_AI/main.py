import gymnasium as gym
import tensorflow as tf
import numpy as np
import os
from config import Config
from agent import PPOAgent
import utils

class Trainer:
    def __init__(self):
        self.cfg = Config()
        # Using Async for multiple cores on the server
        self.envs = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(self.cfg.ENV_NAME) for _ in range(self.cfg.NUM_ENVS)]
        )
        self.agent = PPOAgent(self.cfg)
        self.writer = tf.summary.create_file_writer(self.cfg.LOG_DIR)
        
        self.global_step = self.agent.load() 
        self.total_timesteps = self.global_step * self.cfg.ROLLOUT_STEPS * self.cfg.NUM_ENVS

    @tf.function
    def train_step(self, states, actions, old_log_probs, returns, advantages):
        with tf.GradientTape(persistent=True) as tape:
            # 1. Get new predictions from separate networks[cite: 1]
            probs = self.agent.actor(states, training=True)
            values = self.agent.critic(states, training=True)
            
            action_masks = tf.one_hot(actions, self.cfg.ACTION_DIM)
            new_log_probs = tf.math.log(tf.reduce_sum(probs * action_masks, axis=1) + 1e-10)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.cfg.PPO_EPSILON, 1.0 + self.cfg.PPO_EPSILON) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            total_actor_loss = actor_loss - (self.cfg.ENTROPY_BETA * tf.reduce_mean(entropy))
            
            # Critic Loss (Value Estimation)[cite: 1]
            critic_loss = tf.keras.losses.MSE(returns, tf.squeeze(values))

        # Applying gradients separately to maintain architectural independence[cite: 1]
        actor_grads = tape.gradient(total_actor_loss, self.agent.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.agent.critic.trainable_variables)
        
        self.agent.actor_opt.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables))
        self.agent.critic_opt.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))
        
        return actor_loss, critic_loss, tf.reduce_mean(entropy)

    def run(self):
        states, _ = self.envs.reset()
        print(f"Starting training on {self.cfg.DEVICE}...")

        while self.total_timesteps < self.cfg.MAX_TIMESTEPS:

            mb_states, mb_actions, mb_log_probs, mb_values, mb_rewards, mb_masks = [], [], [], [], [], []
            
            for _ in range(self.cfg.ROLLOUT_STEPS):
                # Run forward pass on GPU[cite: 1]
                actions, log_probs, values = self.agent.get_action_and_value(states)
                
                next_states, rewards, terminations, truncations, _ = self.envs.step(actions.numpy())
                masks = 1.0 - (terminations | truncations)
                
                mb_states.append(states)
                mb_actions.append(actions)
                mb_log_probs.append(log_probs)
                mb_values.append(values)
                mb_rewards.append(rewards)
                mb_masks.append(masks)
                
                states = next_states

            _, _, next_value = self.agent.get_action_and_value(states)
            returns, advantages = utils.compute_gae(
                next_value, mb_rewards, mb_masks, mb_values, self.cfg.GAMMA, self.cfg.GAE_LAMBDA
            )

            a_loss, c_loss, ent = self.train_step(
                np.concatenate(mb_states), 
                np.concatenate(mb_actions), 
                np.concatenate(mb_log_probs), 
                np.array(returns).flatten(), 
                utils.normalize(np.array(advantages).flatten())
            )

            self.total_timesteps += (self.cfg.NUM_ENVS * self.cfg.ROLLOUT_STEPS)
            self.global_step += 1

            with self.writer.as_default():
                tf.summary.scalar('Loss/Actor_Loss', a_loss, step=self.global_step)
                tf.summary.scalar('Loss/Critic_Loss', c_loss, step=self.global_step)
                tf.summary.scalar('Stats/Entropy', ent, step=self.global_step)
                tf.summary.scalar('Stats/Mean_Reward', np.mean(mb_rewards), step=self.global_step)
                tf.summary.scalar('Progress/Total_Timesteps', self.total_timesteps, step=self.global_step)

            # Save Checkpoint and .h5 models[cite: 1]
            if self.global_step % self.cfg.SAVE_FREQ == 0:
                self.agent.save(self.global_step)
                print(f"Step: {self.total_timesteps} | Reward: {np.mean(mb_rewards):.2f} | Entropy: {ent:.4f}")

        print("Target Timesteps reached. Training complete.")
        self.envs.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()