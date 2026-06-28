import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import config

class DiscreteActorCritic(tf.keras.Model):
    def __init__(self, num_actions=3):
        super(DiscreteActorCritic, self).__init__()
        # 🎭 Actor Layers (Outputs raw unnormalized log probabilities / logits)
        self.actor_dense1 = tf.keras.layers.Dense(128, activation='tanh', name='actor_dense1')
        self.actor_dense2 = tf.keras.layers.Dense(64, activation='tanh', name='actor_dense2')
        self.logits_layer = tf.keras.layers.Dense(num_actions, activation=None, name='logits')

        # 🧠 Critic Layers (Outputs state value scalar estimation)
        self.critic_dense1 = tf.keras.layers.Dense(128, activation='tanh', name='critic_dense1')
        self.critic_dense2 = tf.keras.layers.Dense(64, activation='tanh', name='critic_dense2')
        self.value_layer  = tf.keras.layers.Dense(1, activation=None, name='value')

    def call(self, state):

        x = self.actor_dense1(state)
        x = self.actor_dense2(x)
        logits = self.logits_layer(x)
        

        y = self.critic_dense1(state)
        y = self.critic_dense2(y)
        value = self.value_layer(y)
        
        return logits, value

class DiscretePPOAgent:
    def __init__(self, num_actions=3, lr_actor=2.5e-4, lr_critic=2.5e-4):
        self.num_actions = num_actions
        self.ac = DiscreteActorCritic(num_actions=self.num_actions)
        
        # Optimizers bound natively to bare-metal devices
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def get_vector_actions(self, states):
        """
        Batched action sampling loop for parallel environment data gathering.
        """
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        logits, values = self.ac(states_tensor)
        
        dist = tfp.distributions.Categorical(logits=logits)
        
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions.numpy(), log_probs.numpy(), tf.squeeze(values, axis=-1).numpy()
    
    def get_action_distribution(self, state):


        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)
            
        logits, _ = self.ac(state)
        
        probs = tf.nn.softmax(logits)
        
        return probs.numpy()[0]

    @tf.function
    def train_step(self, states, actions, old_log_probs, advantages, returns):

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            logits, values = self.ac(states)
            values = tf.squeeze(values, axis=-1)
            
            dist = tfp.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = tf.reduce_mean(dist.entropy())

            ratios = tf.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1.0 - config.POLICY_CLIP, 1.0 + config.POLICY_CLIP) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            critic_loss = tf.reduce_mean(tf.square(returns - values))

            total_actor_loss = actor_loss - (config.ENTROPY_COEF * entropy)
            total_critic_loss = config.CRITIC_COEF * critic_loss

        actor_grads = tape.gradient(total_actor_loss, self.ac.actor_dense1.trainable_variables + 
                                    self.ac.actor_dense2.trainable_variables + 
                                    self.ac.logits_layer.trainable_variables)
        
        critic_grads = tape.gradient(total_critic_loss, self.ac.critic_dense1.trainable_variables + 
                                     self.ac.critic_dense2.trainable_variables + 
                                     self.ac.value_layer.trainable_variables)

        self.actor_opt.apply_gradients(zip(actor_grads, self.ac.actor_dense1.trainable_variables + 
                                           self.ac.actor_dense2.trainable_variables + 
                                           self.ac.logits_layer.trainable_variables))
        
        self.critic_opt.apply_gradients(zip(critic_grads, self.ac.critic_dense1.trainable_variables + 
                                            self.ac.critic_dense2.trainable_variables + 
                                            self.ac.value_layer.trainable_variables))

        del tape # Explicitly purge persistent tape memory allocations
        return actor_loss, critic_loss, entropy