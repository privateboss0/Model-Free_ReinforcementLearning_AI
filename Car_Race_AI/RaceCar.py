import tensorflow as tf
import keras
from keras import layers, Model
import tensorflow_probability as tfp

tfd = tfp.distributions

class ActorCritic(keras.Model):
    def __init__(self, action_dim, num_stack_frames, img_height, img_width, hidden_layer_sizes):
        super().__init__()
        self.conv_layers = keras.Sequential([
            layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(img_height, img_width, num_stack_frames)),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
        ])

        actor_layers = []
        for size in hidden_layer_sizes:
            actor_layers.append(layers.Dense(size, activation="relu"))
        self.common_actor_layer = keras.Sequential(actor_layers)

        self.actor_mean = layers.Dense(action_dim, activation=None) 
        self.actor_log_std = tf.Variable(tf.zeros(action_dim, dtype=tf.float32), trainable=True)

        critic_layers = []
        for size in hidden_layer_sizes:
            critic_layers.append(layers.Dense(size, activation="relu"))
        self.common_critic_layer = keras.Sequential(critic_layers)

        self.critic_value = layers.Dense(1, activation=None) 

    def call(self, inputs):
        normalized_inputs = tf.cast(inputs, tf.float32) / 255.0
        
        features = self.conv_layers(normalized_inputs)
        
        actor_features = self.common_actor_layer(features)
        mean = self.actor_mean(actor_features)
        
        std = tf.exp(self.actor_log_std) 
        
        action_distribution = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

        critic_features = self.common_critic_layer(features)
        value = self.critic_value(critic_features)
        
        return action_distribution, value