import tensorflow as tf

class AIFStabilityLoss(tf.keras.losses.Loss):
    def __init__(self, target_equilibrium_variance=1.0, penalty_weight=0.5, name="aif_stability_loss"):
        super(AIFStabilityLoss, self).__init__(name=name)
        self.target_variance = target_equilibrium_variance
        self.penalty_weight = penalty_weight

    def call(self, initial_state, next_state):
        """
        Calculates loss based on how perfectly the loop stabilizes the agent.
        
        initial_state: The agent's intelligence vector BEFORE the loop (batch_size, 16)
        next_state: The agent's intelligence vector AFTER the loop (batch_size, 16)
        """
        # 1. State Drift Loss: Measures the literal distance shifted during the loop
        state_drift = tf.reduce_mean(tf.square(next_state - initial_state), axis=-1)
        
        # 2. Chaos Penalty: Prevents the parameters from exploding or flatlining.
        # It forces the variance of the new state to match your target equilibrium.
        current_variance = tf.math.reduce_variance(next_state, axis=-1)
        chaos_penalty = tf.square(current_variance - self.target_variance)
        
        # 3. Total Closed-Loop Loss
        total_loss = state_drift + (self.penalty_weight * chaos_penalty)
        
        return tf.reduce_mean(total_loss)

loss_fn = AIFStabilityLoss(target_equilibrium_variance=1.0, penalty_weight=0.1)

# Simulating a stable transition vs an unstable, chaotic explosion
initial_intelligence = tf.random.normal((1, 16), mean=0.0, stddev=1.0)
stable_next_state = initial_intelligence + tf.random.normal((1, 16), mean=0.0, stddev=0.1)
exploded_next_state = initial_intelligence * 5.0  # Massive parameter explosion

print("Loss for Stable Loop Transition:", loss_fn(initial_intelligence, stable_next_state).numpy())
print("Loss for Chaotic Loop Explosion :", loss_fn(initial_intelligence, exploded_next_state).numpy())
