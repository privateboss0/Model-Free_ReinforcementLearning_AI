import numpy as np

def compute_gae(next_value, rewards, masks, values, gamma, lam):
    """
    Computes Generalized Advantage Estimation (GAE) and Returns.
    This logic smears the 'success' of a drop-off back through the 11 parallel streams.
    """
    values = values + [next_value]
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        # Temporal Difference (TD) error
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]

        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
        
    advantages = np.array(returns) - np.array(values[:-1])
    return returns, advantages

def normalize(x):
    """Standardizes advantages to have mean 0 and std 1 for training stability."""
    return (x - x.mean()) / (x.std() + 1e-8)