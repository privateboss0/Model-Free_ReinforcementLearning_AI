import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium import Wrapper

TIME_PENALTY = -0.05 

class LunarLanderRewardShaping(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.last_shaping_reward = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        x_pos = observation[0] 
        y_pos = observation[1] 
        x_vel = observation[2] 
        y_vel = observation[3] 
        angle = observation[4] 
        angular_vel = observation[5] 
        
        left_leg_contact = observation[6]
        right_leg_contact = observation[7]
        
        main_engine_fired = 1 if action == 2 else 0
        
        any_thruster_fired = 1 if action != 0 else 0
        
        current_shaping_reward = 0.0
        
        current_shaping_reward += -30 * np.abs(x_pos)
        
       
        y_factor = 1.0 - y_pos                                      
        current_shaping_reward += -20 * np.abs(y_vel) * y_factor 
        
        current_shaping_reward += -10 * np.abs(x_vel)
        
        current_shaping_reward += -5 * np.abs(angle)

        current_shaping_reward += -10 * np.abs(angular_vel)
        
        current_shaping_reward += -50 * main_engine_fired * (1.0 - y_pos) 
        
        current_shaping_reward += -20 * y_pos
        
        contact_sum = left_leg_contact + right_leg_contact
        if contact_sum > 0:
            current_shaping_reward += -900 * any_thruster_fired * contact_sum
        
        current_shaping_reward += 15 * contact_sum
        
        if self.last_shaping_reward is not None:
            shaping_reward_diff = current_shaping_reward - self.last_shaping_reward

            reward += np.clip(shaping_reward_diff, -10.0, 10.0) 
            
        self.last_shaping_reward = current_shaping_reward
        
        reward += TIME_PENALTY 
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
       
        self.last_shaping_reward = None
        return self.env.reset(**kwargs)
    
    def render(self):
       
        return self.env.render()
    
    def close(self):
        """Passes close call to the base environment."""
        return self.env.close()
    
    def __getattr__(self, name):
        
        return getattr(self.env, name)
