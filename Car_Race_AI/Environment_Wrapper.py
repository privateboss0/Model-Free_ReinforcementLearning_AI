#G
#D
import gymnasium as gym
import numpy as np
from collections import deque
import cv2 

class CarRacingEnvWrapper(gym.Wrapper):
    def __init__(self, env, num_stack_frames=4, grayscale=True, resize_dim=(84, 84)):
        super().__init__(env)
        self.num_stack_frames = num_stack_frames
        self.grayscale = grayscale
        self.resize_dim = resize_dim 

        self.frames = deque(maxlen=num_stack_frames)

        original_shape = self.env.observation_space.shape
        if grayscale:
        
            original_shape = original_shape[:-1] 

        if resize_dim:
            self.observation_shape = (resize_dim[1], resize_dim[0])
        else:
            self.observation_shape = original_shape[:2]

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.observation_shape[0], self.observation_shape[1], num_stack_frames),
            dtype=np.uint8 
        )

        self.OFF_TRACK_PENALTY_SCALE = 0.1 
        self.GRASS_COLOR_THRESHOLD = 180 

    def _preprocess_frame(self, frame):

        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.resize_dim:
            frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_AREA)
        return frame

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        processed_frame = self._preprocess_frame(observation)

        for _ in range(self.num_stack_frames):
            self.frames.append(processed_frame)

        stacked_frames = np.stack(self.frames, axis=-1)
        return stacked_frames, info

    def step(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        modified_reward = reward

        is_on_grass = np.mean(observation[:, :, 1]) > self.GRASS_COLOR_THRESHOLD
        
        if is_on_grass:
            modified_reward -= self.OFF_TRACK_PENALTY_SCALE
           
        info['is_on_grass'] = is_on_grass
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward

        processed_frame = self._preprocess_frame(observation)

        self.frames.append(processed_frame)
        stacked_frames = np.stack(self.frames, axis=-1)


        return stacked_frames, modified_reward, terminated, truncated, info
