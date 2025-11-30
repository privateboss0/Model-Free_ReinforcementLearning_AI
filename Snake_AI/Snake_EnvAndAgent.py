import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import numpy as np
import collections
from collections import deque 
from Environment_Constants import (
    GRID_SIZE, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    WHITE, BLACK, GREEN, RED, BLUE,
    UP, DOWN, LEFT, RIGHT,
    REWARD_FOOD, REWARD_COLLISION, REWARD_STEP,
    FPS
)

class SnakeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._init_game_state()

    def _init_game_state(self):
        self.snake = deque()
        self.head = (self.grid_size // 2, self.grid_size // 2) 
        self.snake.append(self.head)
    
        for _ in range(2):
            self.snake.append((self.head[0], self.head[1] + (_ + 1))) 

        self.direction = UP
        self.score = 0
        self.food = self._place_food()
        self.game_over = False
        self.steps_since_food = 0

    def _place_food(self):
        while True:
            x = random.randrange(self.grid_size)
            y = random.randrange(self.grid_size)
            food_pos = (x, y)
            if food_pos not in self.snake:
                return food_pos

    def _get_observation(self):
      
        obs = np.zeros(11, dtype=np.float32)

        hx, hy = self.head

        if self.direction == UP:
            dir_straight = UP
            dir_right = RIGHT
            dir_left = LEFT
        elif self.direction == DOWN:
            dir_straight = DOWN
            dir_right = LEFT
            dir_left = RIGHT
        elif self.direction == LEFT:
            dir_straight = LEFT
            dir_right = UP
            dir_left = DOWN
        elif self.direction == RIGHT:
            dir_straight = RIGHT
            dir_right = DOWN
            dir_left = UP

        check_pos_straight = (hx + dir_straight[0], hy + dir_straight[1])
        check_pos_right = (hx + dir_right[0], hy + dir_right[1])
        check_pos_left = (hx + dir_left[0], hy + dir_left[1])

        def is_danger(pos):
            px, py = pos

            if not (0 <= px < self.grid_size and 0 <= py < self.grid_size):
                return True
            
            if pos in list(self.snake)[1:]:
                return True
            return False

        obs[0] = 1 if is_danger(check_pos_straight) else 0 
        obs[1] = 1 if is_danger(check_pos_right) else 0  
        obs[2] = 1 if is_danger(check_pos_left) else 0 

        
        fx, fy = self.food
        if fy < hy: obs[3] = 1
        if fy > hy: obs[4] = 1 
        if fx < hx: obs[5] = 1 
        if fx > hx: obs[6] = 1 

        if self.direction == UP: obs[7] = 1
        elif self.direction == DOWN: obs[8] = 1
        elif self.direction == LEFT: obs[9] = 1
        elif self.direction == RIGHT: obs[10] = 1

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_game_state()

        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def _get_info(self):
        return {"score": self.score, "snake_length": len(self.snake)}

    def step(self, action):
      
        current_dir_idx = [UP, DOWN, LEFT, RIGHT].index(self.direction)
        
        if action == 0: 
            new_direction = self.direction
        elif action == 1:
            if self.direction == UP: new_direction = RIGHT
            elif self.direction == DOWN: new_direction = LEFT
            elif self.direction == LEFT: new_direction = UP
            elif self.direction == RIGHT: new_direction = DOWN
        elif action == 2:
            if self.direction == UP: new_direction = LEFT
            elif self.direction == DOWN: new_direction = RIGHT
            elif self.direction == LEFT: new_direction = DOWN
            elif self.direction == RIGHT: new_direction = UP
        else:
            raise ValueError(f"Received invalid action={action} which is not part of the action space.")

        self.direction = new_direction

        hx, hy = self.head
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        reward = REWARD_STEP

        terminated = False
       
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            terminated = True
            reward = REWARD_COLLISION
        
        elif new_head in list(self.snake):
            terminated = True
            reward = REWARD_COLLISION

        if terminated:
            self.game_over = True
        else:
            self.snake.appendleft(new_head) 
            self.head = new_head 

            if new_head == self.food:
                self.score += 1
                reward = REWARD_FOOD
                self.food = self._place_food()
                self.steps_since_food = 0 
            else:
                self.snake.pop()
                self.steps_since_food += 1

            if self.steps_since_food > self.grid_size * self.grid_size * 2:
                 terminated = True
                 reward = REWARD_COLLISION 

        observation = self._get_observation()
        info = self._get_info()
        truncated = False 

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Snake AI Training")
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        if self.render_mode == 'human':
        
            self.window.fill(BLACK)

            pygame.draw.rect(self.window, RED, (self.food[0] * self.cell_size,
                                               self.food[1] * self.cell_size,
                                               self.cell_size, self.cell_size))

            for i, segment in enumerate(self.snake):
                color = BLUE if i == 0 else GREEN 
                pygame.draw.rect(self.window, color, (segment[0] * self.cell_size,
                                                      segment[1] * self.cell_size,
                                                      self.cell_size, self.cell_size))

            for x in range(0, self.screen_width, self.cell_size):
                pygame.draw.line(self.window, WHITE, (x, 0), (x, self.screen_height))
            for y in range(0, self.screen_height, self.cell_size):
                pygame.draw.line(self.window, WHITE, (0, y), (self.screen_width, y))

            font = pygame.font.Font(None, 25)
            text = font.render(f"Score: {self.score}", True, WHITE)
            self.window.blit(text, (5, 5))

            pygame.event.pump() 
            pygame.display.flip() 

            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":

            surf = pygame.Surface((self.screen_width, self.screen_height))
            surf.fill(BLACK)

            pygame.draw.rect(surf, RED, (self.food[0] * self.cell_size,
                                         self.food[1] * self.cell_size,
                                         self.cell_size, self.cell_size))
            
            for i, segment in enumerate(self.snake):
                color = BLUE if i == 0 else GREEN
                pygame.draw.rect(surf, color, (segment[0] * self.cell_size,
                                               segment[1] * self.cell_size,
                                               self.cell_size, self.cell_size))

            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
