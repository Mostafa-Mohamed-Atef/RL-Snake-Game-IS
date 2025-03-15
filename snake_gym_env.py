import gymnasium as gym
import numpy as np
from gymnasium import spaces
from snake_game import SnakeGame, Direction, Point, BLOCK_SIZE

class SnakeGameEnv(gym.Env):
    """Wrapper around PyGame Snake game to make it compatible with gymnasium"""
    def __init__(self, render_mode=None):
        super().__init__()
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(4)  # UP, RIGHT, DOWN, LEFT
        
        # Observation space will be a binary grid representing:
        # Channel 1: Snake body
        # Channel 2: Snake head
        # Channel 3: Food
        grid_size = (self.game.h // BLOCK_SIZE, self.game.w // BLOCK_SIZE)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size[0], grid_size[1], 3),
            dtype=np.float32
        )
        self.render_mode = render_mode

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the game
        self.game = SnakeGame()
        return self._get_obs(), {}

    def _get_obs(self):
        grid_size = (self.game.h // BLOCK_SIZE, self.game.w // BLOCK_SIZE)
        obs = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.float32)

        # Mark snake body
        for point in self.game.snake:
            x, y = int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE)
            obs[y, x, 0] = 1  # Snake body in first channel

        # Mark snake head
        head_x, head_y = int(self.game.head.x // BLOCK_SIZE), int(self.game.head.y // BLOCK_SIZE)
        obs[head_y, head_x, 1] = 1  # Snake head in second channel

        # Mark food
        food_x, food_y = int(self.game.food.x // BLOCK_SIZE), int(self.game.food.y // BLOCK_SIZE)
        obs[food_y, food_x, 2] = 1  # Food in third channel

        return obs

    def step(self, action):
        # Convert action (0,1,2,3) to Direction enum
        direction_map = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }
        self.game.direction = direction_map[action]
        
        # Take step in game
        game_over, score = self.game.play_step()
        
        # Calculate reward
        if game_over:
            reward = -10
        elif score > self.game.score - 1:  # If score increased (food eaten)
            reward = 10
        else:
            reward = -0.1  # Small negative reward for each step
            
            # Add reward for getting closer to food
            head = self.game.head
            food = self.game.food
            distance_to_food = abs(head.x - food.x) + abs(head.y - food.y)
            reward += 1 / (distance_to_food + 1)  # Reward inversely proportional to distance
        
        return self._get_obs(), reward, game_over, False, {"score": score}

    def render(self):
        # PyGame already handles rendering
        pass

    def close(self):
        self.game.display.quit() 