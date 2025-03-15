import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from snake_gym_env import SnakeGameEnv

class SnakeEnv(gym.Env):
    """Custom Snake Environment that follows gym interface"""
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1,
                                         shape=(self.grid_size, self.grid_size, 3),
                                         dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.snake_pos = [(self.grid_size//2, self.grid_size//2)]
        self.snake_direction = 1  
        self.food_pos = self._place_food()
        self.steps = 0
        self.max_steps = 100
        return self._get_obs(), {}
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size-1), 
                   random.randint(0, self.grid_size-1))
            if food not in self.snake_pos:
                return food
    
    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)        
        for x, y in self.snake_pos:
            obs[x, y, 0] = 1
        head_x, head_y = self.snake_pos[0]
        obs[head_x, head_y, 1] = 1
        obs[self.food_pos[0], self.food_pos[1], 2] = 1
        return obs
    
    def step(self, action):
        self.steps += 1
        head_x, head_y = self.snake_pos[0]
        if action == 0:  # up
            new_head = (head_x - 1, head_y)
        elif action == 1:  # right
            new_head = (head_x, head_y + 1)
        elif action == 2:  # down
            new_head = (head_x + 1, head_y)
        else:  # left
            new_head = (head_x, head_y - 1)
        
        done = False
        reward = -0.1  
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake_pos):
            done = True
            reward = -10  # Larger negative reward for collisions
        else:
            # Move snake
            self.snake_pos.insert(0, new_head)
            
            # Check if food is eaten
            if new_head == self.food_pos:
                reward = 10  # Larger reward for eating food
                self.food_pos = self._place_food()
            else:
                self.snake_pos.pop()
                reward += 0.1  # Small positive reward for surviving each step
        
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, False, {}

# Training function
def train_and_evaluate(model_class, env, total_timesteps, model_name):
    model = model_class("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model.save(model_name)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return model, mean_reward, std_reward

def main():
    # Create and wrap the environment
    env = SnakeGameEnv()
    env = Monitor(env)
    
    print("Training DQN...")
    dqn_model, dqn_mean_reward, dqn_std_reward = train_and_evaluate(
        DQN, env, total_timesteps=200000, model_name="dqn_snake"
    )
    
    print("Training PPO...")
    ppo_model, ppo_mean_reward, ppo_std_reward = train_and_evaluate(
        PPO, env, total_timesteps=200000, model_name="ppo_snake"
    )
    
    print("\nResults:")
    print(f"DQN - Mean reward: {dqn_mean_reward:.2f} +/- {dqn_std_reward:.2f}")
    print(f"PPO - Mean reward: {ppo_mean_reward:.2f} +/- {ppo_std_reward:.2f}")

if __name__ == "__main__":
    main()