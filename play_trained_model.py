from stable_baselines3 import DQN, PPO
from snake_gym_env import SnakeGameEnv
import time

def play_model(model_path, model_type=DQN):
    # Load the trained model
    model = model_type.load(model_path)
    
    # Create environment
    env = SnakeGameEnv()
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        # Get model's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Add small delay to make visualization easier to follow
        time.sleep(0.1)
    
    print(f"Game Over! Final Score: {info['score']}")
    return total_reward

if __name__ == "__main__":
    # Play with DQN model
    print("\nPlaying DQN model...")
    play_model("dqn_snake", DQN)
    
    # Play with PPO model
    print("\nPlaying PPO model...")
    play_model("ppo_snake", PPO) 