import gymnasium as gym
from stable_baselines3 import PPO
from drone_balloon_env import DroneBalloonEnv
import time

# Register the environment
gym.register(
    id='DroneBalloon-v0',
    entry_point='drone_balloon_env:DroneBalloonEnv',
)


def test_trained_drone(model_path="drone_ppo_final.zip", num_episodes=5):
    """Test the trained drone and visualize with pygame"""
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    print("Creating environment with rendering...")
    env = DroneBalloonEnv(render_mode="human")
    
    print(f"\nTesting for {num_episodes} episodes...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while True:
            # Predict action using trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Small delay to make visualization watchable
            time.sleep(0.03)
            
            if terminated or truncated:
                print(f"  Score: {info['score']}/{env.max_score}")
                print(f"  Steps: {steps}")
                print(f"  Total Reward: {episode_reward:.2f}")
                
                if info['score'] >= env.max_score:
                    print("  Status: ✓ SUCCESS!")
                else:
                    print("  Status: ✗ Timeout")
                
                time.sleep(1)  # Pause between episodes
                break
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    env.close()


def test_random_agent(num_episodes=3):
    """Test with random actions for comparison"""
    print("Testing RANDOM agent for comparison...")
    print("=" * 50)
    
    env = DroneBalloonEnv(render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1} (RANDOM)")
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            time.sleep(0.03)
            
            if terminated or truncated:
                print(f"  Score: {info['score']}/{env.max_score}")
                print(f"  Steps: {steps}")
                print(f"  Total Reward: {episode_reward:.2f}")
                time.sleep(1)
                break
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--random":
        test_random_agent()
    else:
        # Test trained agent
        test_trained_drone(
            model_path="best_model/best_model.zip",  # or "drone_ppo_final.zip"
            num_episodes=5
        )