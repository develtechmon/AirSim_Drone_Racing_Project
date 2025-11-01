import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from drone_balloon_env import DroneBalloonEnv

# Register the custom environment
gym.register(
    id='DroneBalloon-v0',
    entry_point='drone_balloon_env:DroneBalloonEnv',
)


def train_drone():
    """Train the drone using PPO algorithm with VecNormalize"""
    
    print("Creating training environments...")
    # Create 4 parallel environments for faster training
    train_env = make_vec_env(
        'DroneBalloon-v0',
        n_envs=4, # create seperate copies of the environment
        seed=42 # ensure the environment start with same randomess pattern each time
    )
    
    # Wrap with monitor to track episode statistics
    train_env = VecMonitor(train_env)
    
    # ✨ NEW: Add VecNormalize for better performance
    train_env = VecNormalize(
        train_env,
        norm_obs=True,        # Normalize observations (drone/balloon positions)
        norm_reward=True,     # Normalize rewards
        clip_obs=10.0,        # Clip normalized observations
        clip_reward=10.0,     # Clip normalized rewards
        gamma=0.99            # Should match PPO's gamma
    )
    
    print("Creating evaluation environment...")
    # Separate environment for evaluation
    eval_env = make_vec_env(
        'DroneBalloon-v0',
        n_envs=1,
        seed=123
    )
    eval_env = VecMonitor(eval_env)
    
    # ✨ NEW: Also normalize eval environment (but it will sync stats from training)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=False  # Don't update stats during evaluation
    )
    
    # Callbacks for saving and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='drone_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    print("\nInitializing PPO agent...")
    print("=" * 50)
    
    # Create PPO model with tuned hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,          # Number of steps to run per update
        batch_size=64,         # Minibatch size
        n_epochs=10,           # Number of epochs per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clipping parameter
        ent_coef=0.01,         # Entropy coefficient (encourages exploration)
        vf_coef=0.5,           # Value function coefficient
        max_grad_norm=0.5,     # Gradient clipping
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("\nStarting training...")
    print("=" * 50)
    print("Training for 200,000 timesteps...")
    print("This might take 5-15 minutes depending on your hardware.")
    print()
    
    # Train the model
    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # ✨ NEW: Save both model AND normalization statistics
    print("\nSaving final model and normalization stats...")
    model.save("drone_ppo_final")
    train_env.save("vec_normalize.pkl")  # CRITICAL: Save normalization stats!
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Final model saved as: drone_ppo_final.zip")
    print(f"Normalization stats saved as: vec_normalize.pkl")
    print(f"Best model saved in: ./best_model/")
    print(f"Checkpoints saved in: ./checkpoints/")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    trained_model = train_drone()