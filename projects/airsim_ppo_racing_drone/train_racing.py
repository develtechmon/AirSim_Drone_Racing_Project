"""
PPO Training Script for Circular Gate Racing

This uses stable-baselines3's PPO implementation.
Think of PPO like a coach that:
1. Lets the drone try different strategies
2. Keeps track of what works and what doesn't
3. Gradually improves the drone's "policy" (decision-making)
4. But doesn't make huge changes that could break what it already learned

That's the "Proximal" part - keeping policy updates close to the previous policy.
"""

import os
import numpy as np
from racing_env import CircularGateEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch


def make_env(gate_positions):
    """
    Create and wrap the environment.
    
    The Monitor wrapper tracks episode statistics.
    Think of it like a fitness tracker for your drone.
    """
    def _init():
        env = CircularGateEnv(gate_positions=gate_positions)
        env = Monitor(env)
        return env
    return _init


def train_ppo_agent():
    """
    Train the PPO agent to fly through gates in a circle.
    
    This is the main training loop - like sending your dog to obedience school.
    """
    
    # Check AirSim connection first
    print("=" * 60)
    print("CHECKING AIRSIM CONNECTION...")
    print("=" * 60)
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ AirSim connected successfully!")
    except Exception as e:
        print(f"❌ Cannot connect to AirSim: {e}")
        print("\nMake sure:")
        print("  1. AirSim Drone Racing Lab is running")
        print("  2. The simulator is not paused")
        print("  3. No firewall is blocking connections")
        return None, None
    
    print("\n" + "=" * 60)
    print("CONFIGURING TRAINING...")
    print("=" * 60)
    
    # Define gate positions in a circle
    # You should adjust these based on your AirSim environment
    num_gates = 4
    radius = 15.0
    gate_positions = []
    
    for i in range(num_gates):
        angle = (2 * np.pi * i) / num_gates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = -5.0  # Adjust height as needed
        gate_positions.append([x, y, z])
    
    print(f"\nGate Configuration:")
    print(f"  Number of gates: {num_gates}")
    print(f"  Circle radius: {radius}m")
    print(f"  Gate height: {gate_positions[0][2]}m")
    print(f"\nGate positions:")
    for i, pos in enumerate(gate_positions):
        print(f"  Gate {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
    
    # Create training environment
    # DummyVecEnv wraps the environment for stable-baselines3
    # VecNormalize normalizes observations and rewards (makes learning more stable)
    print(f"\nCreating training environment...")
    env = DummyVecEnv([make_env(gate_positions)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment (for testing during training)
    print(f"Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(gate_positions)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create directories for saving
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    print(f"✅ Directories created: ./models/, ./logs/")
    
    # Callbacks for saving and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path="./models/",
        name_prefix="ppo_drone_racing"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,  # Evaluate every 5k steps
        deterministic=True,
        render=False
    )
    
    # PPO Hyperparameters
    # These are carefully tuned - don't change randomly!
    """
    Think of hyperparameters like recipe ingredients:
    - learning_rate: How big of steps to take when learning (too big = unstable, too small = slow)
    - n_steps: How much experience to gather before updating (like batch size in cooking)
    - batch_size: How many samples to learn from at once
    - n_epochs: How many times to go over the same data when updating
    - gamma: How much to care about future rewards (0.99 = care a lot about long-term)
    - ent_coef: Encourages exploration (like curiosity in a dog)
    - clip_range: How much we allow the policy to change (the "Proximal" in PPO)
    """
    
    model = PPO(
        "MlpPolicy",  # Multi-Layer Perceptron (standard neural network)
        env,
        learning_rate=3e-4,  # Standard learning rate
        n_steps=2048,  # Collect 2048 steps before update
        batch_size=64,  # Mini-batch size for optimization
        n_epochs=10,  # How many passes over the data
        gamma=0.99,  # Discount factor (value future rewards highly)
        gae_lambda=0.95,  # Generalized Advantage Estimation parameter
        clip_range=0.2,  # PPO clipping parameter (standard value)
        ent_coef=0.01,  # Entropy coefficient (encourages exploration)
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping (prevents instability)
        use_sde=False,  # State-dependent exploration (we don't need it)
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Network architecture
            # pi = policy network (decides actions)
            # vf = value network (estimates future rewards)
            # [256, 256] = two hidden layers with 256 neurons each
        ),
        verbose=1,  # Print training progress
        tensorboard_log="./logs/tensorboard/",  # For TensorBoard visualization
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nTraining on device: {model.device}")
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Expected duration: ~{total_timesteps / 50000:.1f} hours (on decent GPU)")
    print("\nMonitor training:")
    print(f"  TensorBoard: tensorboard --logdir ./logs/tensorboard")
    print(f"  Then open: http://localhost:6006")
    print("\nPress Ctrl+C to stop training at any time (progress will be saved)")
    print("=" * 60 + "\n")
    
    # Train the model with error handling
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        training_completed = True
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        training_completed = False
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        training_completed = False
    
    # Save final model
    print("\n" + "=" * 60)
    if training_completed:
        print("TRAINING COMPLETED!")
    else:
        print("SAVING PROGRESS...")
    print("=" * 60)
    
    model.save("./models/ppo_drone_racing_final")
    env.save("./models/vec_normalize.pkl")
    
    print(f"✅ Model saved to: ./models/ppo_drone_racing_final.zip")
    print(f"✅ Normalization saved to: ./models/vec_normalize.pkl")
    
    if training_completed:
        print(f"\n🎉 Training complete! Test your agent:")
        print(f"   python train_ppo.py --mode test")
    else:
        print(f"\n💡 You can resume training by loading the saved model")
    
    print("=" * 60)
    
    return model, env


def test_trained_agent(model_path="./models/ppo_drone_racing_final", num_episodes=5):
    """
    Test the trained agent to see how well it performs.
    
    This is like the final exam after training school.
    """
    
    # Define gate positions (same as training)
    num_gates = 4
    radius = 15.0
    gate_positions = []
    
    for i in range(num_gates):
        angle = (2 * np.pi * i) / num_gates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = -5.0
        gate_positions.append([x, y, z])
    
    print("=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Model not found: {model_path}.zip")
        print("\nTrain a model first:")
        print("  python train_ppo.py --mode train")
        return
    
    # Create environment
    env = CircularGateEnv(gate_positions=gate_positions)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Load normalization parameters
    env = DummyVecEnv([lambda: Monitor(env)])
    
    if os.path.exists("./models/vec_normalize.pkl"):
        print(f"Loading normalization from ./models/vec_normalize.pkl...")
        env = VecNormalize.load("./models/vec_normalize.pkl", env)
        env.training = False  # Don't update normalization during testing
        env.norm_reward = False
    else:
        print("⚠️  No normalization file found, using unnormalized observations")
    
    print("\n" + "=" * 60)
    print("RUNNING TEST EPISODES")
    print("=" * 60 + "\n")
    
    total_success = 0
    total_gates = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        gates_passed = 0
        
        while not (terminated or truncated):
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward[0]
            steps += 1
            gates_passed = info[0]['gates_passed']
            
            if terminated or truncated:
                break
        
        success = gates_passed >= num_gates
        total_success += int(success)
        total_gates += gates_passed
        total_reward += episode_reward
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Gates Passed: {gates_passed}/{num_gates}")
        print(f"  Steps: {steps}")
        print(f"  Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print("-" * 60)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {total_success}/{num_episodes} ({100*total_success/num_episodes:.1f}%)")
    print(f"Average Gates Passed: {total_gates/num_episodes:.1f}/{num_gates}")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    
    if total_success == num_episodes:
        print("\n🏆 Perfect! All test episodes completed successfully!")
    elif total_success >= num_episodes * 0.8:
        print("\n🎉 Great! Agent is performing well!")
    elif total_success >= num_episodes * 0.5:
        print("\n👍 Good progress, but could be better. Consider more training.")
    else:
        print("\n🤔 Agent needs more training. Current performance is suboptimal.")
    
    print("=" * 60)
    
    env.close()


def resume_training(model_path="./models/ppo_drone_racing_final", additional_timesteps=500_000):
    """
    Resume training from a checkpoint.
    
    Useful if training was interrupted or you want to train longer.
    """
    
    # Define gate positions (same as training)
    num_gates = 4
    radius = 15.0
    gate_positions = []
    
    for i in range(num_gates):
        angle = (2 * np.pi * i) / num_gates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = -5.0
        gate_positions.append([x, y, z])
    
    print("=" * 60)
    print("RESUMING TRAINING")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Model not found: {model_path}.zip")
        print("\nNo checkpoint to resume from. Start fresh training:")
        print("  python train_ppo.py --mode train")
        return
    
    # Load existing model
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    print(f"Creating environment...")
    env = DummyVecEnv([make_env(gate_positions)])
    
    if os.path.exists("./models/vec_normalize.pkl"):
        print(f"Loading normalization...")
        env = VecNormalize.load("./models/vec_normalize.pkl", env)
    else:
        print("⚠️  No normalization file found, creating new normalization")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Set the environment in the model
    model.set_env(env)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="ppo_drone_racing_resumed"
    )
    
    eval_env = DummyVecEnv([make_env(gate_positions)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    print(f"\n" + "=" * 60)
    print("CONTINUING TRAINING")
    print("=" * 60)
    print(f"Additional timesteps: {additional_timesteps:,}")
    print(f"Press Ctrl+C to stop at any time")
    print("=" * 60 + "\n")
    
    # Continue training
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False  # Don't reset step counter
        )
        print("\n✅ Training completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
    
    # Save
    model.save("./models/ppo_drone_racing_final")
    env.save("./models/vec_normalize.pkl")
    
    print(f"✅ Updated model saved to: ./models/ppo_drone_racing_final.zip")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test PPO drone racing agent")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "resume"],
                       help="Whether to train, test, or resume training")
    parser.add_argument("--model-path", type=str, default="./models/ppo_drone_racing_final",
                       help="Path to trained model (for testing or resuming)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Additional timesteps for resume mode (default: 500k)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        model, env = train_ppo_agent()
    elif args.mode == "test":
        test_trained_agent(model_path=args.model_path)
    elif args.mode == "resume":
        resume_training(model_path=args.model_path, additional_timesteps=args.timesteps)