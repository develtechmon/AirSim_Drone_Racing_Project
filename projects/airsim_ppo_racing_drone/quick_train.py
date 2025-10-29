"""
PPO Training Script for Circular Gate Racing

This uses Stable-Baselines3 for PPO implementation.
Includes curriculum learning to progressively teach the drone.

ANALOGY: Learning to drive
- Stage 0: Learn to move and find any gate (basic motor skills)
- Stage 1: Learn to go through gates in sequence (following a path)
- Stage 2: Master the circular trajectory (precision driving)
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime


class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning.
    
    ANALOGY: Like a video game that unlocks harder levels as you progress.
    - Level 1: Just fly through any gate (easy)
    - Level 2: Fly through gates in order (medium)  
    - Level 3: Follow the exact circular path (hard)
    """
    
    def __init__(self, stage_thresholds, verbose=0):
        """
        Args:
            stage_thresholds: Dict mapping {stage: (success_rate, num_episodes)}
                             e.g., {0: (0.3, 100), 1: (0.5, 200)}
        """
        super(CurriculumCallback, self).__init__(verbose)
        self.stage_thresholds = stage_thresholds
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_gates_passed = []
        self.current_stage = 0
        self.episodes_in_stage = 0
    
    def _on_step(self):
        # Check if episode ended
        if self.locals.get('dones')[0]:
            # Get episode info
            info = self.locals['infos'][0]
            gates_passed = info.get('gates_passed', 0)
            
            self.episode_gates_passed.append(gates_passed)
            self.episodes_in_stage += 1
            
            # Check if we should advance curriculum
            if self.episodes_in_stage >= 50:  # Evaluate every 50 episodes
                recent_gates = self.episode_gates_passed[-50:]
                avg_gates = np.mean(recent_gates)
                success_rate = np.mean([g >= 7 for g in recent_gates])
                
                if self.verbose > 0:
                    print(f"\n--- Curriculum Stage {self.current_stage} ---")
                    print(f"Episodes in stage: {self.episodes_in_stage}")
                    print(f"Avg gates passed: {avg_gates:.2f}/7")
                    print(f"Success rate: {success_rate:.1%}")
                
                # Check advancement criteria
                if self.current_stage == 0:
                    # Stage 0 → 1: Can pass at least 2 gates consistently
                    if avg_gates >= 2.0 and self.episodes_in_stage >= 100:
                        self.current_stage = 1
                        self.episodes_in_stage = 0
                        print(f"\n🎓 ADVANCING TO STAGE 1! (Gate sequencing)")
                        # Update environment
                        self.training_env.env_method('set_curriculum_stage', 1)
                
                elif self.current_stage == 1:
                    # Stage 1 → 2: Can complete ~4+ gates in sequence
                    if avg_gates >= 4.0 and self.episodes_in_stage >= 200:
                        self.current_stage = 2
                        self.episodes_in_stage = 0
                        print(f"\n🎓 ADVANCING TO STAGE 2! (Circular precision)")
                        self.training_env.env_method('set_curriculum_stage', 2)
        
        return True


class ProgressCallback(BaseCallback):
    """
    Callback to log training progress.
    
    Prints nice stats every N episodes so you know training is working.
    """
    
    def __init__(self, check_freq=10, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_gates = []
    
    def _on_step(self):
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Get episode stats
            info = self.locals['infos'][0]
            
            # Try to get episode reward from monitor wrapper
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
            
            gates_passed = info.get('gates_passed', 0)
            self.episode_gates.append(gates_passed)
            
            # Log every N episodes
            if self.episode_count % self.check_freq == 0:
                recent_rewards = self.episode_rewards[-self.check_freq:]
                recent_lengths = self.episode_lengths[-self.check_freq:]
                recent_gates = self.episode_gates[-self.check_freq:]
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_length = np.mean(recent_lengths) if recent_lengths else 0
                avg_gates = np.mean(recent_gates)
                max_gates = np.max(recent_gates)
                
                print(f"\n📊 Episode {self.episode_count}")
                print(f"   Avg Reward: {avg_reward:.2f}")
                print(f"   Avg Length: {avg_length:.0f} steps")
                print(f"   Avg Gates: {avg_gates:.2f}/7")
                print(f"   Best Gates: {max_gates}/7")
                print(f"   Total Steps: {self.num_timesteps}")
        
        return True


def make_env(gate_positions, curriculum_stage=0):
    """
    Create and wrap environment.
    
    Monitor wrapper tracks episode statistics automatically.
    """
    from racing_env import CircularGateEnv
    
    def _init():
        env = CircularGateEnv(
            gate_positions=gate_positions,
            curriculum_stage=curriculum_stage
        )
        env = Monitor(env)  # Wrap with Monitor to track episodes
        return env
    
    return _init


def train_ppo(gate_positions, total_timesteps=500000, curriculum=True):
    """
    Train PPO agent to fly through circular gates.
    
    Args:
        gate_positions: List of 7 gate positions
        total_timesteps: How many steps to train (500k = ~6 hours on good GPU)
        curriculum: Whether to use curriculum learning
    
    HYPERPARAMETER CHOICES (and why):
    - learning_rate=3e-4: Standard for PPO, not too fast/slow
    - n_steps=2048: Collect 2048 steps before updating (balance data/computation)
    - batch_size=64: Mini-batch size for gradient updates
    - n_epochs=10: How many times to reuse each batch of data
    - gamma=0.99: Discount factor (value future rewards at 99% of present)
    - gae_lambda=0.95: Generalized Advantage Estimation smoothing
    - clip_range=0.2: PPO clipping (prevents too-large policy updates)
    - ent_coef=0.01: Entropy bonus (encourages exploration)
    """
    
    # Create log directory
    log_dir = f"./ppo_drone_racing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📁 Logs will be saved to: {log_dir}")
    
    # Create environment
    env = DummyVecEnv([make_env(gate_positions, curriculum_stage=0)])
    
    # Normalize observations (IMPORTANT for PPO)
    # ANALOGY: Like standardizing test scores (mean=0, std=1)
    # Makes learning more stable
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",  # Multi-Layer Perceptron (standard neural network)
        env=env,
        learning_rate=5e-4,  # Increased from 3e-4 - learn faster!
        n_steps=2048,  # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.02,  # Increased from 0.01 - more exploration!
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        use_sde=False,  # State-dependent exploration (we don't need it)
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 2-layer network, 256 units
        ),
        verbose=1
    )
    
    print("\n🤖 PPO Model Created")
    print(f"   Policy: MlpPolicy with [256, 256] hidden layers")
    print(f"   Learning rate: 3e-4")
    print(f"   Batch size: 64")
    print(f"   Training steps: {total_timesteps:,}")
    
    # Set up callbacks
    callbacks = []
    
    # Progress logging
    progress_callback = ProgressCallback(check_freq=10, verbose=1)
    callbacks.append(progress_callback)
    
    # Curriculum learning
    if curriculum:
        curriculum_callback = CurriculumCallback(
            stage_thresholds={
                0: (0.3, 100),  # Stage 0: 30% success after 100 episodes
                1: (0.5, 200),  # Stage 1: 50% success after 200 episodes
            },
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    # Checkpoint saving
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=log_dir,
        name_prefix="ppo_drone"
    )
    callbacks.append(checkpoint_callback)
    
    # Train!
    print("\n🚀 Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    
    # Save final model
    final_path = os.path.join(log_dir, "ppo_drone_final")
    model.save(final_path)
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {final_path}")
    print(f"   Normalization stats: {log_dir}/vec_normalize.pkl")
    
    return model, env


def test_model(model_path, gate_positions, episodes=5):
    """
    Test a trained model.
    
    Args:
        model_path: Path to saved model (without .zip)
        gate_positions: List of 7 gate positions
        episodes: Number of test episodes
    """
    from racing_env import CircularGateEnv
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment (no normalization for clarity)
    env = CircularGateEnv(gate_positions=gate_positions, curriculum_stage=2)
    
    print(f"\n🧪 Testing model for {episodes} episodes...\n")
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"Episode {ep + 1}:")
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 50 == 0:
                print(f"  Step {steps}: Gates {info['gates_passed']}/7, Reward: {total_reward:.1f}")
            
            if terminated or truncated:
                print(f"  ✅ Episode ended: {info['gates_passed']}/7 gates in {steps} steps")
                print(f"     Total reward: {total_reward:.2f}")
                if info['crashed']:
                    print(f"     Status: CRASHED 💥")
                elif info['gates_passed'] >= 7:
                    print(f"     Status: COMPLETED! 🏆")
                else:
                    print(f"     Status: Timeout ⏱️")
                break
        
        print()
    
    env.close()


if __name__ == "__main__":
    # Your 7 gate positions (these should match your AirSim environment)
    # Arrange in a circle, radius = 15 meters
    gate_positions = [
        [15, 0, -5],         # Gate 0 (start) - facing +X
        [10.6, 10.6, -5],    # Gate 1 - 45°
        [0, 15, -5],         # Gate 2 - 90°
        [-10.6, 10.6, -5],   # Gate 3 - 135°
        [-15, 0, -5],        # Gate 4 - 180°
        [-10.6, -10.6, -5],  # Gate 5 - 225°
        [0, -15, -5],        # Gate 6 - 270°
    ]
    
    print("="*60)
    print("PPO TRAINING FOR CIRCULAR GATE RACING")
    print("="*60)
    print("\nGate Configuration:")
    for i, pos in enumerate(gate_positions):
        print(f"  Gate {i}: {pos}")
    print()
    
    # Choose mode
    print("Choose mode:")
    print("1. Train new model")
    print("2. Test existing model")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Training parameters
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        
        timesteps = input("Total timesteps (default 500000): ").strip()
        timesteps = int(timesteps) if timesteps else 500000
        
        use_curriculum = input("Use curriculum learning? (y/n, default y): ").strip().lower()
        use_curriculum = use_curriculum != 'n'
        
        print(f"\nStarting training with:")
        print(f"  - Total timesteps: {timesteps:,}")
        print(f"  - Curriculum learning: {'Yes' if use_curriculum else 'No'}")
        print(f"  - Gate positions: 7 gates in circle")
        print("\nPress Ctrl+C to stop training early\n")
        
        input("Press Enter to start training...")
        
        model, env = train_ppo(
            gate_positions=gate_positions,
            total_timesteps=timesteps,
            curriculum=use_curriculum
        )
        
        env.close()
        
    elif choice == "2":
        # Testing
        model_path = input("Enter model path (without .zip extension): ").strip()
        episodes = input("Number of test episodes (default 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        test_model(model_path, gate_positions, episodes)
    
    else:
        print("Invalid choice!")
    
    print("\n🎯 Done!")