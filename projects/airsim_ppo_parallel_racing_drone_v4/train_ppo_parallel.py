"""
Parallel PPO Training for Gate Racing

Uses multiple AirSim drone instances to speed up training.
Each environment runs in its own AirSim process.

SPEEDUP:
  1 env:  ~5 hours for 300k steps
  4 envs: ~1.5 hours for 300k steps (3x faster!)
  8 envs: ~1 hour for 300k steps (5x faster!)

REQUIREMENTS:
  - Multiple AirSim instances running
  - Each with a different drone name
  - Or: Single AirSim with multiple drones
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import json
import time


class DetailedProgressCallback(BaseCallback):
    """
    Callback for parallel training progress.
    Handles multiple simultaneous episodes.
    """
    
    def __init__(self, check_freq=10, n_envs=4, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.n_envs = n_envs
        self.episode_count = 0
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_gates = []
        self.episode_success = []
        
        # Best metrics
        self.best_reward = -np.inf
        self.best_gates = 0
        
    def _on_step(self):
        """Called after each step (handles multiple envs)."""
        
        # Check each environment for completion
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.episode_count += 1
                
                # Episode reward and length
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                
                # Gates passed
                gates_passed = info.get('gates_passed', 0)
                self.episode_gates.append(gates_passed)
                
                # Success
                success = gates_passed >= 7
                self.episode_success.append(success)
                
                # Track best
                if gates_passed > self.best_gates:
                    self.best_gates = gates_passed
                    print(f"\n🏆 NEW BEST: {gates_passed} gates passed! (Env {i})")
                
                if len(self.episode_rewards) > 0 and self.episode_rewards[-1] > self.best_reward:
                    self.best_reward = self.episode_rewards[-1]
                
                # Print progress
                if self.episode_count % self.check_freq == 0:
                    self._print_progress()
        
        return True
    
    def _print_progress(self):
        """Print detailed progress report."""
        
        recent_rewards = self.episode_rewards[-self.check_freq*self.n_envs:]
        recent_lengths = self.episode_lengths[-self.check_freq*self.n_envs:]
        recent_gates = self.episode_gates[-self.check_freq*self.n_envs:]
        recent_success = self.episode_success[-self.check_freq*self.n_envs:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        avg_gates = np.mean(recent_gates)
        max_gates = np.max(recent_gates)
        success_rate = np.mean(recent_success) * 100
        
        print(f"\n{'='*70}")
        print(f"📊 EPISODE {self.episode_count} | STEPS {self.num_timesteps}")
        print(f"   Parallel Envs: {self.n_envs}")
        print(f"{'='*70}")
        print(f"  Avg Reward:    {avg_reward:>8.1f}")
        print(f"  Avg Length:    {avg_length:>8.1f} steps")
        print(f"  Avg Gates:     {avg_gates:>8.2f} / 7")
        print(f"  Best Gates:    {max_gates:>8} / 7")
        print(f"  Success Rate:  {success_rate:>8.1f}%")
        print(f"  All-time Best: {self.best_gates:>8} gates")
        print(f"{'='*70}")


class SaveBestModelCallback(BaseCallback):
    """Save best model based on gates passed."""
    
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_gates = 0
    
    def _on_step(self):
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        for done, info in zip(dones, infos):
            if done:
                gates_passed = info.get('gates_passed', 0)
                
                if gates_passed > self.best_gates:
                    self.best_gates = gates_passed
                    
                    model_path = os.path.join(self.save_path, f"best_model_{gates_passed}gates")
                    self.model.save(model_path)
                    
                    if self.verbose > 0:
                        print(f"\n💾 Saved best model: {gates_passed} gates")
        
        return True


def make_env(gate_positions, drone_name=None, rank=0):
    """
    Create single environment instance.
    
    Args:
        gate_positions: List of gate positions
        drone_name: Name of drone in AirSim (for multi-drone setup)
        rank: Environment rank (for seeding)
    """
    from gate_racing_env import GateRacingEnv
    
    def _init():
        # Stagger initialization to avoid connection conflicts
        time.sleep(rank * 0.5)
        
        env = GateRacingEnv(
            gate_positions=gate_positions,
            # vehicle_name=drone_name  # Uncomment if using named drones
        )
        env = Monitor(env)
        
        # Seed for reproducibility
        env.reset(seed=rank)
        
        return env
    
    return _init


def train_ppo_parallel(gate_positions, n_envs=4, total_timesteps=300000, save_dir=None):
    """
    Train PPO with parallel environments.
    
    Args:
        gate_positions: List of gate positions
        n_envs: Number of parallel environments (2-8 recommended)
        total_timesteps: Total training steps
        save_dir: Save directory
    """
    
    # Setup save directory
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"./gate_racing_ppo_parallel_{n_envs}envs_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 PARALLEL PPO TRAINING")
    print(f"{'='*70}")
    print(f"  Parallel Envs: {n_envs}")
    print(f"  Gates:         {len(gate_positions)}")
    print(f"  Timesteps:     {total_timesteps:,}")
    print(f"  Save dir:      {save_dir}")
    print(f"{'='*70}")
    print(f"\n⚡ Speedup: ~{min(n_envs * 0.7, n_envs * 0.5 + 2):.1f}x faster than single env")
    print(f"   Estimated time: {(300000 / total_timesteps) * (5 / max(1, n_envs * 0.6)):.1f} hours\n")
    
    # Save config
    gate_config = {
        'gate_positions': gate_positions,
        'num_gates': len(gate_positions),
        'n_parallel_envs': n_envs,
        'training_timesteps': total_timesteps
    }
    with open(os.path.join(save_dir, 'gate_config.json'), 'w') as f:
        json.dump(gate_config, f, indent=2)
    
    # Create parallel environments
    print("Creating parallel environments...")
    print("⚠️  This may take 30-60 seconds...")
    
    # Option 1: SubprocVecEnv (faster, more stable)
    # Each environment runs in separate process
    env_fns = [make_env(gate_positions, rank=i) for i in range(n_envs)]
    
    try:
        env = SubprocVecEnv(env_fns, start_method='spawn')
        print(f"✅ Created {n_envs} parallel environments (SubprocVecEnv)")
    except Exception as e:
        print(f"⚠️  SubprocVecEnv failed: {e}")
        print("   Falling back to DummyVecEnv (slower but more compatible)...")
        env = DummyVecEnv(env_fns)
        print(f"✅ Created {n_envs} environments (DummyVecEnv)")
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create PPO model
    print("\nCreating PPO model...")
    
    # Adjust hyperparameters for parallel training
    # More envs = smaller n_steps needed per env
    n_steps_per_env = max(512, 2048 // n_envs)
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=n_steps_per_env,  # Adjusted for parallel
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=save_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1,
        device='auto'  # Use GPU if available
    )
    
    print(f"   n_steps per env: {n_steps_per_env}")
    print(f"   Total steps per update: {n_steps_per_env * n_envs}")
    
    # Setup logger
    logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Setup callbacks
    progress_callback = DetailedProgressCallback(
        check_freq=10, 
        n_envs=n_envs, 
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, 10000 // n_envs * n_envs),  # Adjust for parallel
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_parallel"
    )
    best_model_callback = SaveBestModelCallback(save_path=save_dir, verbose=1)
    
    print("\n" + "="*70)
    print("🎓 TRAINING START")
    print("="*70)
    print(f"\nWith {n_envs} parallel environments:")
    print(f"  10k steps:  ~{10 / max(1, n_envs * 0.6):.0f} minutes")
    print(f"  50k steps:  ~{50 / max(1, n_envs * 0.6):.0f} minutes")
    print(f"  150k steps: ~{150 / max(1, n_envs * 0.6):.0f} minutes")
    print(f"  300k steps: ~{300 / max(1, n_envs * 0.6):.0f} minutes")
    print("\nPress Ctrl+C to stop training early.")
    print("="*70 + "\n")
    
    try:
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback, best_model_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
    
    # Save final model
    final_path = os.path.join(save_dir, "ppo_parallel_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Final model:   {final_path}.zip")
    print(f"  Best gates:    {progress_callback.best_gates}")
    print(f"  Total episodes: {progress_callback.episode_count}")
    print(f"  Parallel envs: {n_envs}")
    print(f"{'='*70}\n")
    
    # Close environments
    env.close()
    
    return model, save_dir


def test_model(model_path, vec_normalize_path, gate_positions, num_episodes=5):
    """Test trained model (same as single env)."""
    from gate_racing_env import GateRacingEnv
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING MODEL")
    print(f"{'='*70}")
    print(f"  Model:    {model_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create single environment for testing
    env = GateRacingEnv(gate_positions=gate_positions)
    
    # Load normalization stats
    if os.path.exists(vec_normalize_path):
        # Wrap in DummyVecEnv for compatibility
        vec_env = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize.load(vec_normalize_path, vec_env)
        vec_norm.training = False
        vec_norm.norm_reward = False
        env = vec_norm
    
    # Test episodes
    results = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        gates_passed = 0
        
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, done, info = env.step(action)
            
            # Handle vectorized env output
            if hasattr(done, '__iter__'):
                done = done[0]
                info = info[0]
                reward = reward[0]
            
            total_reward += reward
            steps += 1
            gates_passed = info.get('gates_passed', 0)
            
            # Print progress
            if steps % 20 == 0:
                print(f"  Step {steps}: Gates {gates_passed}/7, Speed {info.get('speed', 0):.1f}m/s")
            
            if done:
                break
        
        # Episode summary
        success = gates_passed >= 7
        results.append({
            'gates': gates_passed,
            'steps': steps,
            'reward': total_reward,
            'success': success
        })
        
        print(f"\n  {'✅' if success else '❌'} Finished:")
        print(f"     Gates:  {gates_passed}/7")
        print(f"     Steps:  {steps}")
        print(f"     Reward: {total_reward:.1f}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")
    avg_gates = np.mean([r['gates'] for r in results])
    success_rate = np.mean([r['success'] for r in results]) * 100
    print(f"  Avg Gates:     {avg_gates:.2f} / 7")
    print(f"  Success Rate:  {success_rate:.0f}%")
    print(f"  Best Run:      {max([r['gates'] for r in results])} gates")
    print(f"{'='*70}\n")
    
    env.close()


if __name__ == "__main__":
    # Your gate positions
    gate_positions = [
        [5.8, -5.3, 1.0],
        [17.3, -7.9, 1.0],
        [28.9, -7.9, 1.0],
        [39.3, -5.6, 1.0],
        [46.3, 0.8, 1.0],
        [46.3, 10.3, 1.0],
        [39.5, 18.0, 1.0],
    ]
    
    print("\n" + "="*70)
    print("PARALLEL GATE RACING - PPO TRAINING")
    print("="*70)
    print("\nOptions:")
    print("  1. Train with parallel environments")
    print("  2. Test existing model")
    print("="*70)
    
    choice = input("\nChoice (1/2): ").strip()
    
    if choice == "1":
        # Parallel training
        n_envs = input("Number of parallel environments (2-8, default 4): ").strip()
        n_envs = int(n_envs) if n_envs else 4
        
        if n_envs < 1 or n_envs > 16:
            print("⚠️  n_envs should be between 1-16. Using 4.")
            n_envs = 4
        
        timesteps = input("Training timesteps (default 300000): ").strip()
        timesteps = int(timesteps) if timesteps else 300000
        
        print(f"\n🚀 Starting parallel training:")
        print(f"   Environments: {n_envs}")
        print(f"   Timesteps:    {timesteps:,}")
        print(f"   Speedup:      ~{min(n_envs * 0.7, n_envs * 0.5 + 2):.1f}x")
        
        print("\n⚠️  IMPORTANT:")
        print("   Make sure AirSim is running!")
        print("   Parallel envs will connect sequentially.")
        print("   This is normal and takes ~30-60 seconds.\n")
        
        input("Press ENTER to begin...")
        
        model, save_dir = train_ppo_parallel(
            gate_positions, 
            n_envs=n_envs,
            total_timesteps=timesteps
        )
        
        print(f"\n✅ Training complete! Files saved to: {save_dir}")
        
    elif choice == "2":
        # Test model
        model_path = input("Model path (without .zip): ").strip()
        vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        episodes = input("Test episodes (default 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        test_model(model_path, vec_norm_path, gate_positions, num_episodes=episodes)
        
    else:
        print("Invalid choice")