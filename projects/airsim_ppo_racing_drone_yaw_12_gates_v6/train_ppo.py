"""
PPO Training for Gate Racing

This trains a drone to fly through gates in sequence using PPO.

TRAINING PHASES:
  0-50k steps:   Learning basic flight and gate approach
  50-150k steps: Learning to pass through gates
  150k+ steps:   Optimizing speed and completing full circuits

EXPECTED RESULTS:
  30k steps:  Approaches first gate consistently
  80k steps:  Passes 2-3 gates per episode
  150k steps: Completes 5+ gates, occasional full circuit
  300k steps: Reliably completes full circuit
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import json


class DetailedProgressCallback(BaseCallback):
    """
    Callback that tracks and displays detailed training progress.
    """
    
    def __init__(self, check_freq=10, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
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
        """Called after each step."""
        
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Get episode info
            info = self.locals['infos'][0]
            
            # Episode reward
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
            
            # Gates passed
            gates_passed = info.get('gates_passed', 0)
            self.episode_gates.append(gates_passed)
            
            # Success (completed all gates)
            success = gates_passed >= 12  # Adjust if different number of gates
            self.episode_success.append(success)
            
            # Track best
            if gates_passed > self.best_gates:
                self.best_gates = gates_passed
                print(f"\n🏆 NEW BEST: {gates_passed} gates passed!")
            
            if len(self.episode_rewards) > 0 and self.episode_rewards[-1] > self.best_reward:
                self.best_reward = self.episode_rewards[-1]
            
            # Print progress every N episodes
            if self.episode_count % self.check_freq == 0:
                self._print_progress()
        
        return True
    
    def _print_progress(self):
        """Print detailed progress report."""
        
        # Get recent stats
        recent_rewards = self.episode_rewards[-self.check_freq:]
        recent_lengths = self.episode_lengths[-self.check_freq:]
        recent_gates = self.episode_gates[-self.check_freq:]
        recent_success = self.episode_success[-self.check_freq:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        avg_gates = np.mean(recent_gates)
        max_gates = np.max(recent_gates)
        success_rate = np.mean(recent_success) * 100
        
        print(f"\n{'='*70}")
        print(f"📊 EPISODE {self.episode_count} | STEPS {self.num_timesteps}")
        print(f"{'='*70}")
        print(f"  Avg Reward:    {avg_reward:>8.1f}")
        print(f"  Avg Length:    {avg_length:>8.1f} steps")
        print(f"  Avg Gates:     {avg_gates:>8.2f} / 12")
        print(f"  Best Gates:    {max_gates:>8} / 12")
        print(f"  Success Rate:  {success_rate:>8.1f}%")
        print(f"  All-time Best: {self.best_gates:>8} gates")
        print(f"{'='*70}")


class SaveBestModelCallback(BaseCallback):
    """
    Callback to save the best model based on gates passed.
    """
    
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_gates = 0
    
    def _on_step(self):
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            gates_passed = info.get('gates_passed', 0)
            
            if gates_passed > self.best_gates:
                self.best_gates = gates_passed
                
                # Save model
                model_path = os.path.join(self.save_path, f"best_model_{gates_passed}gates")
                self.model.save(model_path)
                
                if self.verbose > 0:
                    print(f"\n💾 Saved best model: {gates_passed} gates")
        
        return True


def make_env(gate_positions):
    """Create and wrap environment."""
    from gate_racing_env import GateRacingEnv
    
    def _init():
        env = GateRacingEnv(gate_positions=gate_positions)
        env = Monitor(env)
        return env
    
    return _init


def train_ppo(gate_positions, total_timesteps=300000, save_dir=None):
    """
    Train PPO agent for gate racing.
    
    Args:
        gate_positions: List of [x, y, z] gate positions
        total_timesteps: Total training steps
        save_dir: Directory to save models and logs
    """
    
    # Setup save directory
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"./gate_racing_ppo_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 GATE RACING PPO TRAINING")
    print(f"{'='*70}")
    print(f"  Gates:        {len(gate_positions)}")
    print(f"  Timesteps:    {total_timesteps:,}")
    print(f"  Save dir:     {save_dir}")
    print(f"{'='*70}\n")
    
    # Save gate positions
    gate_config = {
        'gate_positions': gate_positions,
        'num_gates': len(gate_positions),
        'training_timesteps': total_timesteps
    }
    with open(os.path.join(save_dir, 'gate_config.json'), 'w') as f:
        json.dump(gate_config, f, indent=2)
    
    # Create environment
    print("Creating environment...")
    env = DummyVecEnv([make_env(gate_positions)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=save_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1
    )
    
    # Setup logger
    logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Setup callbacks
    progress_callback = DetailedProgressCallback(check_freq=10, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_checkpoint"
    )
    best_model_callback = SaveBestModelCallback(save_path=save_dir, verbose=1)
    
    print("\n" + "="*70)
    print("🎓 TRAINING START")
    print("="*70)
    print("\nExpected milestones:")
    print("  30k steps:  Learns to approach gates")
    print("  80k steps:  Passes 2-3 gates consistently")
    print("  150k steps: Completes 5+ gates")
    print("  300k steps: Reliably completes full circuit")
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
    final_path = os.path.join(save_dir, "ppo_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best gates:  {progress_callback.best_gates}")
    print(f"  Episodes:    {progress_callback.episode_count}")
    print(f"{'='*70}\n")
    
    return model, env, save_dir


def test_model(model_path, vec_normalize_path, gate_positions, num_episodes=5):
    """
    Test a trained model.
    
    Args:
        model_path: Path to saved model (without .zip)
        vec_normalize_path: Path to vec_normalize.pkl
        gate_positions: List of gate positions
        num_episodes: Number of test episodes
    """
    from gate_racing_env import GateRacingEnv
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING MODEL")
    print(f"{'='*70}")
    print(f"  Model:    {model_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = GateRacingEnv(gate_positions=gate_positions)
    
    # Load normalization stats
    if os.path.exists(vec_normalize_path):
        vec_norm = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: env]))
        vec_norm.training = False
        vec_norm.norm_reward = False
    
    # Test episodes
    results = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        gates_passed = 0
        
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            gates_passed = info['gates_passed']
            
            # Print progress
            if steps % 20 == 0:
                print(f"  Step {steps}: Gates {gates_passed}/12, Speed {info['speed']:.1f}m/s")
            
            if terminated or truncated:
                break
        
        # Episode summary
        success = gates_passed >= 12
        results.append({
            'gates': gates_passed,
            'steps': steps,
            'reward': total_reward,
            'success': success
        })
        
        print(f"\n  {'✅' if success else '❌'} Finished:")
        print(f"     Gates:  {gates_passed}/12")
        print(f"     Steps:  {steps}")
        print(f"     Reward: {total_reward:.1f}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")
    avg_gates = np.mean([r['gates'] for r in results])
    success_rate = np.mean([r['success'] for r in results]) * 100
    print(f"  Avg Gates:     {avg_gates:.2f} / 12")
    print(f"  Success Rate:  {success_rate:.0f}%")
    print(f"  Best Run:      {max([r['gates'] for r in results])} gates")
    print(f"{'='*70}\n")
    
    env.close()


if __name__ == "__main__":
    # Your gate positions
    gate_positions = [
        # [5.8, -5.3, -0.5],
        # [17.3, -7.9, 1.0],
        # [28.9, -7.9, 1.1],
        # [39.3, -5.6, 1.3],
        # [46.3, 0.8, 1.1],
        # [46.3, 10.3, 0.7],
        # [39.5, 18.0, 0.8],
        
        # [5.8, -5.3,  1.0],
        # [17.3, -7.9, 1.0],
        # [28.9, -7.9, 1.0],
        # [39.3, -5.6, 1.0],
        # [46.3, 0.8,  1.0],
        # [46.3, 10.3, 1.0],
        # [39.5, 18.0, 1.0],
        
        [9.1, -5.2, 1.0],  # Gate 0
        [18.9, -7.8, 1.0],  # Gate 1
        [29.8, -7.8, 1.0],  # Gate 2
        [38.8, -4.8, 1.0],  # Gate 3
        [45.3, 2.5, 1.0],  # Gate 4
        [45.2, 11.9, 1.0],  # Gate 5
        [38.1, 19.3, 1.0],  # Gate 6
        [29.1, 22.4, 1.1],  # Gate 7
        [17.8, 22.3, 1.0],  # Gate 8
        [7.2, 18.5, 0.7],  # Gate 9
        [0.8, 10.7, 0.8],  # Gate 10
        [-1.0, 1.7, 0.8],  # Gate 11
    ]
    
    print("\n" + "="*70)
    print("GATE RACING - PPO TRAINING")
    print("="*70)
    print("\nOptions:")
    print("  1. Train new model")
    print("  2. Test existing model")
    print("  3. Continue training existing model")
    print("="*70)
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        # New training
        timesteps = input("Training timesteps (default 300000): ").strip()
        timesteps = int(timesteps) if timesteps else 300000
        
        print(f"\n🚀 Starting training with {timesteps:,} timesteps...")
        print("This will take several hours. Progress is saved automatically.")
        
        input("Press ENTER to begin...")
        
        model, env, save_dir = train_ppo(gate_positions, total_timesteps=timesteps)
        
        env.close()
        
        print(f"\n✅ Training complete! Files saved to: {save_dir}")
        
    elif choice == "2":
        # Test model
        model_path = input("Model path (without .zip): ").strip()
        vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        episodes = input("Test episodes (default 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        test_model(model_path, vec_norm_path, gate_positions, num_episodes=episodes)
        
    elif choice == "3":
        # Continue training
        model_path = input("Model path to continue (without .zip): ").strip()
        vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        save_dir = input("Save directory: ").strip()
        timesteps = input("Additional timesteps (default 100000): ").strip()
        timesteps = int(timesteps) if timesteps else 100000
        
        print(f"\n📖 Loading model from {model_path}...")
        
        # Load model and environment
        model = PPO.load(model_path)
        env = DummyVecEnv([make_env(gate_positions)])
        env = VecNormalize.load(vec_norm_path, env)
        model.set_env(env)
        
        # Setup callbacks
        progress_callback = DetailedProgressCallback(check_freq=10)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(save_dir, "checkpoints"),
            name_prefix="ppo_continued"
        )
        
        print(f"🚀 Continuing training for {timesteps:,} more steps...")
        
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=[progress_callback, checkpoint_callback],
                progress_bar=True,
                reset_num_timesteps=False
            )
        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted")
        
        # Save
        final_path = os.path.join(save_dir, "ppo_continued")
        model.save(final_path)
        env.save(os.path.join(save_dir, "vec_normalize_continued.pkl"))
        
        print(f"\n✅ Continued training complete! Saved to {final_path}")
        
        env.close()
    
    else:
        print("Invalid choice")