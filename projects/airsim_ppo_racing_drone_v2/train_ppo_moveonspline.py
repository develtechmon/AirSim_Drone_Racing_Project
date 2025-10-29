"""
PPO Training for Position Control Gate Racing

This uses standard AirSim's moveToPositionAsync (no special packages needed!)
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime


class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=10, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_gates = []
    
    def _on_step(self):
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.episode_rewards.append(ep_reward)
            
            gates_passed = info.get('gates_passed', 0)
            self.episode_gates.append(gates_passed)
            
            if self.episode_count % self.check_freq == 0:
                recent_rewards = self.episode_rewards[-self.check_freq:]
                recent_gates = self.episode_gates[-self.check_freq:]
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_gates = np.mean(recent_gates)
                max_gates = np.max(recent_gates)
                
                print(f"\n📊 Episode {self.episode_count}")
                print(f"   Avg Reward: {avg_reward:.2f}")
                print(f"   Avg Gates: {avg_gates:.2f}/7")
                print(f"   Best: {max_gates}/7")
        
        return True


def make_env(gate_positions):
    from gate_racing_position_control import PositionControlGateEnv
    
    def _init():
        env = PositionControlGateEnv(gate_positions=gate_positions)
        env = Monitor(env)
        return env
    
    return _init


def train_ppo(gate_positions, total_timesteps=150000):
    """
    Train PPO with position control.
    
    Faster than velocity control because:
    - AirSim handles low-level stabilization
    - 5Hz decision rate (not 10Hz)
    - Agent only learns WHERE to go, not HOW
    """
    
    log_dir = f"./ppo_position_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📁 Logs: {log_dir}")
    
    # Create environment
    env = DummyVecEnv([make_env(gate_positions)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # PPO model
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1
    )
    
    print("\n🤖 PPO Model (Position Control)")
    print(f"   Control type: Position setpoints")
    print(f"   Decision rate: 5Hz")
    print(f"   Training steps: {total_timesteps:,}")
    
    # Callbacks
    progress_callback = ProgressCallback(check_freq=10, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="ppo_position"
    )
    
    print("\n🚀 Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
    
    # Save
    final_path = os.path.join(log_dir, "ppo_position_final")
    model.save(final_path)
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete! Model: {final_path}")
    
    return model, env


def test_model(model_path, gate_positions, episodes=5):
    """Test trained model."""
    from gate_racing_position_control import PositionControlGateEnv
    
    model = PPO.load(model_path)
    env = PositionControlGateEnv(gate_positions=gate_positions)
    
    print(f"\n🧪 Testing for {episodes} episodes...\n")
    
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
            
            if steps % 10 == 0:
                print(f"  Step {steps}: Gates {info['gates_passed']}/7")
            
            if terminated or truncated:
                print(f"  ✅ Finished: {info['gates_passed']}/7 gates in {steps} steps")
                print(f"     Reward: {total_reward:.2f}\n")
                break
        
    env.close()


if __name__ == "__main__":
    # Your 7 gates - UPDATE THESE!
    gate_positions = [
        [5.8, -5.3, -0.7],      # Gate 0
        [17.3, -7.9, 1.0],      # Gate 1
        [28.9, -7.9, 1.1],      # Gate 2
        [39.3, -5.6, 1.3],      # Gate 3
        [46.3, 0.8, 1.1],       # Gate 4
        [46.3, 10.3, 0.7],      # Gate 5
        [39.5, 18.0, 0.8],      # Gate 6
    ]
    
    print("="*60)
    print("POSITION CONTROL PPO TRAINING")
    print("="*60)
    print("\nUsing standard AirSim API (moveToPositionAsync)")
    print("Agent learns: WHERE to go (position targets)")
    print("AirSim handles: HOW to get there (flight control)\n")
    
    choice = input("1. Train\n2. Test\nChoice: ").strip()
    
    if choice == "1":
        timesteps = input("Timesteps (default 150000): ").strip()
        timesteps = int(timesteps) if timesteps else 150000
        
        print(f"\nTraining with {timesteps:,} steps...")
        print("Expected results:")
        print("  - 30k steps: Learns to approach gates")
        print("  - 80k steps: Passes 2-3 gates")
        print("  - 150k steps: Completes most laps\n")
        
        input("Press Enter to start...")
        
        model, env = train_ppo(gate_positions, timesteps)
        env.close()
        
    elif choice == "2":
        model_path = input("Model path (no .zip): ").strip()
        episodes = input("Episodes (default 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        test_model(model_path, gate_positions, episodes)
    
    print("\n🎯 Done!")