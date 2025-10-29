"""
Gate Racing with Position Control (Standard AirSim)

KEY INSIGHT: Instead of low-level velocity control, use position setpoints!
- Agent outputs: target position to fly toward
- AirSim's built-in position controller handles the details
- Much more stable than velocity control
- Works with standard AirSim (no special packages needed)

This is the middle ground between velocity control (too hard) and 
moveOnSpline (not available in your AirSim version).
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class PositionControlGateEnv(gym.Env):
    """
    Gate racing using position control.
    
    Agent decides WHERE to go (position target)
    AirSim's PID controller decides HOW to get there
    
    This is MUCH easier to learn than velocity control!
    """
    
    def __init__(self, gate_positions, gate_radius=2.5, render_mode=None):
        super(PositionControlGateEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Gate setup
        self.gate_positions = np.array(gate_positions, dtype=np.float32)
        self.num_gates = len(gate_positions)
        self.gate_radius = gate_radius
        
        # Tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.steps = 0
        self.max_steps = 300  # Shorter - position control is efficient
        self.last_position = None
        
        # Action space: [dx, dy, dz] - offset from current gate
        # Agent learns to aim NEAR the gate (not exactly at it)
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -2.0]),
            high=np.array([5.0, 5.0, 2.0]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # More info than before
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset AirSim
        self.client.reset()
        time.sleep(0.1)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Start position
        start_pos = self.gate_positions[0].copy()
        start_pos[0] -= 10.0  # 10m before first gate
        
        self.client.moveToPositionAsync(
            float(start_pos[0]),
            float(start_pos[1]),
            float(start_pos[2]),
            velocity=3.0
        ).join()
        
        time.sleep(0.3)
        
        # Reset tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.steps = 0
        
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.last_position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        drone_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # Current gate
        current_gate = self.gate_positions[self.current_gate_idx]
        gate_rel = current_gate - drone_pos
        gate_dist = np.linalg.norm(gate_rel)
        
        # Next gate
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        next_gate_rel = next_gate - drone_pos
        
        # Progress
        progress = self.gates_passed / self.num_gates
        
        obs = np.concatenate([
            drone_pos,          # 3
            drone_vel,          # 3
            gate_rel,           # 3
            [gate_dist],        # 1
            next_gate_rel,      # 3
            [progress],         # 1
            [self.steps / self.max_steps]  # 1
        ]).astype(np.float32)
        
        return obs
    
    def _check_gate_pass(self, position):
        """
        Check if passed gate - STRICT VERSION.
        
        Must satisfy ALL conditions:
        1. Be within gate radius (XY plane)
        2. Be at correct height (Z)
        3. Actually CROSS the gate plane from front to back
        """
        target_gate = self.gate_positions[self.current_gate_idx]
        
        # Check proximity first (cheap check)
        distance_xy = np.linalg.norm(position[:2] - target_gate[:2])
        distance_z = abs(position[2] - target_gate[2])
        
        # DEBUG: Print distances
        if hasattr(self, 'debug') and self.debug:
            print(f"  Distance XY: {distance_xy:.2f}m (limit: {self.gate_radius * 0.8:.2f}m)")
            print(f"  Distance Z: {distance_z:.2f}m (limit: 1.5m)")
        
        # STRICTER: Must be very close
        if distance_xy > self.gate_radius * 0.8:  # 80% of gate radius
            return False
        
        if distance_z > 1.5:  # Must be at correct height
            return False
        
        # Now check if we actually CROSSED the gate plane
        if self.last_position is None:
            return False
        
        # Get direction to next gate (defines "forward")
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        
        # Forward direction is toward next gate
        forward_vec = next_gate[:2] - target_gate[:2]
        forward_vec_norm = forward_vec / (np.linalg.norm(forward_vec) + 1e-6)
        
        # Project last and current position onto forward direction
        last_proj = np.dot(self.last_position[:2] - target_gate[:2], forward_vec_norm)
        curr_proj = np.dot(position[:2] - target_gate[:2], forward_vec_norm)
        
        # DEBUG: Print projections
        if hasattr(self, 'debug') and self.debug:
            print(f"  Last projection: {last_proj:.2f}m")
            print(f"  Current projection: {curr_proj:.2f}m")
        
        # Must have crossed from negative (before gate) to positive (after gate)
        # AND the crossing distance should be reasonable (not teleporting)
        crossing_distance = abs(curr_proj - last_proj)
        
        if last_proj < -0.5 and curr_proj > 0.5 and crossing_distance < 5.0:
            # Actually crossed through the gate!
            if hasattr(self, 'debug') and self.debug:
                print(f"  ✅ GATE PASSED! Crossed {crossing_distance:.2f}m")
            return True
        
        if hasattr(self, 'debug') and self.debug and distance_xy < self.gate_radius:
            print(f"  ❌ Close but didn't cross gate plane properly")
        
        return False
    
    def step(self, action):
        """
        Execute step with position control.
        
        Agent outputs: offset from gate position
        AirSim: flies to that position using built-in controller
        """
        self.steps += 1
        
        # Get current state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Target position = gate + action offset
        target_gate = self.gate_positions[self.current_gate_idx]
        action = np.array(action, dtype=np.float32)
        target_pos = target_gate + action
        
        # Fly to target position (non-blocking)
        self.client.moveToPositionAsync(
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            velocity=6.0  # Reasonable speed
        )
        
        # Wait a bit (5Hz control rate)
        time.sleep(0.2)
        
        # Get new position
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        new_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        
        # Check gate pass
        gate_passed = self._check_gate_pass(new_pos)
        
        # Compute reward
        reward = self._compute_reward(current_pos, new_pos, target_gate, gate_passed, speed)
        
        # Check termination
        terminated = False
        truncated = False
        
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            terminated = True
            reward -= 100.0
            print("💥 Crashed!")
        elif self.gates_passed >= self.num_gates:
            terminated = True
            reward += 500.0
            print("🏆 Circle complete!")
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Update tracking
        if gate_passed:
            self.gates_passed += 1
            self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates
            print(f"🎯 Gate {self.gates_passed}/7 passed!")
        
        self.last_position = new_pos
        
        obs = self._get_observation()
        info = {
            'gates_passed': self.gates_passed,
            'steps': self.steps
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, old_pos, new_pos, target_gate, gate_passed, speed):
        """
        Reward function for position control.
        
        Since AirSim handles flight, we focus on:
        1. Getting closer to gate
        2. Passing gates
        3. Moving (not stuck)
        """
        reward = 0.0
        
        # Gate passing - HUGE reward!
        if gate_passed:
            return 200.0
        
        # Progress toward gate
        old_dist = np.linalg.norm(old_pos - target_gate)
        new_dist = np.linalg.norm(new_pos - target_gate)
        progress = old_dist - new_dist
        reward += progress * 20.0  # Strong signal
        
        # Proximity bonus (getting close is good)
        if new_dist < 5.0:
            reward += (5.0 - new_dist) * 5.0
        
        # Movement bonus (encourage action)
        if speed > 1.0:
            reward += speed * 0.5
        else:
            reward -= 2.0  # Penalty for being slow/stuck
        
        # Small time penalty
        reward -= 0.5
        
        return reward
    
    def close(self):
        """Clean up."""
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
    
    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    # Test environment
    gate_positions = [
   [5.8, -5.3, -0.7],      # Gate 0
        [17.3, -7.9, 1.0],      # Gate 1
        [28.9, -7.9, 1.1],      # Gate 2
        [39.3, -5.6, 1.3],      # Gate 3
        [46.3, 0.8, 1.1],       # Gate 4
        [46.3, 10.3, 0.7],      # Gate 5
        [39.5, 18.0, 0.8],      # Gate 6
    ]
    
    env = PositionControlGateEnv(gate_positions=gate_positions)
    env.debug = True  # Enable debug output!
    
    print("Testing position control environment (DEBUG MODE)...")
    print("=" * 60)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Starting position near gate 0: {gate_positions[0]}")
    print("=" * 60)
    
    # Test with small random actions
    for i in range(20):
        action = env.action_space.sample() * 0.3  # Small adjustments
        print(f"\nStep {i}: Taking action {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  → Reward={reward:.2f}, Gates={info['gates_passed']}/7")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("If you never see '✅ GATE PASSED', the drone didn't actually")
    print("fly through any gate - just got close to one.")