"""
AirSim Gate Racing Environment for PPO Training

This environment teaches a drone to fly through gates in sequence.
Uses position control (moveToPositionAsync) for stability.

REWARD STRUCTURE:
  +250  = Pass through a gate
  +50   = Get very close to gate (< 3m)
  +20   = Progress toward gate
  -50   = Going backwards / wrong direction
  -100  = Crash
  +1000 = Complete all gates in order
  -1    = Time penalty (efficiency)

OBSERVATION (18 values):
  - Drone position (3)
  - Drone velocity (3)
  - Vector to current gate (3)
  - Distance to current gate (1)
  - Vector to next gate (3)
  - Forward direction unit vector (3)
  - Progress (gates completed / total) (1)
  - Time remaining ratio (1)
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class GateRacingEnv(gym.Env):
    """
    Gate racing environment with sequential gate passing.
    Agent must pass gates IN ORDER to get rewards.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, gate_positions, gate_radius=3.0, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Gate configuration
        self.gate_positions = np.array(gate_positions, dtype=np.float32)
        self.num_gates = len(gate_positions)
        self.gate_radius = gate_radius
        
        # Episode tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.steps = 0
        self.max_steps = 500
        self.last_position = None
        self.last_distance_to_gate = None
        self.passed_gates = set()
        
        # Action space: [forward_offset, lateral_offset, vertical_offset]
        # Agent learns to aim through gates
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -2.0]),
            high=np.array([15.0, 5.0, 2.0]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )
        
        print(f"🏁 Gate Racing Environment initialized")
        print(f"   Gates: {self.num_gates}")
        print(f"   Gate radius: {self.gate_radius}m")
        print(f"   Max steps: {self.max_steps}")
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset AirSim
        self.client.reset()
        time.sleep(0.1)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        time.sleep(0.3)
        
        # Position drone BEFORE first gate
        start_pos = self._compute_start_position()
        self.client.moveToPositionAsync(
            float(start_pos[0]),
            float(start_pos[1]),
            float(start_pos[2]),
            velocity=3.0
        ).join()
        time.sleep(0.3)
        
        # Reset episode state
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.steps = 0
        self.passed_gates.clear()
        
        # Get initial position
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.last_position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        current_gate = self.gate_positions[self.current_gate_idx]
        self.last_distance_to_gate = np.linalg.norm(self.last_position - current_gate)
        
        return self._get_observation(), {}
    
    def _compute_start_position(self):
        """
        Compute starting position BEFORE first gate.
        Places drone 20m before gate 0, facing toward it.
        """
        gate_0 = self.gate_positions[0]
        gate_1 = self.gate_positions[1] if self.num_gates > 1 else gate_0 + np.array([10, 0, 0])
        
        # Direction from gate 1 to gate 0 (backwards)
        backward_dir = gate_0 - gate_1
        backward_dir_norm = backward_dir / (np.linalg.norm(backward_dir) + 1e-6)
        
        # Start 20m before gate 0
        start_pos = gate_0 + backward_dir_norm * 20.0
        
        return start_pos
    
    def _get_observation(self):
        """
        Build observation vector for agent.
        
        Returns array of 18 values describing current state.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        # Drone state
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        drone_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # Current gate info
        current_gate = self.gate_positions[self.current_gate_idx]
        vec_to_gate = current_gate - drone_pos
        dist_to_gate = np.linalg.norm(vec_to_gate)
        
        # Next gate info
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        vec_to_next = next_gate - current_gate
        
        # Forward direction (gate-to-gate)
        forward_dir = next_gate - current_gate
        forward_dir_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
        
        # Progress metrics
        progress = self.gates_passed / self.num_gates
        time_remaining = 1.0 - (self.steps / self.max_steps)
        
        obs = np.concatenate([
            drone_pos,              # 3: current position
            drone_vel,              # 3: current velocity
            vec_to_gate,            # 3: vector to current gate
            [dist_to_gate],         # 1: distance to gate
            vec_to_next,            # 3: vector from current to next gate
            forward_dir_norm,       # 3: direction to fly through gate
            [progress],             # 1: completion progress
            [time_remaining]        # 1: time pressure
        ]).astype(np.float32)
        
        return obs
    
    def _check_gate_pass(self, position):
        """
        Check if drone passed through current gate.
        
        Returns True if:
        1. Drone is within gate radius (XY plane)
        2. Drone is at correct altitude (Z within tolerance)
        3. Drone hasn't counted this gate before
        """
        target_gate = self.gate_positions[self.current_gate_idx]
        
        # Check if already passed this gate
        if self.current_gate_idx in self.passed_gates:
            return False
        
        # Distance checks
        distance_xy = np.linalg.norm(position[:2] - target_gate[:2])
        distance_z = abs(position[2] - target_gate[2])
        
        # Must be inside gate cylinder
        inside_gate = (distance_xy < self.gate_radius and distance_z < 2.5)
        
        if inside_gate:
            # Mark as passed
            self.passed_gates.add(self.current_gate_idx)
            return True
        
        return False
    
    def step(self, action):
        """
        Execute one step in environment.
        
        Agent provides action (target offset), environment:
        1. Computes target position
        2. Commands drone to fly there
        3. Waits for movement
        4. Checks for gate pass
        5. Computes reward
        6. Returns observation
        """
        self.steps += 1
        
        # Get current state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Compute target position from action
        target_pos = self._compute_target_from_action(action, current_pos)
        
        # Command drone
        self.client.moveToPositionAsync(
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            velocity=6.0
        )
        
        # Wait for movement (2Hz control rate)
        time.sleep(0.5)
        
        # Get new state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        new_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        
        # Check gate pass
        gate_passed = self._check_gate_pass(new_pos)
        
        # Compute reward
        current_gate = self.gate_positions[self.current_gate_idx]
        reward = self._compute_reward(
            old_pos=current_pos,
            new_pos=new_pos,
            velocity=velocity,
            target_gate=current_gate,
            gate_passed=gate_passed,
            speed=speed
        )
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Collision check
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            terminated = True
            reward -= 100.0
        
        # Success check
        elif self.gates_passed >= self.num_gates:
            terminated = True
            reward += 1000.0
        
        # Timeout check
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Update tracking
        if gate_passed:
            self.gates_passed += 1
            old_gate = self.current_gate_idx
            self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates
            
            # Update distance to new gate
            new_gate = self.gate_positions[self.current_gate_idx]
            self.last_distance_to_gate = np.linalg.norm(new_pos - new_gate)
        
        self.last_position = new_pos
        
        # Build info dict
        info = {
            'gates_passed': self.gates_passed,
            'current_gate': self.current_gate_idx,
            'steps': self.steps,
            'speed': speed
        }
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_target_from_action(self, action, current_pos):
        """
        Convert action to target position.
        
        Action is offset relative to gate-to-gate direction:
        - action[0]: forward/backward offset
        - action[1]: lateral (left/right) offset  
        - action[2]: vertical (up/down) offset
        """
        current_gate = self.gate_positions[self.current_gate_idx]
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        
        # Forward direction
        forward_vec = next_gate - current_gate
        forward_norm = forward_vec / (np.linalg.norm(forward_vec) + 1e-6)
        
        # Right direction (perpendicular)
        right_norm = np.array([-forward_norm[1], forward_norm[0], 0])
        
        # Up direction
        up_vec = np.array([0, 0, 1])
        
        # Target = gate + action offsets
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        target = (
            current_gate +
            forward_norm * action[0] +  # Forward/back
            right_norm * action[1] +     # Left/right
            up_vec * action[2]           # Up/down
        )
        
        return target
    
    def _compute_reward(self, old_pos, new_pos, velocity, target_gate, gate_passed, speed):
        """
        Reward function that encourages:
        1. Passing through gates (huge reward)
        2. Getting closer to gates
        3. Moving forward (not backward)
        4. Speed (efficiency)
        5. Not wasting time
        """
        reward = 0.0
        
        # HUGE reward for passing gate
        if gate_passed:
            return 250.0
        
        # Distance-based reward
        old_dist = np.linalg.norm(old_pos - target_gate)
        new_dist = np.linalg.norm(new_pos - target_gate)
        progress = old_dist - new_dist
        
        # Strong reward for getting closer
        reward += progress * 50.0
        
        # Proximity bonus (getting close is good)
        if new_dist < 5.0:
            reward += (5.0 - new_dist) * 10.0
        
        # Extra bonus for being very close
        if new_dist < 3.0:
            reward += 50.0
        
        # Forward motion check
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        forward_dir = next_gate - target_gate
        forward_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
        
        # Reward for moving in correct direction
        velocity_forward = np.dot(velocity, forward_norm)
        if velocity_forward > 0:
            reward += velocity_forward * 5.0
        else:
            # Penalty for going backward
            reward += velocity_forward * 10.0
        
        # Speed bonus (encourage action)
        if speed > 2.0:
            reward += 3.0
        elif speed < 0.5:
            reward -= 5.0
        
        # Small time penalty (encourage efficiency)
        reward -= 1.0
        
        return reward
    
    def close(self):
        """Clean up environment."""
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
    
    def render(self):
        """Render environment (optional)."""
        pass


# Quick test
if __name__ == "__main__":
    gate_positions = [
        # [5.8, -5.3, -0.5],
        # [17.3, -7.9, 1.0],
        # [28.9, -7.9, 1.1],
        # [39.3, -5.6, 1.3],
        # [46.3, 0.8, 1.1],
        # [46.3, 10.3, 0.7],
        # [39.5, 18.0, 0.8],
        [5.8, -5.3,  1.0],
        [17.3, -7.9, 1.0],
        [28.9, -7.9, 1.0],
        [39.3, -5.6, 1.0],
        [46.3, 0.8,  1.0],
        [46.3, 10.3, 1.0],
        [39.5, 18.0, 1.0],
    ]
    
    env = GateRacingEnv(gate_positions=gate_positions)
    
    print("\n" + "="*70)
    print("ENVIRONMENT TEST")
    print("="*70)
    
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps with forward actions
    print("\nTesting forward flight...")
    for i in range(10):
        action = np.array([10.0, 0.0, 0.0])  # Fly forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Reward={reward:.1f}, Gates={info['gates_passed']}/{len(gate_positions)}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Environment test complete!")