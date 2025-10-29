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

OBSERVATION (21 values):
  - Drone position (3)
  - Drone velocity (3)
  - Vector to current gate (3)
  - Distance to current gate (1)
  - Vector to next gate (3)
  - Forward direction unit vector (3)
  - Current yaw angle (1)
  - Desired yaw angle (1)
  - Yaw alignment error (1)
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
        
        # NEW: Track when we just passed a gate to give extra "momentum" reward
        self.just_passed_gate = False
        self.steps_since_gate_pass = 0
        
        # Action space: [forward_offset, lateral_offset, vertical_offset, yaw_angle]
        # Agent learns to aim through gates AND align with gate orientation
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -2.0, -np.pi]),  # yaw: -180° to +180°
            high=np.array([15.0, 5.0, 2.0, np.pi]),
            dtype=np.float32
        )
        
        # Observation space (21 values now - added yaw info)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),  # Fixed: 3+3+3+1+3+3+1+1+1+1+1 = 21
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
        
        # Reset momentum tracking
        self.just_passed_gate = False
        self.steps_since_gate_pass = 0
        
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
        
        Returns array of 21 values describing current state.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        # Drone state
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        drone_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # Current yaw (heading)
        # Convert quaternion to yaw angle
        current_yaw = self._get_yaw_from_quaternion(orientation)
        
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
        
        # Desired yaw (angle to face toward next gate)
        desired_yaw = np.arctan2(forward_dir_norm[1], forward_dir_norm[0])
        
        # Yaw error (how much we need to turn)
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        
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
            [current_yaw],          # 1: current heading angle
            [desired_yaw],          # 1: desired heading angle
            [yaw_error],            # 1: yaw alignment error
            [progress],             # 1: completion progress
            [time_remaining]        # 1: time pressure
        ]).astype(np.float32)
        
        return obs
    
    def _get_yaw_from_quaternion(self, q):
        """
        Convert quaternion to yaw angle (heading).
        
        Args:
            q: AirSim quaternion (w, x, y, z)
        
        Returns:
            yaw: Heading angle in radians (-π to π)
        """
        # Extract quaternion components
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        
        # Convert to yaw (rotation around Z-axis)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        return yaw
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-π, π] range.
        
        Args:
            angle: Angle in radians
        
        Returns:
            Normalized angle in [-π, π]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
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
        
        # Compute target position and yaw from action
        target_pos, target_yaw = self._compute_target_from_action(action, current_pos)
        
        # Command drone with position AND yaw
        self.client.moveToPositionAsync(
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            velocity=6.0,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=float(np.degrees(target_yaw)))
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
        
        # Update momentum tracking
        if gate_passed:
            self.just_passed_gate = True
            self.steps_since_gate_pass = 0
        else:
            self.steps_since_gate_pass += 1
            # After 10 steps, we're no longer "just passed"
            if self.steps_since_gate_pass > 10:
                self.just_passed_gate = False
        
        # Compute reward (pass momentum info)
        current_gate = self.gate_positions[self.current_gate_idx]
        reward = self._compute_reward(
            old_pos=current_pos,
            new_pos=new_pos,
            velocity=velocity,
            target_gate=current_gate,
            gate_passed=gate_passed,
            speed=speed,
            just_passed=self.just_passed_gate,  # NEW: Pass momentum state
            steps_since_pass=self.steps_since_gate_pass
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
        Convert action to target position AND yaw.
        
        Action is offset relative to gate-to-gate direction:
        - action[0]: forward/backward offset
        - action[1]: lateral (left/right) offset  
        - action[2]: vertical (up/down) offset
        - action[3]: yaw angle (heading)
        
        Returns:
            target_pos: Target position (x, y, z)
            target_yaw: Target yaw angle in radians
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
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Target position = gate + action offsets
        target_pos = (
            current_gate +
            forward_norm * action[0] +  # Forward/back
            right_norm * action[1] +     # Left/right
            up_vec * action[2]           # Up/down
        )
        
        # Target yaw = base direction + action yaw adjustment
        base_yaw = np.arctan2(forward_norm[1], forward_norm[0])
        target_yaw = base_yaw + action[3]  # Add yaw adjustment
        target_yaw = self._normalize_angle(target_yaw)
        
        return target_pos, target_yaw
    
    def _compute_reward(self, old_pos, new_pos, velocity, target_gate, gate_passed, speed, just_passed, steps_since_pass):
        """
        IMPROVED reward function that encourages:
        1. Passing through gates (huge reward)
        2. Getting closer to gates
        3. CONTINUING to next gate after passing current one
        4. Moving forward (not backward)
        5. Speed (efficiency)
        6. Not wasting time
        """
        reward = 0.0
        
        # HUGE reward for passing gate + bonus for maintaining speed
        if gate_passed:
            base_gate_reward = 250.0
            
            # BONUS: If moving fast when passing gate, extra reward
            # This encourages "flying through" not "hovering through"
            speed_bonus = min(speed * 10.0, 50.0)  # Up to +50 for fast gate pass
            
            return base_gate_reward + speed_bonus
        
        # NEW: MOMENTUM BONUS - Extra rewards right after passing a gate
        # Encourages agent to "keep flying" to next gate
        if just_passed and steps_since_pass <= 10:
            # We just passed a gate! Give extra reward for continuing forward
            next_idx = (self.current_gate_idx + 1) % self.num_gates
            next_gate = self.gate_positions[next_idx]
            
            # Distance to next gate
            dist_to_next = np.linalg.norm(new_pos - next_gate)
            old_dist_to_next = np.linalg.norm(old_pos - next_gate)
            progress_to_next = old_dist_to_next - dist_to_next
            
            # HUGE reward for making progress toward next gate immediately after passing
            momentum_reward = progress_to_next * 100.0  # 2x normal reward!
            reward += momentum_reward
            
            # Extra bonus for high speed toward next gate
            direction_to_next = (next_gate - new_pos) / (dist_to_next + 1e-6)
            velocity_toward_next = np.dot(velocity, direction_to_next)
            
            if velocity_toward_next > 3.0:
                reward += 20.0  # Big bonus for fast movement to next gate
            elif velocity_toward_next > 1.0:
                reward += 10.0
            elif velocity_toward_next < 0:
                reward -= 30.0  # Big penalty for going wrong way after gate pass!
        
        # Distance-based reward TO CURRENT GATE
        old_dist = np.linalg.norm(old_pos - target_gate)
        new_dist = np.linalg.norm(new_pos - target_gate)
        progress = old_dist - new_dist
        
        # Strong reward for getting closer to current gate
        reward += progress * 50.0
        
        # Proximity bonus (getting close is good)
        if new_dist < 5.0:
            reward += (5.0 - new_dist) * 10.0
        
        # Extra bonus for being very close
        if new_dist < 3.0:
            reward += 50.0
        
        # NEW: Yaw alignment reward
        # Get current yaw
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        current_yaw = self._get_yaw_from_quaternion(orientation)
        
        # Get desired yaw (toward next gate)
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        forward_dir = next_gate - target_gate
        forward_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
        desired_yaw = np.arctan2(forward_norm[1], forward_norm[0])
        
        # Yaw error (how misaligned we are)
        yaw_error = abs(self._normalize_angle(desired_yaw - current_yaw))
        
        # Reward for good alignment (error close to 0)
        # When well-aligned (< 15°), give bonus
        if yaw_error < np.pi / 12:  # < 15 degrees
            alignment_bonus = (1.0 - yaw_error / (np.pi / 12)) * 15.0
            reward += alignment_bonus
        elif yaw_error < np.pi / 6:  # < 30 degrees
            alignment_bonus = (1.0 - yaw_error / (np.pi / 6)) * 8.0
            reward += alignment_bonus
        
        # Penalty for being very misaligned when close to gate
        if new_dist < 5.0 and yaw_error > np.pi / 3:  # > 60 degrees
            reward -= 15.0
        
        # NEW: Reward for being positioned BETWEEN current and next gate
        # This encourages "gate-to-gate" thinking, not just "get to gate and stop"
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_idx]
        
        # Calculate if we're "on the path" from current to next gate
        gate_to_gate_vec = next_gate - target_gate
        gate_to_gate_dist = np.linalg.norm(gate_to_gate_vec)
        gate_to_gate_norm = gate_to_gate_vec / (gate_to_gate_dist + 1e-6)
        
        # NEW: Reward for moving in the "gate-to-gate" direction
        velocity_along_path = np.dot(velocity, gate_to_gate_norm)
        if velocity_along_path > 0:
            # Moving toward next gate = good!
            reward += velocity_along_path * 8.0  # Increased from 5.0
        else:
            # Moving backward = very bad!
            reward += velocity_along_path * 15.0  # Increased penalty
        
        # NEW: If close to current gate, start rewarding progress toward NEXT gate
        if new_dist < 4.0:
            # We're close to current gate, so also look ahead to next gate
            dist_to_next = np.linalg.norm(new_pos - next_gate)
            old_dist_to_next = np.linalg.norm(old_pos - next_gate)
            progress_to_next = old_dist_to_next - dist_to_next
            
            # Reward for also getting closer to next gate
            reward += progress_to_next * 20.0
            
            # Encourage "lining up" for next gate while passing current
            if velocity_along_path > 3.0:  # Moving fast toward next gate
                reward += 10.0
        
        # Speed bonus (encourage continuous action)
        if speed > 3.0:
            reward += 5.0  # Increased bonus for higher speed
        elif speed > 2.0:
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
        [5.8, -5.3, -0.7],
        [17.3, -7.9, 1.0],
        [28.9, -7.9, 1.1],
        [39.3, -5.6, 1.3],
        [46.3, 0.8, 1.1],
        [46.3, 10.3, 0.7],
        [39.5, 18.0, 0.8],
    ]
    
    env = GateRacingEnv(gate_positions=gate_positions)
    
    print("\n" + "="*70)
    print("ENVIRONMENT TEST")
    print("="*70)
    
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps with forward actions
    print("\nTesting forward flight with yaw control...")
    for i in range(10):
        action = np.array([10.0, 0.0, 0.0, 0.0])  # Fly forward, no yaw change
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Reward={reward:.1f}, Gates={info['gates_passed']}/{len(gate_positions)}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Environment test complete!")