"""
Fixed Circular Gate Racing Environment for PPO Training

Key fixes:
1. Reward function that FORCES movement (no hover farming)
2. Proper gate detection with initialization handling
3. Full observation space (all gates + drone state)
4. Progressive difficulty (start easy, get harder)
5. Curriculum learning support
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import math


class CircularGateEnv(gym.Env):
    """
    Circular gate racing environment designed for PPO training.
    
    ANALOGY: Think of this like a video game racing level.
    - Easy mode: Just get through any gate (early training)
    - Normal mode: Go through gates in sequence
    - Hard mode: Go through gates AND follow circular path
    
    The environment progressively teaches these skills.
    """
    
    def __init__(self, gate_positions, gate_radius=2.5, circle_radius=15.0, 
                 curriculum_stage=0, render_mode=None):
        """
        Args:
            gate_positions: List of 7 (x, y, z) coordinates for gates
            gate_radius: Radius of gate opening (meters)
            circle_radius: Radius of desired circular path
            curriculum_stage: 0=easy, 1=medium, 2=hard (progressive training)
            render_mode: gymnasium standard (unused, AirSim handles rendering)
        """
        super(CircularGateEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Gate setup
        self.gate_positions = np.array(gate_positions, dtype=np.float32)
        self.num_gates = len(gate_positions)
        if self.num_gates != 7:
            raise ValueError(f"Expected 7 gates, got {self.num_gates}")
        
        self.gate_radius = gate_radius
        self.circle_radius = circle_radius
        self.curriculum_stage = curriculum_stage
        
        # Tracking variables
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.gates_passed_set = set()  # Track which gates we've passed
        self.last_gate_passed_time = 0
        self.steps = 0
        self.max_steps = 1000  # Shorter - end hovering episodes fast!
        
        # Circle center
        self.circle_center = np.mean(self.gate_positions[:, :2], axis=0)
        
        # State tracking
        self.prev_position = None
        self.prev_distance_to_gate = None
        self.consecutive_hovering_steps = 0  # Count how long drone hasn't moved
        
        # Define action space: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -2.0, -1.0]),
            high=np.array([5.0, 5.0, 2.0, 1.0]),
            dtype=np.float32
        )
        
        # IMPROVED observation space
        # We need: drone state (10) + current gate info (4) + next gate info (4) = 18
        # [drone_x, drone_y, drone_z,  # position (3)
        #  drone_vx, drone_vy, drone_vz,  # velocity (3)
        #  drone_roll, drone_pitch, drone_yaw,  # orientation (3)
        #  drone_yaw_rate,  # angular velocity (1)
        #  gate_rel_x, gate_rel_y, gate_rel_z, gate_distance,  # current gate (4)
        #  next_gate_rel_x, next_gate_rel_y, next_gate_rel_z, next_gate_distance,  # next gate (4)
        #  gates_passed_normalized,  # progress (1)
        #  steps_since_last_gate_normalized]  # urgency signal (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to starting state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset AirSim
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Start position: in front of gate 0, facing it
        start_pos = self.gate_positions[0].copy()
        start_pos[0] -= 8.0  # 8 meters before first gate
        start_pos[2] += 1.0  # Slightly above gate center
        
        self.client.moveToPositionAsync(
            float(start_pos[0]), 
            float(start_pos[1]), 
            float(start_pos[2]), 
            velocity=2.0
        ).join()
        time.sleep(0.5)
        
        # Orient toward first gate
        gate_dir = self.gate_positions[0] - start_pos
        target_yaw = np.arctan2(gate_dir[1], gate_dir[0])
        self.client.rotateToYawAsync(float(np.degrees(target_yaw))).join()
        time.sleep(0.3)
        
        # Reset tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.gates_passed_set = set()
        self.last_gate_passed_time = 0
        self.steps = 0
        self.consecutive_hovering_steps = 0
        
        # Initialize previous state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.prev_position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        target_gate = self.gate_positions[self.current_gate_idx]
        self.prev_distance_to_gate = np.linalg.norm(self.prev_position - target_gate)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Get current observation.
        
        ANALOGY: This is like a race car driver's dashboard.
        - Speedometer (velocity)
        - GPS position
        - Compass (orientation)
        - Map showing next 2 checkpoints
        """
        # Get drone state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        drone_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self._quaternion_to_euler(orientation)
        
        # Angular velocity (yaw rate)
        ang_vel = state.kinematics_estimated.angular_velocity
        yaw_rate = ang_vel.z_val
        
        # Current target gate
        target_gate = self.gate_positions[self.current_gate_idx]
        gate_rel = target_gate - drone_pos
        gate_distance = np.linalg.norm(gate_rel)
        
        # Next gate (for lookahead)
        next_gate_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_gate_idx]
        next_gate_rel = next_gate - drone_pos
        next_gate_distance = np.linalg.norm(next_gate_rel)
        
        # Progress indicators
        gates_progress = self.gates_passed / self.num_gates
        steps_since_gate = (self.steps - self.last_gate_passed_time) / 500.0
        
        # Construct observation
        obs = np.concatenate([
            drone_pos,  # 3
            drone_vel,  # 3
            [roll, pitch, yaw],  # 3
            [yaw_rate],  # 1
            gate_rel,  # 3
            [gate_distance],  # 1
            next_gate_rel,  # 3
            [next_gate_distance],  # 1
            [gates_progress],  # 1
            [steps_since_gate]  # 1
        ]).astype(np.float32)
        
        return obs
    
    def _quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
        cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _check_gate_pass(self, position):
        """
        Robust gate passing detection.
        
        ANALOGY: Like a finish line detector at a race track.
        - Must cross the line (not just get close)
        - Must be within the lane boundaries
        - Must cross in the right direction (no going backwards)
        """
        if self.prev_position is None:
            return False
        
        target_gate = self.gate_positions[self.current_gate_idx]
        
        # Check proximity (must be close to gate)
        distance_xy = np.linalg.norm(position[:2] - target_gate[:2])
        distance_z = abs(position[2] - target_gate[2])
        
        if distance_xy > self.gate_radius or distance_z > 2.0:
            return False
        
        # Check plane crossing
        # ANALOGY: Imagine the gate as a vertical plane.
        # We need to check if we crossed from the "front" side to the "back" side.
        
        # Get the direction from gate to next gate (this is the "forward" direction)
        next_gate_idx = (self.current_gate_idx + 1) % self.num_gates
        next_gate = self.gate_positions[next_gate_idx]
        forward_dir = next_gate[:2] - target_gate[:2]
        forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
        
        # Project positions onto this direction
        prev_proj = np.dot(self.prev_position[:2] - target_gate[:2], forward_dir)
        curr_proj = np.dot(position[:2] - target_gate[:2], forward_dir)
        
        # Did we cross from negative to positive? (front to back)
        if prev_proj < 0 and curr_proj >= 0:
            return True
        
        return False
    
    def _compute_reward(self, position, velocity, terminated, truncated):
        """
        FIXED reward function that prevents hover farming.
        
        KEY PRINCIPLE: Make standing still WORSE than trying and failing.
        
        ANALOGY: You're coaching a kid learning to ride a bike.
        - Sitting still with training wheels: 0 points (boring!)
        - Pedaling, even if wobbly: +5 points (effort!)
        - Crashing while trying: -2 points (small penalty, we learn from this)
        - Reaching the checkpoint: +50 points (success!)
        
        This way, the kid learns that moving > standing still.
        """
        reward = 0.0
        
        # === COMPONENT 1: ANTI-HOVER MECHANISM ===
        # This is CRITICAL. We MUST punish hovering more than we punish trying.
        speed = np.linalg.norm(velocity)
        
        # MUCH MORE AGGRESSIVE: Require actual forward movement
        if speed < 1.0:  # Changed from 0.3 to 1.0 - must actually fly!
            self.consecutive_hovering_steps += 1
            # MUCH stronger penalty
            hover_penalty = -2.0 * (1 + self.consecutive_hovering_steps / 50.0)
            reward += hover_penalty
            
            # Extra punishment if basically stationary
            if speed < 0.2:
                reward -= 5.0  # Severe penalty for not moving at all
        else:
            self.consecutive_hovering_steps = 0
            # Strong reward for actual movement!
            movement_reward = speed * 2.0  # Doubled
            reward += movement_reward
        
        # === COMPONENT 2: PROGRESS TOWARD GATE ===
        target_gate = self.gate_positions[self.current_gate_idx]
        distance_to_gate = np.linalg.norm(position - target_gate)
        
        if self.prev_distance_to_gate is not None:
            progress = self.prev_distance_to_gate - distance_to_gate
            # Strong reward for getting closer
            reward += progress * 15.0
        
        self.prev_distance_to_gate = distance_to_gate
        
        # === COMPONENT 3: HEADING ALIGNMENT ===
        # Reward for facing the gate AND moving toward it
        gate_direction = target_gate - position
        gate_dir_normalized = gate_direction / (np.linalg.norm(gate_direction) + 1e-6)
        
        if speed > 0.5:  # Only check alignment when actually moving
            velocity_normalized = velocity / (speed + 1e-6)
            alignment = np.dot(gate_dir_normalized, velocity_normalized)
            # Alignment ranges from -1 (wrong way) to +1 (perfect)
            
            # CRITICAL: Big penalty for moving AWAY from gate
            if alignment < 0:
                reward += alignment * 10.0  # Punish wrong direction strongly
            else:
                reward += alignment * 5.0  # Reward right direction
        else:
            # Not moving fast enough - penalty!
            reward -= 1.0
        
        # === COMPONENT 4: GATE PASSING (BIG REWARD) ===
        if self._check_gate_pass(position):
            gate_reward = 200.0  # HUGE reward!
            
            # Bonus for passing gates in sequence (harder)
            if self.current_gate_idx not in self.gates_passed_set:
                gate_reward += 50.0
                self.gates_passed_set.add(self.current_gate_idx)
            
            reward += gate_reward
            self.gates_passed += 1
            self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates
            self.last_gate_passed_time = self.steps
            
            print(f"🎯 Gate {self.gates_passed}/7 passed! Total reward this step: +{gate_reward:.1f}")
        
        # === COMPONENT 5: TIME PRESSURE ===
        # Penalty for taking too long between gates
        steps_since_gate = self.steps - self.last_gate_passed_time
        if steps_since_gate > 300:  # 30 seconds at 10Hz
            time_penalty = -(steps_since_gate - 300) * 0.02
            reward += time_penalty
        
        # === COMPONENT 6: COMPLETION BONUS ===
        if self.gates_passed >= self.num_gates:
            completion_bonus = 1000.0
            reward += completion_bonus
            print(f"🏆 FULL CIRCLE COMPLETED! Bonus: +{completion_bonus}")
        
        # === COMPONENT 7: CRASH PENALTY ===
        if terminated and self.gates_passed < self.num_gates:
            # Make crash penalty LESS than hover farming penalty
            # If hovering farms +0.1/step, and an episode is 2000 steps,
            # hovering gives +200 total. So crash penalty must be less.
            crash_penalty = -100.0
            reward += crash_penalty
            print(f"💥 Crashed! Penalty: {crash_penalty}")
        
        # === COMPONENT 8: CURRICULUM-BASED REWARDS ===
        # Gradually introduce circle-following constraint
        if self.curriculum_stage >= 1:
            # Only penalize circle deviation in later stages
            position_2d = position[:2]
            distance_from_center = np.linalg.norm(position_2d - self.circle_center)
            deviation = abs(distance_from_center - self.circle_radius)
            
            if self.curriculum_stage == 1:
                circle_penalty = -deviation * 0.1  # Weak penalty
            else:  # curriculum_stage == 2
                circle_penalty = -deviation * 0.3  # Stronger penalty
            
            reward += circle_penalty
        
        # === COMPONENT 9: DISTANCE-BASED SHAPING ===
        # Potential-based reward shaping (helps with sparse rewards)
        distance_shaping = -distance_to_gate * 0.02
        reward += distance_shaping
        
        return reward
    
    def step(self, action):
        """Execute one timestep."""
        self.steps += 1
        
        # Convert to native Python floats (AirSim msgpack requirement)
        action = np.array(action, dtype=np.float64)
        vx, vy, vz, yaw_rate = [float(x) for x in action]
        
        # Send velocity command
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.1,  # 10 Hz
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()
        
        # Get new state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        # Check collision
        collision_info = self.client.simGetCollisionInfo()
        crashed = collision_info.has_collided
        
        # Episode termination logic
        terminated = False
        truncated = False
        
        if crashed:
            terminated = True
        elif self.gates_passed >= self.num_gates:
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Compute reward
        reward = self._compute_reward(position, velocity, terminated, truncated)
        
        # Get observation
        obs = self._get_observation()
        
        # Update previous position
        self.prev_position = position
        
        # Info dict
        info = {
            'gates_passed': self.gates_passed,
            'crashed': crashed,
            'steps': self.steps,
            'current_gate': self.current_gate_idx,
            'consecutive_hovering': self.consecutive_hovering_steps
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """AirSim handles rendering."""
        pass
    
    def set_curriculum_stage(self, stage):
        """
        Update curriculum stage during training.
        
        Called by the curriculum callback when the agent is ready to advance.
        """
        self.curriculum_stage = stage
        print(f"Environment curriculum stage updated to: {stage}")
    
    def close(self):
        """Clean up."""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


if __name__ == "__main__":
    # Test with your 7 gates
    # These should match your AirSim environment
    gate_positions = [
        [15, 0, -5],     # Gate 0 (start)
        [10.6, 10.6, -5],  # Gate 1
        [0, 15, -5],     # Gate 2
        [-10.6, 10.6, -5],  # Gate 3
        [-15, 0, -5],    # Gate 4
        [-10.6, -10.6, -5],  # Gate 5
        [0, -15, -5],    # Gate 6
    ]
    
    env = CircularGateEnv(gate_positions=gate_positions, curriculum_stage=0)
    
    print("Testing environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial obs: {obs}")
    
    # Test a few random actions
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Gates={info['gates_passed']}/7, " 
              f"Hovering={info['consecutive_hovering']}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("Test complete!")