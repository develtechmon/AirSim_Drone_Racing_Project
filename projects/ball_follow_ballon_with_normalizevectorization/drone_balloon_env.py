import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class DroneBalloonEnv(gym.Env):
    """
    Custom Gymnasium environment where a drone chases and pops balloons.
    
    Observation Space: [drone_x, drone_y, balloon_x, balloon_y]
    Action Space: Discrete(5) - [stay, up, down, left, right]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, canvas_size=400):
        super().__init__()
        
        # Environment parameters
        self.canvas_size = canvas_size
        self.drone_size = 20
        self.balloon_size = 15
        self.drone_speed = 10  # pixels per step
        self.max_score = 6
        self.hit_distance = 25  # Distance threshold for hitting balloon
        
        # Observation space: [drone_x, drone_y, balloon_x, balloon_y]
        # Normalized to [0, 1] range
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = spaces.Discrete(5)
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # State variables
        self.drone_pos = None
        self.balloon_pos = None
        self.score = 0
        self.steps = 0
        self.max_steps_per_episode = 1000
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset score and steps
        self.score = 0
        self.steps = 0
        
        # Spawn drone at random position
        self.drone_pos = np.array([
            self.np_random.integers(50, self.canvas_size - 50),
            self.np_random.integers(50, self.canvas_size - 50)
        ], dtype=np.float32)
        
        # Spawn balloon at random position (away from drone)
        self.balloon_pos = self._spawn_balloon()
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # Move drone based on action
        previous_pos = self.drone_pos.copy()
        
        if action == 1:  # up
            self.drone_pos[1] -= self.drone_speed
        elif action == 2:  # down
            self.drone_pos[1] += self.drone_speed
        elif action == 3:  # left
            self.drone_pos[0] -= self.drone_speed
        elif action == 4:  # right
            self.drone_pos[0] += self.drone_speed
        # action == 0: stay (no movement)
        
        # Check if drone is out of bounds
        out_of_bounds = (
            self.drone_pos[0] < 0 or 
            self.drone_pos[0] > self.canvas_size or
            self.drone_pos[1] < 0 or 
            self.drone_pos[1] > self.canvas_size
        )
        
        if out_of_bounds:
            # Penalize and revert position
            reward = -1.0
            self.drone_pos = previous_pos
        else:
            # Calculate distance to balloon
            distance = np.linalg.norm(self.drone_pos - self.balloon_pos)
            
            # Check if drone hit the balloon
            if distance < self.hit_distance:
                # Balloon popped! Big reward
                reward = 10.0
                self.score += 1
                
                # Spawn new balloon if not reached max score
                if self.score < self.max_score:
                    self.balloon_pos = self._spawn_balloon()
            else:
                # Small penalty for each step to encourage efficiency
                # Give slight reward for getting closer to balloon
                prev_distance = np.linalg.norm(previous_pos - self.balloon_pos)
                distance_reward = (prev_distance - distance) * 0.01
                reward = -0.01 + distance_reward
        
        # Move balloon randomly (makes it harder!)
        self._move_balloon_randomly()
        
        # Check if episode is done
        terminated = (self.score >= self.max_score)
        truncated = (self.steps >= self.max_steps_per_episode)
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _spawn_balloon(self):
        """Spawn balloon at random position away from drone"""
        while True:
            pos = np.array([
                self.np_random.integers(30, self.canvas_size - 30),
                self.np_random.integers(30, self.canvas_size - 30)
            ], dtype=np.float32)
            
            # Ensure balloon spawns at least 100 pixels away from drone
            distance = np.linalg.norm(pos - self.drone_pos)
            if distance > 100:
                return pos
    
    def _move_balloon_randomly(self):
        """Move balloon slightly in random direction"""
        if self.np_random.random() < 0.2:  # 20% chance to move each step
            move = self.np_random.integers(-5, 6, size=2)
            new_pos = self.balloon_pos + move
            
            # Keep balloon within bounds
            new_pos[0] = np.clip(new_pos[0], 20, self.canvas_size - 20)
            new_pos[1] = np.clip(new_pos[1], 20, self.canvas_size - 20)
            
            self.balloon_pos = new_pos
    
    def _get_observation(self):
        """Get current observation (normalized positions)"""
        return np.array([
            self.drone_pos[0] / self.canvas_size,
            self.drone_pos[1] / self.canvas_size,
            self.balloon_pos[0] / self.canvas_size,
            self.balloon_pos[1] / self.canvas_size
        ], dtype=np.float32)
    
    def _get_info(self):
        """Get additional info"""
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_balloon": np.linalg.norm(self.drone_pos - self.balloon_pos)
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Draw the current state using pygame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.canvas_size, self.canvas_size))
            pygame.display.set_caption("Drone Balloon Chase")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create canvas
        canvas = pygame.Surface((self.canvas_size, self.canvas_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Draw balloon (red circle)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),  # Red
            self.balloon_pos.astype(int),
            self.balloon_size
        )
        
        # Draw drone (blue square)
        drone_rect = pygame.Rect(
            self.drone_pos[0] - self.drone_size // 2,
            self.drone_pos[1] - self.drone_size // 2,
            self.drone_size,
            self.drone_size
        )
        pygame.draw.rect(canvas, (0, 0, 255), drone_rect)  # Blue
        
        # Draw drone propellers (just for fun)
        pygame.draw.line(
            canvas,
            (0, 0, 0),
            (self.drone_pos[0] - 15, self.drone_pos[1]),
            (self.drone_pos[0] + 15, self.drone_pos[1]),
            2
        )
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}/{self.max_score}", True, (0, 0, 0))
        canvas.blit(score_text, (10, 10))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up pygame"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Test the environment
if __name__ == "__main__":
    env = DroneBalloonEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Testing environment with random actions...")
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished! Score: {info['score']}")
            obs, info = env.reset()
    
    env.close()