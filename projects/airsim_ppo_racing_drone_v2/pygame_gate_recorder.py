"""
Pygame Drone Control + Gate Position Recorder
(No admin rights needed!)

Controls:
- W/S: Forward/Backward
- A/D: Left/Right  
- Q/E: Up/Down
- Arrow Keys: Rotate
- SPACEBAR: Record gate position
- R: Reset
- P: Print positions
- ESC: Quit

Use this code to find the positions of your gates in AirSim.

[5.8, -5.3, -0.7]
 ^^^   ^^^   ^^^
  |     |     |
  x     y     z

Keep this window focused to use controls!
"""

import airsim
import time
import sys
import threading

# Try pygame (better than keyboard, no admin needed)
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("Installing pygame...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
    import pygame
    HAS_PYGAME = True


class PygameDroneController:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Drone Controller - Gate Position Recorder")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Control parameters
        self.speed = 2.0
        self.yaw_rate = 30.0
        self.running = True
        
        # Gate positions
        self.gate_positions = []
        
        # Current commands
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_cmd = 0.0
        
        # Status
        self.last_action = "Ready"
        
    def draw_ui(self):
        """Draw control interface."""
        self.screen.fill((20, 20, 30))
        
        # Title
        title = self.font.render("DRONE GATE POSITION RECORDER", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Instructions
        y = 70
        instructions = [
            "CONTROLS:",
            "  W/S - Forward/Back    A/D - Left/Right",
            "  Q/E - Up/Down         ←/→ - Rotate",
            "  SPACE - Record Gate   R - Reset",
            "  P - Print             ESC - Quit",
            "",
            f"SPEED: {self.speed:.1f} m/s  (+/- to adjust)",
            "",
            f"Gates Recorded: {len(self.gate_positions)}/7",
            f"Last Action: {self.last_action}",
        ]
        
        for line in instructions:
            text = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(text, (20, y))
            y += 25
        
        # Recorded positions
        y += 20
        if self.gate_positions:
            header = self.font.render("RECORDED GATES:", True, (100, 255, 100))
            self.screen.blit(header, (20, y))
            y += 30
            
            for i, pos in enumerate(self.gate_positions):
                text = self.small_font.render(f"Gate {i}: {pos}", True, (150, 255, 150))
                self.screen.blit(text, (40, y))
                y += 20
        
        # Current velocity display
        y = 500
        vel_text = self.small_font.render(
            f"Current: vx={self.vx:.1f} vy={self.vy:.1f} vz={self.vz:.1f} yaw={self.yaw_cmd:.0f}°/s",
            True, (150, 150, 255)
        )
        self.screen.blit(vel_text, (20, y))
        
        pygame.display.flip()
    
    def handle_keys(self):
        """Handle keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Movement
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_cmd = 0.0
        
        if keys[pygame.K_w]:
            self.vx = self.speed
        if keys[pygame.K_s]:
            self.vx = -self.speed
        if keys[pygame.K_a]:
            self.vy = -self.speed
        if keys[pygame.K_d]:
            self.vy = self.speed
        if keys[pygame.K_q]:
            self.vz = -self.speed
        if keys[pygame.K_e]:
            self.vz = self.speed
        if keys[pygame.K_LEFT]:
            self.yaw_cmd = -self.yaw_rate
        if keys[pygame.K_RIGHT]:
            self.yaw_cmd = self.yaw_rate
    
    def control_loop(self):
        """Send velocity commands."""
        while self.running:
            self.client.moveByVelocityAsync(
                self.vx, self.vy, self.vz,
                duration=0.1,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_cmd)
            )
            time.sleep(0.05)
    
    def record_position(self):
        """Record current position."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        position = [
            round(pos.x_val, 1),
            round(pos.y_val, 1),
            round(pos.z_val, 1)
        ]
        
        self.gate_positions.append(position)
        gate_num = len(self.gate_positions) - 1
        
        self.last_action = f"Gate {gate_num} recorded: {position}"
        print(f"\n✅ {self.last_action}")
        
        if len(self.gate_positions) >= 7:
            print("\n🎉 All 7 gates recorded!")
    
    def print_positions(self):
        """Print positions to console."""
        print("\n" + "="*60)
        print("GATE POSITIONS")
        print("="*60)
        print("\ngate_positions = [")
        for i, pos in enumerate(self.gate_positions):
            print(f"    {pos},  # Gate {i}")
        print("]\n" + "="*60)
    
    def reset_positions(self):
        """Clear positions."""
        self.gate_positions = []
        self.last_action = "Positions cleared"
        print(f"\n🗑️  {self.last_action}")
    
    def adjust_speed(self, delta):
        """Adjust speed."""
        self.speed = max(0.5, min(5.0, self.speed + delta))
        self.last_action = f"Speed: {self.speed:.1f} m/s"
    
    def run(self):
        """Main loop."""
        # Takeoff
        print("\n🚁 Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(1)
        print("✅ Ready! Keep pygame window focused!\n")
        
        # Start control thread
        control_thread = threading.Thread(target=self.control_loop)
        control_thread.daemon = True
        control_thread.start()
        
        # Main pygame loop
        space_pressed = False
        r_pressed = False
        p_pressed = False
        plus_pressed = False
        minus_pressed = False
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE and not space_pressed:
                        self.record_position()
                        space_pressed = True
                    elif event.key == pygame.K_r and not r_pressed:
                        self.reset_positions()
                        r_pressed = True
                    elif event.key == pygame.K_p and not p_pressed:
                        self.print_positions()
                        p_pressed = True
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS] and not plus_pressed:
                        self.adjust_speed(0.5)
                        plus_pressed = True
                    elif event.key == pygame.K_MINUS and not minus_pressed:
                        self.adjust_speed(-0.5)
                        minus_pressed = True
                
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        space_pressed = False
                    elif event.key == pygame.K_r:
                        r_pressed = False
                    elif event.key == pygame.K_p:
                        p_pressed = False
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        plus_pressed = False
                    elif event.key == pygame.K_MINUS:
                        minus_pressed = False
            
            self.handle_keys()
            self.draw_ui()
            self.clock.tick(30)
        
        # Cleanup
        print("\n🛬 Landing...")
        self.client.landAsync().join()
        
        # Final output
        self.print_positions()
        
        # Save
        if self.gate_positions:
            with open("gate_positions.txt", "w") as f:
                f.write("# Copy this into your code:\n\n")
                f.write("gate_positions = [\n")
                for i, pos in enumerate(self.gate_positions):
                    f.write(f"    {pos},  # Gate {i}\n")
                f.write("]\n")
            print("💾 Saved to gate_positions.txt")
        
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        pygame.quit()
        print("\n✅ Done!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PYGAME DRONE CONTROLLER")
    print("="*60)
    print("\n📋 A pygame window will open.")
    print("   Keep that window FOCUSED to use controls!")
    print("   Fly to each gate and press SPACEBAR to record.\n")
    
    input("Press ENTER to start...")
    
    controller = PygameDroneController()
    controller.run()