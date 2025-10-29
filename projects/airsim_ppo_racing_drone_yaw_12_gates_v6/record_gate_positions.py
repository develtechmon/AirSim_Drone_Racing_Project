"""
Gate Position Recorder - Pygame (Simplified - No Threading)

Uses direct velocity control without threading to avoid AirSim conflicts.

CONTROLS:
  W/S - Forward/Backward
  A/D - Left/Right
  U/I - Up/Down
  Q/E - Yaw Left/Right
  SPACE - Capture gate position
  +/- - Adjust speed
  R - Reset
  ESC - Quit

REQUIREMENTS:
  pip install pygame
"""

import airsim
import numpy as np
import time
import pygame
import sys


class GatePositionRecorder:
    """Simplified pygame-based recorder without threading"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Window setup
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gate Position Recorder")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (100, 150, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (150, 150, 150)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Connect to AirSim
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Movement parameters
        self.speed = 2.0
        self.yaw_rate = 30.0
        
        # Current velocity commands
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_cmd = 0.0
        
        # Recorded gate positions
        self.gate_positions = []
        self.current_gate_num = 0
        
        # UI state
        self.status_message = "Ready!"
        self.status_color = self.GREEN
        self.last_action_time = time.time()
        
        # Running flag
        self.running = True
        
        print("✅ Connected to AirSim!")
        print("✅ Pygame window opened!")
    
    def set_status(self, message, color=None):
        """Update status message"""
        self.status_message = message
        self.status_color = color if color else self.GREEN
        self.last_action_time = time.time()
    
    def takeoff(self):
        """Take off and hover"""
        print("🚁 Taking off...")
        self.set_status("Taking off...", self.YELLOW)
        self.client.takeoffAsync().join()
        time.sleep(2)
        self.set_status("Hovering - Ready!", self.GREEN)
        print("✅ Ready!")
    
    def land(self):
        """Land the drone"""
        print("🛬 Landing...")
        self.set_status("Landing...", self.YELLOW)
        self.client.landAsync().join()
        self.set_status("Landed!", self.GREEN)
        print("✅ Landed!")
    
    def get_position(self):
        """Get current drone position"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])
    
    def get_yaw(self):
        """Get current yaw angle in degrees"""
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        
        w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        yaw_rad = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_deg = np.degrees(yaw_rad)
        
        return yaw_deg
    
    def get_velocity(self):
        """Get current drone velocity"""
        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])
    
    def capture_gate_position(self):
        """Record current position as a gate"""
        pos = self.get_position()
        yaw = self.get_yaw()
        
        self.gate_positions.append(pos.tolist())
        self.current_gate_num += 1
        
        self.set_status(f"🎯 Gate {self.current_gate_num} captured!", self.GREEN)
        
        print(f"\n{'='*60}")
        print(f"🎯 GATE {self.current_gate_num} CAPTURED!")
        print(f"{'='*60}")
        print(f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        print(f"Yaw:      {yaw:.1f}°")
        print(f"{'='*60}\n")
    
    def delete_last_gate(self):
        """Delete the last recorded gate"""
        if self.gate_positions:
            deleted = self.gate_positions.pop()
            self.current_gate_num -= 1
            self.set_status(f"🗑️ Deleted Gate {self.current_gate_num + 1}", self.RED)
            print(f"\n🗑️ Deleted Gate {self.current_gate_num + 1}: {deleted}")
        else:
            self.set_status("❌ No gates to delete!", self.RED)
            print("\n❌ No gates to delete!")
    
    def clear_all_gates(self):
        """Clear all recorded gates"""
        if self.gate_positions:
            count = len(self.gate_positions)
            self.gate_positions = []
            self.current_gate_num = 0
            self.set_status(f"🗑️ Cleared {count} gates!", self.RED)
            print(f"\n🗑️ Cleared all {count} gates!")
        else:
            self.set_status("❌ No gates to clear!", self.RED)
            print("\n❌ No gates to clear!")
    
    def handle_keys(self):
        """Handle keyboard input and send velocity commands"""
        keys = pygame.key.get_pressed()
        
        # Reset velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_cmd = 0.0
        
        status_updated = False
        
        # Movement keys
        if keys[pygame.K_w]:
            self.vx = self.speed
            if not status_updated:
                self.set_status("⬆️ Forward", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_s]:
            self.vx = -self.speed
            if not status_updated:
                self.set_status("⬇️ Backward", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_a]:
            self.vy = -self.speed
            if not status_updated:
                self.set_status("⬅️ Left", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_d]:
            self.vy = self.speed
            if not status_updated:
                self.set_status("➡️ Right", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_u]:
            self.vz = -self.speed
            if not status_updated:
                self.set_status("⬆️ Up", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_i]:
            self.vz = self.speed
            if not status_updated:
                self.set_status("⬇️ Down", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_q]:
            self.yaw_cmd = -self.yaw_rate
            if not status_updated:
                self.set_status("↺ Yaw Left", self.BLUE)
                status_updated = True
            
        if keys[pygame.K_e]:
            self.yaw_cmd = self.yaw_rate
            if not status_updated:
                self.set_status("↻ Yaw Right", self.BLUE)
                status_updated = True
        
        # Send velocity command (non-blocking, no .join())
        self.client.moveByVelocityAsync(
            float(self.vx),
            float(self.vy),
            float(self.vz),
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(self.yaw_cmd))
        )
        
        # Show hovering if no keys pressed
        if not status_updated and time.time() - self.last_action_time > 2.0:
            self.set_status("Hovering...", self.GREEN)
    
    def draw_ui(self):
        """Draw the user interface"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Title
        title = self.font_large.render("Gate Position Recorder", True, self.WHITE)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 20))
        
        # Get current state
        pos = self.get_position()
        yaw = self.get_yaw()
        vel = self.get_velocity()
        speed = np.linalg.norm(vel)
        
        # Left column: Current State
        y_offset = 80
        pygame.draw.rect(self.screen, self.GRAY, (20, y_offset, 380, 220), 2)
        
        pos_title = self.font_medium.render("Current State", True, self.YELLOW)
        self.screen.blit(pos_title, (30, y_offset + 10))
        
        # Position
        pos_x = self.font_small.render(f"X: {pos[0]:>7.1f} m", True, self.WHITE)
        pos_y = self.font_small.render(f"Y: {pos[1]:>7.1f} m", True, self.WHITE)
        pos_z = self.font_small.render(f"Z: {pos[2]:>7.1f} m", True, self.GREEN if pos[2] < 0 else self.RED)
        yaw_text = self.font_small.render(f"Yaw: {yaw:>6.1f}°", True, self.WHITE)
        
        self.screen.blit(pos_x, (30, y_offset + 50))
        self.screen.blit(pos_y, (30, y_offset + 80))
        self.screen.blit(pos_z, (30, y_offset + 110))
        self.screen.blit(yaw_text, (30, y_offset + 140))
        
        # Velocity
        vel_text = self.font_small.render(f"Speed: {speed:>5.2f} m/s", True, self.BLUE)
        vel_cmd = self.font_small.render(
            f"Cmd: vx={self.vx:.1f} vy={self.vy:.1f} vz={self.vz:.1f}",
            True, self.GRAY
        )
        self.screen.blit(vel_text, (30, y_offset + 170))
        self.screen.blit(vel_cmd, (30, y_offset + 195))
        
        # Z warning
        if pos[2] > 0:
            warning = self.font_small.render("⚠️ Underground!", True, self.RED)
            self.screen.blit(warning, (220, y_offset + 110))
        
        # Right column: Recorded Gates
        y_offset = 80
        pygame.draw.rect(self.screen, self.GRAY, (420, y_offset, 360, 220), 2)
        
        gates_title = self.font_medium.render(f"Recorded Gates ({self.current_gate_num})", True, self.YELLOW)
        self.screen.blit(gates_title, (430, y_offset + 10))
        
        # Display recorded gates (last 6)
        if self.gate_positions:
            start_idx = max(0, len(self.gate_positions) - 6)
            display_gates = self.gate_positions[start_idx:]
            
            for i, gate_pos in enumerate(display_gates):
                gate_num = start_idx + i
                gate_text = self.font_small.render(
                    f"Gate {gate_num}: [{gate_pos[0]:.1f}, {gate_pos[1]:.1f}, {gate_pos[2]:.1f}]",
                    True, self.GREEN
                )
                self.screen.blit(gate_text, (430, y_offset + 45 + i * 28))
        else:
            no_gates = self.font_small.render("No gates recorded yet", True, self.GRAY)
            self.screen.blit(no_gates, (430, y_offset + 50))
        
        # Speed display
        speed_text = self.font_small.render(f"Set Speed: {self.speed:.1f} m/s (+/- to adjust)", True, self.YELLOW)
        self.screen.blit(speed_text, (430, y_offset + 195))
        
        # Controls box
        y_offset = 320
        pygame.draw.rect(self.screen, self.GRAY, (20, y_offset, 760, 165), 2)
        
        controls_title = self.font_medium.render("Controls (HOLD keys for movement)", True, self.YELLOW)
        self.screen.blit(controls_title, (30, y_offset + 10))
        
        controls = [
            "W/S - Forward/Back    A/D - Left/Right    U/I - Up/Down    Q/E - Yaw",
            "SPACE - Capture Gate    BACKSPACE - Delete Last    C - Clear All",
            "+/- - Speed    R - Reset Drone    P - Print to Console    ESC - Quit"
        ]
        
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, self.WHITE)
            self.screen.blit(text, (30, y_offset + 45 + i * 35))
        
        # Status message
        y_offset = 505
        status = self.font_medium.render(self.status_message, True, self.status_color)
        self.screen.blit(status, (self.width // 2 - status.get_width() // 2, y_offset))
        
        # Instructions
        y_offset = 550
        hint = self.font_small.render("Keep window focused • HOLD movement keys • Tap action keys", True, self.GRAY)
        self.screen.blit(hint, (self.width // 2 - hint.get_width() // 2, y_offset))
        
        # Update display
        pygame.display.flip()
    
    def show_summary(self):
        """Display captured gate positions"""
        print("\n" + "="*70)
        print("📊 CAPTURED GATE POSITIONS")
        print("="*70)
        
        if not self.gate_positions:
            print("❌ No gates captured!")
            return
        
        print(f"\nTotal gates: {len(self.gate_positions)}\n")
        
        print("gate_positions = [")
        for i, pos in enumerate(self.gate_positions):
            print(f"    [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}],  # Gate {i}")
        print("]")
        
        print("\n" + "="*70)
        print("✅ Copy the above list into your training script!")
        print("="*70 + "\n")
    
    def run(self):
        """Main control loop"""
        # Takeoff
        self.takeoff()
        
        print("\n" + "="*70)
        print("🎮 PYGAME WINDOW IS OPEN!")
        print("="*70)
        print("HOLD keys for continuous movement (not tap!)")
        print("Press ESC to quit and see results")
        print("="*70 + "\n")
        
        # Main pygame loop
        clock = pygame.time.Clock()
        space_pressed = False
        backspace_pressed = False
        c_pressed = False
        p_pressed = False
        r_pressed = False
        plus_pressed = False
        minus_pressed = False
        
        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                            
                        elif event.key == pygame.K_SPACE and not space_pressed:
                            self.capture_gate_position()
                            space_pressed = True
                            
                        elif event.key == pygame.K_BACKSPACE and not backspace_pressed:
                            self.delete_last_gate()
                            backspace_pressed = True
                            
                        elif event.key == pygame.K_c and not c_pressed:
                            self.clear_all_gates()
                            c_pressed = True
                            
                        elif event.key == pygame.K_p and not p_pressed:
                            self.show_summary()
                            p_pressed = True
                            
                        elif event.key == pygame.K_r and not r_pressed:
                            self.set_status("🔄 Resetting...", self.YELLOW)
                            self.client.reset()
                            time.sleep(1)
                            self.client.enableApiControl(True)
                            self.client.armDisarm(True)
                            self.takeoff()
                            r_pressed = True
                            
                        elif event.key in [pygame.K_PLUS, pygame.K_EQUALS] and not plus_pressed:
                            self.speed = min(5.0, self.speed + 0.5)
                            self.set_status(f"Speed: {self.speed:.1f} m/s", self.YELLOW)
                            plus_pressed = True
                            
                        elif event.key == pygame.K_MINUS and not minus_pressed:
                            self.speed = max(0.5, self.speed - 0.5)
                            self.set_status(f"Speed: {self.speed:.1f} m/s", self.YELLOW)
                            minus_pressed = True
                    
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_SPACE:
                            space_pressed = False
                        elif event.key == pygame.K_BACKSPACE:
                            backspace_pressed = False
                        elif event.key == pygame.K_c:
                            c_pressed = False
                        elif event.key == pygame.K_p:
                            p_pressed = False
                        elif event.key == pygame.K_r:
                            r_pressed = False
                        elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                            plus_pressed = False
                        elif event.key == pygame.K_MINUS:
                            minus_pressed = False
                
                # Handle continuous movement keys and send velocity commands
                self.handle_keys()
                
                # Draw UI
                self.draw_ui()
                
                # Cap at 30 FPS (sends velocity commands 30 times/second)
                clock.tick(30)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Stop movement
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1)
            
            # Clean up
            print("\n🛬 Landing and cleaning up...")
            try:
                self.land()
            except:
                pass
            
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            
            pygame.quit()
            
            # Show results
            self.show_summary()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🚁 GATE POSITION RECORDER (Pygame - Simplified)")
    print("="*70)
    print()
    
    # Check pygame
    try:
        import pygame
    except ImportError:
        print("❌ ERROR: pygame not installed!")
        print("   pip install pygame")
        return
    
    # Check AirSim
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.ping()
        print("✅ AirSim detected!")
        print()
    except:
        print("❌ ERROR: Cannot connect to AirSim!")
        print("   Please start AirSim first.")
        return
    
    # Run recorder
    recorder = GatePositionRecorder()
    recorder.run()
    
    print("\n👋 Done!")


if __name__ == "__main__":
    main()