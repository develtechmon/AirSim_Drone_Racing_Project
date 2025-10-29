"""
Quick Gate Position Checker

Run this to see if your gate positions are correct.
The drone will fly to each gate position you specified.
If it goes to the wrong place, your positions are wrong!

[5.8, -5.3, -0.7]
 ^^^   ^^^   ^^^
  |     |     |
  x     y     z
  
"""

import airsim
import time

# YOUR CURRENT GATE POSITIONS (from the code)
gate_positions = [
    [5.8, -5.3, -0.7],      # Gate 0
    [17.3, -7.9, 1.0],      # Gate 1
    [28.9, -7.9, 1.1],      # Gate 2
    [39.3, -5.6, 1.3],      # Gate 3
    [46.3, 0.8, 1.1],       # Gate 4
    [46.3, 10.3, 0.7],      # Gate 5
    [39.5, 18.0, 0.8],      # Gate 6
    [30.9, 21.0, 1.3],      # Gate 7 (your "current position")
    [19.2, 22.7, 1.0],      # Gate 8
    [9.0, 19.8, 0.7]        # Gate 9
]

def check_gate_positions():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("="*60)
    print("GATE POSITION CHECKER")
    print("="*60)
    print("\nThis will fly the drone to each gate position.")
    print("Watch in AirSim - does it go to the actual gates?")
    print("="*60)
    
    # Takeoff
    print("\n🚁 Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)
    
    for i, gate_pos in enumerate(gate_positions):
        print(f"\n📍 Flying to Gate {i} at {gate_pos}...")
        
        client.moveToPositionAsync(
            float(gate_pos[0]),
            float(gate_pos[1]),
            float(gate_pos[2]),
            velocity=3.0
        ).join()
        
        # Get actual position
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        actual = [pos.x_val, pos.y_val, pos.z_val]
        
        print(f"  Target:  {gate_pos}")
        print(f"  Arrived: [{actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}]")
        
        # Check distance to nearest gate visually
        input(f"  👀 Look at AirSim - is drone at Gate {i}? Press ENTER...")
    
    print("\n" + "="*60)
    print("CHECK COMPLETE")
    print("="*60)
    print("\nIf drone went to wrong places, your gate positions are WRONG!")
    print("Run 'python find_gate_positions.py' to find correct positions.")
    
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    print("\n⚠️  WARNING: Drone will fly automatically!")
    print("Make sure AirSim is running and environment is clear.\n")
    
    choice = input("Press ENTER to start, or 'q' to cancel: ").strip()
    if choice.lower() != 'q':
        check_gate_positions()
    else:
        print("Cancelled.")