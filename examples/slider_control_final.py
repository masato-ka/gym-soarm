#!/usr/bin/env python3
"""
SO-ARM101 Slider Control Sample
Simple slider-based joint position control for gym-soarm environment.
"""

import sys
import cv2
import gymnasium as gym
import numpy as np

# Import gym-soarm
import gym_soarm
from gym_soarm.constants import JOINTS, JOINT_LIMITS


class SoArmSliderControl:
    def __init__(self):
        # Create environment
        self.env = gym.make('SoArm-v0', render_mode='human')
        
        # Initialize joint positions to zero (neutral position)
        self.joint_positions = np.zeros(6)
        
        # Control flags
        self.running = True
        
        # Create control window
        self.setup_control_window()
        
        # Reset environment initially
        self.reset_environment()
    
    def setup_control_window(self):
        """Create OpenCV control window with trackbars"""
        self.window_name = "SO-ARM101 Joint Control"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # Create control panel image
        self.control_height = 500
        self.control_width = 600
        
        # Create trackbars for each joint
        for i, joint_name in enumerate(JOINTS):
            min_val, max_val = JOINT_LIMITS[joint_name]
            
            # Convert to integer scale (multiply by 1000 for precision)
            min_int = int(min_val * 1000)
            max_int = int(max_val * 1000)
            initial_val = 0  # Start at 0
            
            cv2.createTrackbar(f"{joint_name}", self.window_name, 
                             initial_val - min_int, max_int - min_int, 
                             self.on_trackbar_change)
        
        # Add reset button as trackbar (0=no action, 1=reset)
        cv2.createTrackbar("Reset (0->1)", self.window_name, 0, 1, self.on_reset_trackbar)
        
        # Update control panel
        self.update_control_panel()
    
    def on_trackbar_change(self, val):
        """Update joint positions when trackbars change"""
        for i, joint_name in enumerate(JOINTS):
            min_val, max_val = JOINT_LIMITS[joint_name]
            min_int = int(min_val * 1000)
            
            # Get trackbar value and convert back to radians
            trackbar_val = cv2.getTrackbarPos(f"{joint_name}", self.window_name)
            joint_val = (trackbar_val + min_int) / 1000.0
            
            self.joint_positions[i] = joint_val
        
        # Update control panel display
        self.update_control_panel()
    
    def on_reset_trackbar(self, val):
        """Handle reset button"""
        if val == 1:
            self.reset_environment()
            # Reset trackbar back to 0
            cv2.setTrackbarPos("Reset (0->1)", self.window_name, 0)
    
    def update_control_panel(self):
        """Update the control panel display"""
        # Create control panel image
        panel = np.ones((self.control_height, self.control_width, 3), dtype=np.uint8) * 240
        
        # Title
        cv2.putText(panel, "SO-ARM101 Joint Control", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Joint values
        y_pos = 70
        for i, joint_name in enumerate(JOINTS):
            text = f"{joint_name}: {self.joint_positions[i]:.3f} rad"
            cv2.putText(panel, text, (20, y_pos + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Instructions
        instructions = [
            "Instructions:",
            "- Use trackbars above to control joints",
            "- Set 'Reset' trackbar to 1 to reset environment", 
            "- Press ESC key to exit",
            "- Press SPACE to step simulation",
            "- Initial position: all joints at 0.0 rad"
        ]
        
        y_start = 260
        for i, instruction in enumerate(instructions):
            color = (0, 0, 0) if i == 0 else (50, 50, 50)
            thickness = 2 if i == 0 else 1
            cv2.putText(panel, instruction, (20, y_start + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # Current joint limits info
        limits_y = 420
        cv2.putText(panel, "Joint Limits (rad):", (20, limits_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow(self.window_name, panel)
    
    def reset_environment(self):
        """Reset environment to initial state"""
        try:
            # Reset all trackbars to 0
            for joint_name in JOINTS:
                min_val, _ = JOINT_LIMITS[joint_name]
                min_int = int(min_val * 1000)
                cv2.setTrackbarPos(f"{joint_name}", self.window_name, 0 - min_int)
            
            # Update joint positions
            self.joint_positions = np.zeros(6)
            
            # Reset environment
            obs, info = self.env.reset()
            
            # Update display
            self.update_control_panel()
            
            print("Environment reset to initial position")
            
        except Exception as e:
            print(f"Error resetting environment: {e}")
    
    def step_simulation(self):
        """Step the simulation once"""
        try:
            # Apply current joint positions as action
            action = self.joint_positions.copy()
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Render
            self.env.render()
            
            # Handle episode termination
            if terminated or truncated:
                print("Episode terminated, resetting...")
                obs, info = self.env.reset()
                
        except Exception as e:
            print(f"Simulation error: {e}")
    
    def run(self):
        """Start the control application"""
        print("Starting SO-ARM101 Slider Control...")
        print("Instructions:")
        print("- Use trackbars to control each joint")
        print("- Set 'Reset' trackbar to 1 to reset environment")
        print("- Press ESC key to exit application")
        print("- Initial position: all joints at 0.0 radians")
        
        try:
            while self.running:
                # Update display
                self.update_control_panel()
                
                # Check for key press
                key = cv2.waitKey(30) & 0xFF
                
                if key == 27:  # ESC key
                    print("ESC pressed, exiting...")
                    break
                elif key == ord('r') or key == ord('R'):  # R key for reset
                    self.reset_environment()

                self.step_simulation()

            self.stop_application()
            
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting...")
            self.stop_application()
    
    def stop_application(self):
        """Stop the application"""
        print("Stopping application...")
        self.running = False
        
        try:
            self.env.close()
        except:
            pass
        
        cv2.destroyAllWindows()


def main():
    """Main function"""
    try:
        app = SoArmSliderControl()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()