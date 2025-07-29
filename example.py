#!/usr/bin/env python3
"""
Example script demonstrating SO-ARM100 single-arm manipulation environment
"""

import numpy as np
import gymnasium as gym
import gym_soarm
import cv2
import os
from datetime import datetime


def save_frames_to_mp4(frames_dict, output_dir="videos", fps=30):
    """Save frames dictionary to MP4 videos for each camera"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    video_writers = {}
    
    for camera_name, frames in frames_dict.items():
        if len(frames) == 0:
            continue
            
        # Define output path
        output_path = os.path.join(output_dir, f"{camera_name}_{timestamp}.mp4")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Saved {len(frames)} frames to {output_path}")


def main():
    print("Creating SO-ARM101 single-arm manipulation environment...")
    
    # Create environment using gym.make with pixels_agent_pos for both visual and state info
    # camera_config options: 'front_only', 'front_wrist', 'all'
    env = gym.make('SoArm-v0', render_mode='human', obs_type='pixels_agent_pos', camera_config='front_wrist')
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Initialize frames storage for video recording
    frames_storage = {}
    
    # Test grid position functionality
    print("\n=== Testing Grid Position Control ===")
    print("Grid layout (0-8):")
    print("0: (-10cm, -7.5cm)  1: (-10cm,  0cm)   2: (-10cm, +7.5cm)")
    print("3: ( 0cm,  -7.5cm)  4: ( 0cm,   0cm)   5: ( 0cm,  +7.5cm)")
    print("6: (+10cm, -7.5cm)  7: (+10cm,  0cm)   8: (+10cm, +7.5cm)")
    
    # Test with specific position (position 4 = center)
    print(f"\nResetting environment with cube at grid position 4 (center)...")
    observation, info = env.reset(seed=42, options={'cube_grid_position': 4})
    
    print(f"Initial observation keys: {observation.keys()}")
    if "agent_pos" in observation:
        print(f"Initial joint positions: {observation['agent_pos']}")
    if "pixels" in observation:
        print(f"Available camera views: {list(observation['pixels'].keys())}")
        # Initialize frames storage for each camera
        for camera_name in observation['pixels'].keys():
            frames_storage[camera_name] = []
            # Store first frame
            frames_storage[camera_name].append(observation['pixels'][camera_name].copy())
    
    # Initial render to show GUI viewer
    env.render()
    
    # Run a few steps with random actions
    print("\nRunning simulation with cube at position 4...")
    print("GUI viewer should now be visible. Use keys 1/2/3 to switch cameras, 'q' to quit.")
    
    for step in range(200):
        # Sample random action within joint limits
        action = env.action_space.sample()
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Store frames from observation
        if "pixels" in observation:
            for camera_name, frame in observation['pixels'].items():
                frames_storage[camera_name].append(frame.copy())
        
        # Render after each step to update GUI
        env.render()
        
        print(f"Step {step+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    # Test random position
    print(f"\n=== Testing Random Position ===")
    print("Resetting environment with random cube position...")
    observation, info = env.reset(seed=None, options={'cube_grid_position': None})
    
    # Store additional frames from random position test
    if "pixels" in observation:
        for camera_name, frame in observation['pixels'].items():
            frames_storage[camera_name].append(frame.copy())
    
    env.render()
    
    # Test different specific positions
    print(f"\n=== Testing Different Grid Positions ===")
    for test_position in [0, 2, 6, 8]:  # Test corner positions
        print(f"\nTesting position {test_position}...")
        observation, info = env.reset(seed=42, options={'cube_grid_position': test_position})
        
        # Store frames from different positions
        if "pixels" in observation:
            for camera_name, frame in observation['pixels'].items():
                frames_storage[camera_name].append(frame.copy())
        
        env.render()
        
        # Give some time to observe the position
        for _ in range(5):
            action = env.action_space.sample() * 0.1  # Small movements
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Store frames from movements
            if "pixels" in observation:
                for camera_name, frame in observation['pixels'].items():
                    frames_storage[camera_name].append(frame.copy())
            
            env.render()
    
    env.close()
    
    # Save all recorded frames to MP4 videos
    print(f"\n=== Saving Videos ===")
    if frames_storage:
        save_frames_to_mp4(frames_storage)
        print(f"Video files saved for cameras: {list(frames_storage.keys())}")
    else:
        print("No frames were recorded!")
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
