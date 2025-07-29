#!/usr/bin/env python3
"""
Example script demonstrating SO-ARM100 single-arm manipulation environment
"""

import numpy as np
import gymnasium as gym
import gym_soarm


def main():
    print("Creating SO-ARM101 single-arm manipulation environment...")
    
    # Create environment using gym.make with pixels_agent_pos for both visual and state info
    env = gym.make('SoArm-v0', render_mode='human', obs_type='pixels_agent_pos')
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
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
    
    # Initial render to show GUI viewer
    env.render()
    
    # Run a few steps with random actions
    print("\nRunning simulation with cube at position 4...")
    print("GUI viewer should now be visible. Use keys 1/2/3 to switch cameras, 'q' to quit.")
    
    for step in range(20):
        # Sample random action within joint limits
        action = env.action_space.sample()
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
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
    env.render()
    
    # Test different specific positions
    print(f"\n=== Testing Different Grid Positions ===")
    for test_position in [0, 2, 6, 8]:  # Test corner positions
        print(f"\nTesting position {test_position}...")
        observation, info = env.reset(seed=42, options={'cube_grid_position': test_position})
        env.render()
        
        # Give some time to observe the position
        for _ in range(5):
            action = env.action_space.sample() * 0.1  # Small movements
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
    
    env.close()
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
