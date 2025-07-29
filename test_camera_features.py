#!/usr/bin/env python3
"""Simple test script to verify camera configuration features."""

import numpy as np
import gymnasium as gym
import gym_soarm


def test_camera_configurations():
    """Test different camera configurations."""
    print("Testing camera configuration features...")
    
    configs = ['front_only', 'front_wrist', 'all']
    
    for config in configs:
        print(f"\n--- Testing {config} configuration ---")
        
        try:
            # Test pixels observation
            env = gym.make('SoArm-v0', obs_type='pixels', camera_config=config)
            obs, info = env.reset(seed=42)
            
            print(f"Observation keys: {list(obs.keys())}")
            print(f"Expected cameras: {env._camera_names}")
            
            # Verify observation structure
            assert set(obs.keys()) == set(env._camera_names), f"Camera mismatch for {config}"
            
            for camera_name in obs.keys():
                shape = obs[camera_name].shape
                dtype = obs[camera_name].dtype
                print(f"  {camera_name}: shape={shape}, dtype={dtype}")
                assert shape == (480, 640, 3), f"Wrong shape for {camera_name}"
                assert dtype == np.uint8, f"Wrong dtype for {camera_name}"
            
            env.close()
            print(f"✅ {config} configuration passed")
            
        except Exception as e:
            print(f"❌ {config} configuration failed: {e}")
    
    # Test pixels_agent_pos with front_wrist
    print(f"\n--- Testing pixels_agent_pos with front_wrist ---")
    try:
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_wrist')
        obs, info = env.reset(seed=42)
        
        print(f"Top-level keys: {list(obs.keys())}")
        print(f"Pixels keys: {list(obs['pixels'].keys())}")
        print(f"Agent pos shape: {obs['agent_pos'].shape}")
        
        # Verify structure
        assert 'pixels' in obs and 'agent_pos' in obs
        assert set(obs['pixels'].keys()) == {'front_camera', 'wrist_camera'}
        assert obs['agent_pos'].shape == (6,)
        
        env.close()
        print("✅ pixels_agent_pos with front_wrist passed")
        
    except Exception as e:
        print(f"❌ pixels_agent_pos test failed: {e}")


def test_grid_position_with_cameras():
    """Test grid position control with camera observations."""
    print(f"\n--- Testing grid position control with cameras ---")
    
    try:
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_wrist')
        
        # Test specific grid positions
        for grid_pos in [0, 4, 8]:
            obs, info = env.reset(options={'cube_grid_position': grid_pos})
            
            print(f"Grid position {grid_pos}: agent_pos shape = {obs['agent_pos'].shape}")
            print(f"  Available cameras: {list(obs['pixels'].keys())}")
            
            # Verify observation structure
            assert 'pixels' in obs and 'agent_pos' in obs
            assert 'front_camera' in obs['pixels'] and 'wrist_camera' in obs['pixels']
        
        env.close()
        print("✅ Grid position control with cameras passed")
        
    except Exception as e:
        print(f"❌ Grid position test failed: {e}")


def test_environment_steps():
    """Test environment stepping with camera observations."""
    print(f"\n--- Testing environment steps ---")
    
    try:
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_wrist')
        obs, info = env.reset(seed=42)
        
        print(f"Initial observation structure verified")
        
        # Take a few steps
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step+1}: reward={reward:.3f}, cameras={list(obs['pixels'].keys())}")
            
            # Verify structure remains consistent
            assert 'pixels' in obs and 'agent_pos' in obs
            assert 'front_camera' in obs['pixels'] and 'wrist_camera' in obs['pixels']
        
        env.close()
        print("✅ Environment stepping test passed")
        
    except Exception as e:
        print(f"❌ Environment stepping test failed: {e}")


if __name__ == "__main__":
    print("🧪 Running camera configuration tests...")
    
    test_camera_configurations()
    test_grid_position_with_cameras()
    test_environment_steps()
    
    print(f"\n🎉 All tests completed!")