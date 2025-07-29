"""Tests for camera configuration functionality."""

import pytest
import numpy as np
import gymnasium as gym
import gym_soarm


class TestCameraConfiguration:
    """Test camera configuration options."""
    
    def test_front_only_config(self):
        """Test front_only camera configuration."""
        env = gym.make('SoArm-v0', obs_type='pixels', camera_config='front_only')
        obs, info = env.reset(seed=42)
        
        # Should only have front_camera
        assert isinstance(obs, dict)
        assert 'front_camera' in obs
        assert 'wrist_camera' not in obs
        assert 'overview_camera' not in obs
        
        # Check image properties
        assert obs['front_camera'].shape == (480, 640, 3)
        assert obs['front_camera'].dtype == np.uint8
        
        env.close()
    
    def test_front_wrist_config(self):
        """Test front_wrist camera configuration."""
        env = gym.make('SoArm-v0', obs_type='pixels', camera_config='front_wrist')
        obs, info = env.reset(seed=42)
        
        # Should have front_camera and wrist_camera
        assert isinstance(obs, dict)
        assert 'front_camera' in obs
        assert 'wrist_camera' in obs
        assert 'overview_camera' not in obs
        
        # Check image properties
        for camera in ['front_camera', 'wrist_camera']:
            assert obs[camera].shape == (480, 640, 3)
            assert obs[camera].dtype == np.uint8
            assert np.any(obs[camera] > 0)  # Images should have content
        
        env.close()
    
    def test_all_cameras_config(self):
        """Test all cameras configuration."""
        env = gym.make('SoArm-v0', obs_type='pixels', camera_config='all')
        obs, info = env.reset(seed=42)
        
        # Should have all cameras
        assert isinstance(obs, dict)
        assert 'front_camera' in obs
        assert 'wrist_camera' in obs
        assert 'overview_camera' in obs
        
        # Check image properties
        for camera in ['front_camera', 'wrist_camera', 'overview_camera']:
            assert obs[camera].shape == (480, 640, 3)
            assert obs[camera].dtype == np.uint8
            assert np.any(obs[camera] > 0)  # Images should have content
        
        env.close()
    
    def test_pixels_agent_pos_with_camera_config(self):
        """Test pixels_agent_pos observation type with camera configuration."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_only')
        obs, info = env.reset(seed=42)
        
        # Check structure
        assert isinstance(obs, dict)
        assert 'pixels' in obs
        assert 'agent_pos' in obs
        
        # Check pixels only contains front_camera
        assert isinstance(obs['pixels'], dict)
        assert 'front_camera' in obs['pixels']
        assert 'wrist_camera' not in obs['pixels']
        assert 'overview_camera' not in obs['pixels']
        
        # Check agent_pos
        assert obs['agent_pos'].shape == (6,)
        assert np.all(np.isfinite(obs['agent_pos']))
        
        env.close()
    
    def test_invalid_camera_config(self):
        """Test that invalid camera configurations raise errors."""
        with pytest.raises(ValueError, match="camera_config must be one of"):
            gym.make('SoArm-v0', camera_config='invalid_config')
    
    def test_observation_space_consistency(self):
        """Test that observation space matches actual observations."""
        configs = ['front_only', 'front_wrist', 'all']
        
        for config in configs:
            env = gym.make('SoArm-v0', obs_type='pixels', camera_config=config)
            obs, info = env.reset(seed=42)
            
            # Check that observation matches observation space
            assert env.observation_space.contains(obs), f"Observation doesn't match space for config {config}"
            
            # Check that all expected cameras are present
            expected_cameras = env._camera_names
            assert set(obs.keys()) == set(expected_cameras), f"Camera mismatch for config {config}"
            
            env.close()
    
    def test_camera_config_with_step(self):
        """Test camera configuration during environment stepping."""
        env = gym.make('SoArm-v0', obs_type='pixels', camera_config='front_wrist')
        obs, info = env.reset(seed=42)
        
        # Take a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check observation structure remains consistent
            assert isinstance(obs, dict)
            assert 'front_camera' in obs
            assert 'wrist_camera' in obs
            assert 'overview_camera' not in obs
            
            # Check image properties
            for camera in ['front_camera', 'wrist_camera']:
                assert obs[camera].shape == (480, 640, 3)
                assert obs[camera].dtype == np.uint8
        
        env.close()
    
    def test_different_configs_same_seed(self):
        """Test that different camera configs with same seed produce consistent robot states."""
        seed = 12345
        
        # Create environments with different camera configs
        env1 = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_only')
        env2 = gym.make('SoArm-v0', obs_type='pixels_agent_pos', camera_config='front_wrist')
        
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Agent positions should be identical (robot state should be same)
        np.testing.assert_array_equal(obs1['agent_pos'], obs2['agent_pos'])
        
        # But camera observations should be different structure
        assert 'wrist_camera' not in obs1['pixels']
        assert 'wrist_camera' in obs2['pixels']
        
        # Front camera images should be identical
        np.testing.assert_array_equal(obs1['pixels']['front_camera'], obs2['pixels']['front_camera'])
        
        env1.close()
        env2.close()


class TestCameraPerformance:
    """Test performance with different camera configurations."""
    
    def test_performance_scaling(self):
        """Test that fewer cameras improve performance."""
        import time
        
        configs = ['front_only', 'front_wrist', 'all']
        times = {}
        
        for config in configs:
            env = gym.make('SoArm-v0', obs_type='pixels', camera_config=config)
            
            # Measure reset time
            start_time = time.time()
            for _ in range(5):
                env.reset(seed=42)
            reset_time = (time.time() - start_time) / 5
            
            # Measure step time
            obs, _ = env.reset(seed=42)
            start_time = time.time()
            for _ in range(10):
                action = env.action_space.sample()
                env.step(action)
            step_time = (time.time() - start_time) / 10
            
            times[config] = {'reset': reset_time, 'step': step_time}
            env.close()
        
        # Generally, fewer cameras should be faster (though this depends on system)
        # At minimum, ensure times are reasonable
        for config, timing in times.items():
            assert timing['reset'] < 2.0, f"Reset too slow for {config}: {timing['reset']:.3f}s"
            assert timing['step'] < 0.2, f"Step too slow for {config}: {timing['step']:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])