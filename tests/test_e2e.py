"""End-to-End tests for gym-soarm environment."""

import pytest
import numpy as np
import gymnasium as gym
import gym_soarm


class TestE2EEnvironment:
    """End-to-End tests for the complete environment functionality."""
    
    def test_environment_creation(self):
        """Test basic environment creation and registration."""
        env = gym.make('SoArm-v0')
        assert env is not None
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        env.close()
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        obs, info = env.reset(seed=42)
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert 'agent_pos' in obs
        assert 'pixels' in obs
        assert isinstance(info, dict)
        
        # Check agent position
        assert obs['agent_pos'].shape == (6,)  # 6 DOF
        assert np.all(np.isfinite(obs['agent_pos']))
        
        env.close()
    
    def test_environment_step(self):
        """Test environment step functionality."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        obs, info = env.reset(seed=42)
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check return values
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check observation consistency
        assert 'agent_pos' in obs
        assert 'pixels' in obs
        assert obs['agent_pos'].shape == (6,)
        
        env.close()
    
    def test_action_space_bounds(self):
        """Test that actions are properly bounded."""
        env = gym.make('SoArm-v0')
        action_space = env.action_space
        
        # Test action space properties
        assert hasattr(action_space, 'low')
        assert hasattr(action_space, 'high')
        assert hasattr(action_space, 'shape')
        assert action_space.shape == (6,)  # 6 DOF
        
        # Test that sampled actions are within bounds
        for _ in range(10):
            action = action_space.sample()
            assert np.all(action >= action_space.low)
            assert np.all(action <= action_space.high)
        
        env.close()
    
    def test_grid_position_control(self):
        """Test grid position control functionality."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        
        # Test all grid positions
        for grid_pos in range(9):
            obs, info = env.reset(options={'cube_grid_position': grid_pos})
            assert isinstance(obs, dict)
            assert 'agent_pos' in obs
            assert 'pixels' in obs
        
        # Test random position
        obs, info = env.reset(options={'cube_grid_position': None})
        assert isinstance(obs, dict)
        
        # Test invalid position (should raise error)
        with pytest.raises(ValueError):
            env.reset(options={'cube_grid_position': 10})
        
        with pytest.raises(ValueError):
            env.reset(options={'cube_grid_position': -1})
        
        env.close()
    
    def test_camera_observations(self):
        """Test camera observation functionality."""
        env = gym.make('SoArm-v0', obs_type='pixels')
        obs, info = env.reset(seed=42)
        
        # Check that we have camera observations
        assert isinstance(obs, dict)
        expected_cameras = ['front_camera', 'wrist_camera', 'overview_camera']
        for camera in expected_cameras:
            assert camera in obs
            assert obs[camera].shape == (480, 640, 3)
            assert obs[camera].dtype == np.uint8
            # Check that image has some content (not all zeros)
            assert np.any(obs[camera] > 0)
        
        env.close()
    
    def test_different_observation_types(self):
        """Test different observation type configurations."""
        # Test state observation
        env_state = gym.make('SoArm-v0', obs_type='state')
        obs_state, _ = env_state.reset(seed=42)
        assert isinstance(obs_state, np.ndarray)
        assert obs_state.shape[0] > 6  # Should include joint positions + end-effector info
        env_state.close()
        
        # Test pixels observation
        env_pixels = gym.make('SoArm-v0', obs_type='pixels')
        obs_pixels, _ = env_pixels.reset(seed=42)
        assert isinstance(obs_pixels, dict)
        assert 'front_camera' in obs_pixels
        assert 'wrist_camera' in obs_pixels
        assert 'overview_camera' in obs_pixels
        env_pixels.close()
        
        # Test combined observation
        env_combined = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        obs_combined, _ = env_combined.reset(seed=42)
        assert isinstance(obs_combined, dict)
        assert 'agent_pos' in obs_combined
        assert 'pixels' in obs_combined
        env_combined.close()
    
    def test_multiple_episodes(self):
        """Test multiple episode execution."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        
        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)
            assert isinstance(obs, dict)
            
            # Run a few steps
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
        
        env.close()
    
    def test_render_modes(self):
        """Test different render modes."""
        # Test rgb_array mode
        env_rgb = gym.make('SoArm-v0', render_mode='rgb_array')
        obs, info = env_rgb.reset(seed=42)
        
        # Get rendered image
        rendered_image = env_rgb.render()
        assert isinstance(rendered_image, np.ndarray)
        assert rendered_image.shape == (480, 640, 3)  # H, W, C
        assert rendered_image.dtype == np.uint8
        
        env_rgb.close()
        
        # Test human mode (should not crash)
        env_human = gym.make('SoArm-v0', render_mode='human')
        obs, info = env_human.reset(seed=42)
        
        # Render should not crash
        try:
            env_human.render()
        except ImportError:
            # OpenCV might not be available in test environment
            pass
        
        env_human.close()
    
    def test_reproducibility(self):
        """Test that environment is reproducible with same seed."""
        env1 = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        env2 = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        
        # Reset both environments with same seed
        obs1, _ = env1.reset(seed=12345)
        obs2, _ = env2.reset(seed=12345)
        
        # Agent positions should be identical
        np.testing.assert_array_equal(obs1['agent_pos'], obs2['agent_pos'])
        
        # Take same actions and compare
        action = env1.action_space.sample()
        obs1, reward1, term1, trunc1, info1 = env1.step(action)
        obs2, reward2, term2, trunc2, info2 = env2.step(action)
        
        # Results should be identical
        np.testing.assert_array_equal(obs1['agent_pos'], obs2['agent_pos'])
        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2
        
        env1.close()
        env2.close()
    
    def test_task_switching(self):
        """Test different task configurations."""
        # Test pick_place task
        env_pick = gym.make('SoArm-v0', task='pick_place', obs_type='pixels_agent_pos')
        obs, info = env_pick.reset(seed=42)
        assert isinstance(obs, dict)
        env_pick.close()
        
        # Test stacking task
        env_stack = gym.make('SoArm-v0', task='stacking', obs_type='pixels_agent_pos')
        obs, info = env_stack.reset(seed=42)
        assert isinstance(obs, dict)
        env_stack.close()


class TestPerformance:
    """Performance tests to ensure environment runs efficiently."""
    
    def test_reset_performance(self):
        """Test that reset is reasonably fast."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        
        import time
        start_time = time.time()
        
        for _ in range(10):
            env.reset(seed=42)
        
        elapsed = time.time() - start_time
        avg_reset_time = elapsed / 10
        
        # Reset should take less than 1 second on average
        assert avg_reset_time < 1.0, f"Reset too slow: {avg_reset_time:.3f}s average"
        
        env.close()
    
    def test_step_performance(self):
        """Test that steps are reasonably fast."""
        env = gym.make('SoArm-v0', obs_type='pixels_agent_pos')
        env.reset(seed=42)
        
        import time
        start_time = time.time()
        
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)
        
        elapsed = time.time() - start_time
        avg_step_time = elapsed / 50
        
        # Steps should take less than 0.1 seconds on average
        assert avg_step_time < 0.1, f"Steps too slow: {avg_step_time:.3f}s average"
        
        env.close()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])