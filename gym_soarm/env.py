import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_soarm.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
    JOINT_LIMITS,
    get_joint_limits_array,
    clip_action_to_limits,
)
from gym_soarm.tasks.sim import PickPlaceTask, StackingTask


class SoArmAlohaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        task="pick_place",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        camera_config="front_wrist",  # "front_only", "front_wrist", "all"
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_config = camera_config
        
        # Validate camera configuration
        valid_configs = ["front_only", "front_wrist", "all"]
        if camera_config not in valid_configs:
            raise ValueError(f"camera_config must be one of {valid_configs}, got {camera_config}")
        
        # Define which cameras to include based on configuration
        self._camera_names = self._get_camera_names(camera_config)

        self._env = self._make_env_task(self.task)
        self._viewer = None
        self._viewer_thread = None
        self._current_camera = "overview_camera"  # Default camera for viewer
        self._available_cameras = ["overview_camera", "front_camera", "wrist_camera"]

        # Define observation space based on observation type
        if self.obs_type == "state":
            # State observation includes joint positions, velocities, and end-effector pose
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(JOINTS) * 2 + 7,),  # qpos + qvel + ee_pos + ee_quat
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            # Create observation space based on camera configuration
            camera_spaces = {}
            for camera_name in self._camera_names:
                camera_spaces[camera_name] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )
            self.observation_space = spaces.Dict(camera_spaces)
        elif self.obs_type == "pixels_agent_pos":
            # Create camera observation space based on configuration
            camera_spaces = {}
            for camera_name in self._camera_names:
                camera_spaces[camera_name] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )
            
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(camera_spaces),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        # Action space matches SO-ARM100 joint space (6DOF)
        min_limits, max_limits = get_joint_limits_array()
        self.action_space = spaces.Box(
            low=min_limits, 
            high=max_limits, 
            shape=(len(ACTIONS),), 
            dtype=np.float32
        )

    def _get_camera_names(self, camera_config):
        """Get list of camera names based on configuration."""
        if camera_config == "front_only":
            return ["front_camera"]
        elif camera_config == "front_wrist":
            return ["front_camera", "wrist_camera"]
        elif camera_config == "all":
            return ["overview_camera", "front_camera", "wrist_camera"]
        else:
            raise ValueError(f"Unknown camera_config: {camera_config}")

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array(visualize=True)
        elif self.render_mode == "human":
            return self._render_human()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_rgb_array(self, visualize=False):
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        
        # Use current selected camera for visualization
        image = self._env.physics.render(height=height, width=width, camera_id=self._current_camera)
        return image
    
    def _render_human(self):
        """Render in human mode using OpenCV GUI window"""
        try:
            import cv2
            import numpy as np
            
            if self._viewer is None:
                self._viewer = {
                    'window_name': 'SO-ARM101 Simulation',
                    'initialized': True
                }
                print("Launching OpenCV GUI viewer...")
                print("Camera Controls:")
                print("  1 - Overview Camera (top-down view)")
                print("  2 - Front Camera (angled view)") 
                print("  3 - Wrist Camera (end-effector view)")
                print("  q/ESC - Exit viewer")
                
                # Create window
                cv2.namedWindow(self._viewer['window_name'], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._viewer['window_name'], 800, 600)
            
            # Get current image from simulation
            image = self._render_rgb_array(visualize=True)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Add some text overlay with simulation info
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)  # White text
            
            # Add title and camera info
            cv2.putText(image_bgr, 'SO-ARM101 Simulation', (10, 30), 
                       font, 1, text_color, 2, cv2.LINE_AA)
            cv2.putText(image_bgr, f'Camera: {self._current_camera}', (10, 70), 
                       font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add controls instructions
            cv2.putText(image_bgr, 'Controls: 1=Overview 2=Front 3=Wrist Q=Exit', (10, image_bgr.shape[0] - 10), 
                       font, 0.5, text_color, 1, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow(self._viewer['window_name'], image_bgr)
            
            # Process key events (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Viewer window closed by user.")
                cv2.destroyWindow(self._viewer['window_name'])
                self._viewer = None
            elif key == ord('1'):  # Switch to overview camera
                self._current_camera = "overview_camera"
                print(f"Switched to {self._current_camera}")
            elif key == ord('2'):  # Switch to front camera
                self._current_camera = "front_camera"
                print(f"Switched to {self._current_camera}")
            elif key == ord('3'):  # Switch to wrist camera
                self._current_camera = "wrist_camera"
                print(f"Switched to {self._current_camera}")
            
            return image
            
        except ImportError:
            print("Warning: OpenCV not available for human rendering. Using rgb_array mode.")
            return self._render_rgb_array(visualize=True)
        except Exception as e:
            print(f"Warning: Could not display GUI viewer: {e}. Using rgb_array mode.")
            return self._render_rgb_array(visualize=True)

    def _make_env_task(self, task_name):
        # Time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        xml_path = ASSETS_DIR / "so_arm_main_new.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))

        if task_name == "pick_place":
            task = PickPlaceTask()
        elif task_name == "stacking":
            task = StackingTask()
        else:
            raise NotImplementedError(f"Task '{task_name}' not implemented")

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            # Combine joint positions, velocities, and end-effector pose
            qpos = raw_obs["qpos"]
            qvel = raw_obs["qvel"]
            
            # Get end-effector position and orientation from sensors
            ee_pos = raw_obs.get("gripper_pos_sensor", np.zeros(3))
            ee_quat = raw_obs.get("gripper_quat_sensor", np.array([1, 0, 0, 0]))
            
            obs = np.concatenate([qpos, qvel, ee_pos, ee_quat])
            
        elif self.obs_type == "pixels":
            # Only include cameras specified in configuration
            obs = {}
            for camera_name in self._camera_names:
                obs[camera_name] = raw_obs["images"][camera_name].copy()
            
        elif self.obs_type == "pixels_agent_pos":
            # Only include cameras specified in configuration
            camera_obs = {}
            for camera_name in self._camera_names:
                camera_obs[camera_name] = raw_obs["images"][camera_name].copy()
            
            obs = {
                "pixels": camera_obs,
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Handle cube grid position from options
        if options is not None and 'cube_grid_position' in options:
            cube_grid_position = options['cube_grid_position']
            # Validate cube_grid_position
            if cube_grid_position is not None and (cube_grid_position < 0 or cube_grid_position > 8):
                raise ValueError("cube_grid_position must be between 0 and 8 (inclusive), or None for random")
            
            # Set cube grid position in task if it's a PickPlaceTask
            if hasattr(self._env.task, 'set_cube_grid_position'):
                self._env.task.set_cube_grid_position(cube_grid_position)

        # Seed the environment task
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        raw_obs = self._env.reset()
        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        
        # Clip action to joint limits for safety
        clipped_action = clip_action_to_limits(action)
        
        _, reward, _, raw_obs = self._env.step(clipped_action)

        # Check if cube is grasped and lifted for termination condition
        terminated = False
        if hasattr(self._env.task, 'is_cube_grasped_and_lifted'):
            terminated = self._env.task.is_cube_grasped_and_lifted(self._env.physics)
        
        is_success = terminated

        info = {"is_success": is_success}
        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            try:
                if isinstance(self._viewer, dict) and 'render_dir' in self._viewer:
                    # Print summary for file-based human rendering
                    print(f"Human rendering complete. {self._viewer['frame_count']} frames saved to {self._viewer['render_dir']}")
                elif isinstance(self._viewer, dict) and 'window_name' in self._viewer:
                    # Close OpenCV window
                    import cv2
                    cv2.destroyWindow(self._viewer['window_name'])
                    cv2.waitKey(1)  # Process the destroy window event
                    print("OpenCV viewer window closed.")
                elif isinstance(self._viewer, dict) and 'fig' in self._viewer:
                    # Close matplotlib figure
                    import matplotlib.pyplot as plt
                    plt.close(self._viewer['fig'])
                elif self._viewer != "failed":
                    # Close other types of viewers
                    try:
                        if hasattr(self._viewer, 'close'):
                            self._viewer.close()
                        print("Viewer closed.")
                    except:
                        pass
            except:
                pass
            self._viewer = None
        
        # Clean up viewer thread
        if self._viewer_thread is not None:
            self._viewer_thread = None
            
        if hasattr(self, '_env'):
            del self._env