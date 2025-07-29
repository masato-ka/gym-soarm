import collections
import numpy as np
from dm_control.suite import base

from gym_soarm.constants import (
    START_ARM_POSE,
    JOINTS,
    normalize_gripper_position,
    normalize_gripper_velocity,
    unnormalize_gripper_position,
    sample_workspace_position,
    is_in_workspace,
)

"""
Environment for simulated SO-ARM100 single-arm manipulation, with joint position control

Action space:      [arm_qpos (6)]                    # absolute joint position including gripper

Observation space: {"qpos": arm_qpos (6),            # absolute joint position including gripper
                    "qvel": arm_qvel (6),            # absolute joint velocity including gripper
                    "env_state": object_poses,       # positions of objects in environment
                    "images": {"main_camera": (480x640x3)}}  # h, w, c, dtype='uint8'
"""


class SoArmTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1.0

    def before_step(self, action, physics):
        # For SO-ARM101, action is directly the joint positions
        # Set controls for the 6 actuators directly
        physics.data.ctrl[:6] = action[:6]
        
        super().before_step(action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Reset robot to starting pose
        with physics.reset_context():
            # Set joint positions for the first 6 joints (SO-ARM101)
            physics.data.qpos[:6] = START_ARM_POSE
            physics.data.ctrl[:6] = START_ARM_POSE
        
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        # For single arm, return all 6 joint positions (first 6 elements)
        qpos_raw = physics.data.qpos[:6].copy()
        # Normalize gripper position (last element)
        arm_qpos = qpos_raw[:5]
        gripper_qpos = normalize_gripper_position(qpos_raw[5])
        return np.concatenate([arm_qpos, [gripper_qpos]])

    @staticmethod
    def get_qvel(physics):
        # For single arm, return all 6 joint velocities (first 6 elements)
        qvel_raw = physics.data.qvel[:6].copy()
        # Normalize gripper velocity (last element)
        arm_qvel = qvel_raw[:5]
        gripper_qvel = normalize_gripper_velocity(qvel_raw[5])
        return np.concatenate([arm_qvel, [gripper_qvel]])

    @staticmethod
    def get_env_state(physics):
        # Return positions of objects in environment
        # red_cube is at qpos[6:13], blue_cube is at qpos[13:20]
        # Each free joint has 7 DOF (3 pos + 4 quat)
        env_state = []
        if physics.data.qpos.shape[0] >= 13:
            # Red cube position (first 3 of 7 DOF)
            env_state.extend(physics.data.qpos[6:9])
        if physics.data.qpos.shape[0] >= 20:
            # Blue cube position (first 3 of 7 DOF)  
            env_state.extend(physics.data.qpos[13:16])
        return np.array(env_state)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics) 
        obs["env_state"] = self.get_env_state(physics)
        
        # Get end-effector pose from sensors if available
        if "gripper_pos_sensor" in physics.named.data.sensordata:
            obs["gripper_pos_sensor"] = physics.named.data.sensordata["gripper_pos_sensor"]
        if "gripper_quat_sensor" in physics.named.data.sensordata:
            obs["gripper_quat_sensor"] = physics.named.data.sensordata["gripper_quat_sensor"]
        
        obs["images"] = {}
        obs["images"]["front_camera"] = physics.render(height=480, width=640, camera_id="front_camera")
        obs["images"]["wrist_camera"] = physics.render(height=480, width=640, camera_id="wrist_camera")
        obs["images"]["overview_camera"] = physics.render(height=480, width=640, camera_id="overview_camera")

        return obs

    def get_reward(self, physics):
        # Base implementation - should be overridden by specific tasks
        return 0.0


class PickPlaceTask(SoArmTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1.0
        self.target_position = None
        self.object_picked = False
        self.cube_grid_position = None  # Store specified grid position (0-8)
    
    def set_cube_grid_position(self, position):
        """Set blue cube grid position (0-8). None for random position."""
        if position is not None and (position < 0 or position > 8):
            raise ValueError("Grid position must be between 0 and 8 (inclusive), or None for random")
        self.cube_grid_position = position
        
    def initialize_episode(self, physics):
        """Initialize pick and place episode with blue cube at specified or random grid position."""
        super().initialize_episode(physics)
        
        # Reset state tracking
        self.object_picked = False
        
        with physics.reset_context():
            # Use current time-based seed for true randomization
            import time
            current_seed = int(time.time() * 1000000) % 2**32
            random_state = np.random.RandomState(current_seed)
            
            # Define 9 specific positions around table center (0, 0.4) with ±10cm(X), ±7.5cm(Y) spacing
            # Grid layout:
            # 0: (-10, -7.5)  1: (-10,  0)   2: (-10, +7.5)
            # 3: ( 0,  -7.5)  4: ( 0,   0)   5: ( 0,  +7.5)
            # 6: (+10, -7.5)  7: (+10,  0)   8: (+10, +7.5)
            table_center = [0.0, 0.4]
            grid_positions = []
            for dx in [-0.10, 0.0, 0.10]:  # ±10cm in X direction
                for dy in [-0.075, 0.0, 0.075]:  # ±7.5cm in Y direction
                    grid_positions.append([table_center[0] + dx, table_center[1] + dy])
            
            # Use specified position or select random position
            if self.cube_grid_position is not None:
                selected_position = self.cube_grid_position
            else:
                selected_position = random_state.choice(len(grid_positions))
            
            blue_cube_x, blue_cube_y = grid_positions[selected_position]
            blue_cube_z = 0.05  # Place on table surface
            
            # Define 4 rotation angles: 0°, 30°, 45°, 60°
            rotation_angles = [0, 30, 45, 60]  # degrees
            selected_angle = random_state.choice(rotation_angles)
            
            # Convert angle to quaternion (rotation around Z-axis)
            angle_rad = np.radians(selected_angle)
            cos_half = np.cos(angle_rad / 2)
            sin_half = np.sin(angle_rad / 2)
            rotation_quat = [cos_half, 0, 0, sin_half]  # [w, x, y, z]
            
            position_mode = "specified" if self.cube_grid_position is not None else "random"
            print(f"Blue cube placed at {position_mode} position {selected_position}/8: ({blue_cube_x:.3f}, {blue_cube_y:.3f}, {blue_cube_z:.3f})")
            print(f"Blue cube rotation: {selected_angle}° around Z-axis")
            
            # Set blue cube position and orientation
            physics.named.data.qpos['blue_cube'][:3] = [blue_cube_x, blue_cube_y, blue_cube_z]
            physics.named.data.qpos['blue_cube'][3:7] = rotation_quat  # Apply rotation
            
            # Store blue cube position for reward calculation
            self.blue_cube_position = [blue_cube_x, blue_cube_y, blue_cube_z]

    def get_reward(self, physics):
        """Calculate reward based on pick and place progress."""
        # Get all contact pairs
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            if name_geom_1 and name_geom_2:
                contact_pair = (name_geom_1, name_geom_2)
                all_contact_pairs.append(contact_pair)

        # Check if gripper is touching the cube
        gripper_touching_cube = any(
            ("red_cube" in pair[0] and "gripper" in pair[1]) or 
            ("red_cube" in pair[1] and "gripper" in pair[0])
            for pair in all_contact_pairs
        )
        
        # Check if cube is on table
        cube_on_table = any(
            ("red_cube" in pair[0] and "table" in pair[1]) or 
            ("red_cube" in pair[1] and "table" in pair[0])
            for pair in all_contact_pairs
        )
        
        reward = 0.0
        
        # Reward progression: touch -> pick -> place
        if gripper_touching_cube:
            reward = 0.3
            if not cube_on_table:  # Cube lifted
                reward = 0.6
                self.object_picked = True
                
                # Check if cube is near target position
                if "red_cube" in physics.named.data.qpos:
                    cube_pos = physics.named.data.qpos["red_cube"][:3]
                    distance_to_target = np.linalg.norm(cube_pos[:2] - np.array(self.target_position[:2]))
                    
                    if distance_to_target < 0.05:  # Close to target
                        reward = 1.0  # Task completed
        
        return reward


class StackingTask(SoArmTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1.0
        
    def initialize_episode(self, physics):
        """Initialize stacking episode with two cubes."""
        super().initialize_episode(physics)
        
        with physics.reset_context():
            # Sample random positions for two cubes within workspace
            if self.random is not None:
                random_state = self.random
            else:
                random_state = np.random.RandomState()
            
            # Place first cube
            cube1_pos = sample_workspace_position(random_state)
            cube1_pos[2] = 0.05  # On table surface
            
            # Place second cube at different position
            cube2_pos = sample_workspace_position(random_state)
            cube2_pos[2] = 0.05  # On table surface
            
            # Ensure cubes are separated
            while np.linalg.norm(np.array(cube1_pos[:2]) - np.array(cube2_pos[:2])) < 0.08:
                cube2_pos = sample_workspace_position(random_state)
                cube2_pos[2] = 0.05
            
            # Set cube positions if they exist in the XML
            if "red_cube" in physics.named.data.qpos:
                physics.named.data.qpos["red_cube"][:3] = cube1_pos
                physics.named.data.qpos["red_cube"][3:] = [1, 0, 0, 0]
                
            if "blue_cube" in physics.named.data.qpos:
                physics.named.data.qpos["blue_cube"][:3] = cube2_pos
                physics.named.data.qpos["blue_cube"][3:] = [1, 0, 0, 0]

    def get_reward(self, physics):
        """Calculate reward based on stacking progress."""
        # Check if cubes are stacked (one on top of the other)
        if "red_cube" in physics.named.data.qpos and "blue_cube" in physics.named.data.qpos:
            red_pos = physics.named.data.qpos["red_cube"][:3]
            blue_pos = physics.named.data.qpos["blue_cube"][:3]
            
            # Check horizontal alignment (cubes should be close in x,y)
            horizontal_distance = np.linalg.norm(red_pos[:2] - blue_pos[:2])
            
            # Check vertical separation (one should be above the other)
            vertical_separation = abs(red_pos[2] - blue_pos[2])
            
            # Reward for horizontal alignment
            reward = 0.0
            if horizontal_distance < 0.05:  # Cubes aligned horizontally
                reward = 0.5
                
                if 0.04 < vertical_separation < 0.06:  # One cube on top of the other
                    reward = 1.0  # Task completed
                    
            return reward
        
        return 0.0