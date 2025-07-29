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


def forward_kinematics_so_arm_101(joint_angles):
    """
    Calculate end-effector position using forward kinematics for SO-ARM101.
    
    Args:
        joint_angles: Array of 6 joint angles [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    
    Returns:
        numpy.array: 3D position of end-effector (gripperframe site)
    """
    # Extract joint angles (first 5 joints affect end-effector position)
    q1, q2, q3, q4, q5 = joint_angles[:5]  # gripper joint doesn't affect position
    
    # SO-ARM101 kinematic parameters from XML analysis
    # Base frame at robot base (considering 90° rotation: robot at y=0.15, rotated)
    base_offset = np.array([0, 0.15, 0])  # Robot base position
    
    # Link lengths and offsets from XML file analysis
    # Link 1 (base to shoulder): pos="0.0388353 -8.97657e-09 0.0624"
    L1_offset = np.array([0.0388353, 0, 0.0624])
    
    # Link 2 (shoulder to upper_arm): pos="-0.0303992 -0.0182778 -0.0542"
    L2_offset = np.array([-0.0303992, -0.0182778, -0.0542])
    L2_length = 0.11257  # Distance to elbow joint
    
    # Link 3 (upper_arm to lower_arm): pos="-0.11257 -0.028 1.73763e-16"
    L3_offset = np.array([-0.11257, -0.028, 0])
    L3_length = 0.1349  # Distance to wrist joint
    
    # Link 4 (lower_arm to wrist): pos="-0.1349 0.0052 3.62355e-17"
    L4_offset = np.array([-0.1349, 0.0052, 0])
    
    # Link 5 (wrist to gripper): pos="5.55112e-17 -0.0611 0.0181"
    L5_offset = np.array([0, -0.0611, 0.0181])
    
    # End-effector offset (gripperframe site): pos="-0.0079 -0.000218121 -0.0981274"
    EE_offset = np.array([-0.0079, -0.000218121, -0.0981274])
    
    # Create transformation matrices for each joint
    def rotation_z(angle):
        """Create rotation matrix around Z-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0, 0],
                        [s,  c, 0, 0],
                        [0,  0, 1, 0],
                        [0,  0, 0, 1]])
    
    def translation(x, y, z):
        """Create translation matrix"""
        return np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
    
    def rotation_y(angle):
        """Create rotation matrix around Y-axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c,  0, s, 0],
                        [0,  1, 0, 0],
                        [-s, 0, c, 0],
                        [0,  0, 0, 1]])
    
    # Base transformation (robot rotated 90° around Z)
    T_base = rotation_z(np.pi/2) @ translation(base_offset[0], base_offset[1], base_offset[2])
    
    # Joint 1: shoulder_pan (Z-axis rotation)
    T1 = translation(L1_offset[0], L1_offset[1], L1_offset[2]) @ rotation_z(q1)
    
    # Joint 2: shoulder_lift (Z-axis rotation, but coordinate frame is rotated)
    T2 = translation(L2_offset[0], L2_offset[1], L2_offset[2]) @ rotation_z(q2)
    
    # Joint 3: elbow_flex (Z-axis rotation)
    T3 = translation(L3_offset[0], L3_offset[1], L3_offset[2]) @ rotation_z(q3)
    
    # Joint 4: wrist_flex (Z-axis rotation)
    T4 = translation(L4_offset[0], L4_offset[1], L4_offset[2]) @ rotation_z(q4)
    
    # Joint 5: wrist_roll (Z-axis rotation)
    T5 = translation(L5_offset[0], L5_offset[1], L5_offset[2]) @ rotation_z(q5)
    
    # End-effector transformation
    T_ee = translation(EE_offset[0], EE_offset[1], EE_offset[2])
    
    # Compose all transformations
    T_total = T_base @ T1 @ T2 @ T3 @ T4 @ T5 @ T_ee
    
    # Extract position from transformation matrix
    end_effector_pos = T_total[:3, 3]
    
    return end_effector_pos

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
        """Calculate reward based on Euclidean distance between end-effector and blue cube."""
        # Method 1: Calculate end-effector position using forward kinematics
        joint_angles = physics.data.qpos[:6]  # First 6 joints including gripper
        ee_pos = forward_kinematics_so_arm_101(joint_angles)
        
        # Get blue cube position
        # Blue cube is at qpos[6:13] (first 6 are robot joints, next 7 are blue cube free joint)
        if physics.data.qpos.shape[0] < 13:
            print("Warning: blue_cube qpos not available")
            return 0.0
            
        cube_pos = physics.data.qpos[6:9]  # qpos[6:9] are blue cube position (x,y,z)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(ee_pos - cube_pos)
        
        # Optional debug information (commented out for production)
        # import time
        # if not hasattr(self, '_last_debug_time'):
        #     self._last_debug_time = 0
        # current_time = time.time()
        # if current_time - self._last_debug_time > 2.0:  # Print every 2 seconds
        #     print(f"Debug FK - Joint angles: {joint_angles}")
        #     print(f"Debug FK - EE pos: {ee_pos}, Cube pos: {cube_pos}, Distance: {distance:.4f}")
        #     self._last_debug_time = current_time
        
        # Distance-based reward with cutoff
        max_distance = 1.0  # Maximum distance for non-zero reward
        if distance > max_distance:
            return 0.0
        
        # Linear reward: closer distance gives higher reward
        # At distance=0: reward=1.0, at distance=max_distance: reward=0.0
        reward = 1.0 - (distance / max_distance)
        
        return max(0.0, reward)
    
    def is_cube_grasped_and_lifted(self, physics):
        """Check if the blue cube is grasped by gripper and lifted from table."""
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

        # Check if gripper is touching the blue cube
        gripper_touching_cube = any(
            ("blue_cube" in pair[0] and "gripper" in pair[1]) or 
            ("blue_cube" in pair[1] and "gripper" in pair[0])
            for pair in all_contact_pairs
        )
        
        # Check if cube is NOT on table (lifted)
        cube_on_table = any(
            ("blue_cube" in pair[0] and "table" in pair[1]) or 
            ("blue_cube" in pair[1] and "table" in pair[0])
            for pair in all_contact_pairs
        )
        
        # Check if gripper is closed (grasping)
        gripper_qpos = physics.data.qpos[5]  # 6th joint is gripper
        gripper_closed_threshold = 0.0  # Threshold for considering gripper as closed
        gripper_is_closed = gripper_qpos < gripper_closed_threshold
        
        # Cube is grasped and lifted if:
        # 1. Gripper is touching the cube
        # 2. Cube is not on table (lifted)
        # 3. Gripper is closed
        return gripper_touching_cube and not cube_on_table and gripper_is_closed


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