from pathlib import Path

### Simulation environment fixed constants
DT = 0.02  # 20ms timestep -> 50 Hz
FPS = 50

# SO-ARM101 joint configuration (6DOF single arm) - from actual MJCF
JOINTS = [
    "shoulder_pan",     # Base rotation
    "shoulder_lift",    # Shoulder joint
    "elbow_flex",       # Elbow joint
    "wrist_flex",       # Wrist flexion
    "wrist_roll",       # Wrist rotation
    "gripper",          # Gripper joint
]

# Action space matches joint space for direct control
ACTIONS = JOINTS.copy()

# SO-ARM101 joint limits (from actual MJCF file)
JOINT_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.17453, 1.74533),
}

# Starting pose for SO-ARM100 (in radians)
START_ARM_POSE = [
    0.0,     # shoulder_pan (centered)
    0.0,     # shoulder_lift (neutral)
    -1.57,   # elbow_flex (90 degrees bent)
    0.0,     # wrist_flex (neutral)
    0.0,     # wrist_roll (neutral)
    0.0,     # gripper (neutral position)
]

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"

# SO-ARM101 gripper limits (from actual MJCF)
GRIPPER_POSITION_OPEN = 1.74533
GRIPPER_POSITION_CLOSE = -0.17453

# Workspace boundaries for SO-ARM101 (in meters) - adjusted for rotated robot position
WORKSPACE_BOUNDS = {
    'x': [0.2, 0.6],    # Forward reach (robot rotated 90°, now reaches toward +X direction)
    'y': [-0.3, 0.3],   # Left-right reach (robot at y=0.15, can reach ±0.3 in Y direction)
    'z': [0.0, 0.4],    # Vertical reach
}

############################ Helper functions ############################

def normalize_gripper_position(x):
    """Normalize gripper position to [0, 1] where 0=closed, 1=open"""
    return (x - GRIPPER_POSITION_CLOSE) / (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE)

def unnormalize_gripper_position(x):
    """Convert normalized gripper position back to joint space"""
    return x * (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE) + GRIPPER_POSITION_CLOSE

def normalize_gripper_velocity(x):
    """Normalize gripper velocity"""
    return x / (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE)

def get_joint_limits_array():
    """Get joint limits as numpy-compatible arrays"""
    import numpy as np
    limits = np.array([JOINT_LIMITS[joint] for joint in JOINTS])
    return limits[:, 0], limits[:, 1]  # min_limits, max_limits

def clip_action_to_limits(action):
    """Clip action to joint limits"""
    import numpy as np
    min_limits, max_limits = get_joint_limits_array()
    return np.clip(action, min_limits, max_limits)

def is_in_workspace(position):
    """Check if position is within SO-ARM100 workspace"""
    x, y, z = position
    return (WORKSPACE_BOUNDS['x'][0] <= x <= WORKSPACE_BOUNDS['x'][1] and
            WORKSPACE_BOUNDS['y'][0] <= y <= WORKSPACE_BOUNDS['y'][1] and
            WORKSPACE_BOUNDS['z'][0] <= z <= WORKSPACE_BOUNDS['z'][1])

def sample_workspace_position(random_state=None):
    """Sample random position within workspace"""
    import numpy as np
    if random_state is None:
        random_state = np.random.RandomState()
    
    return [
        random_state.uniform(*WORKSPACE_BOUNDS['x']),
        random_state.uniform(*WORKSPACE_BOUNDS['y']),
        random_state.uniform(*WORKSPACE_BOUNDS['z']),
    ]