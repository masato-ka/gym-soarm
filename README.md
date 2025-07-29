# Gym SO-ARM

A gymnasium environment for SO-ARM101 single-arm manipulation based on gym-aloha, featuring multi-camera support and advanced simulation capabilities.

## Features

- **SO-ARM101 6DOF Robotic Arm**: Complete simulation of the SO-ARM101 robotic manipulator
- **Multi-Camera System**: Three camera views with runtime switching:
  - Overview camera: Top-down perspective
  - Front camera: Side view of the workspace
  - Wrist camera: First-person view from the robot's gripper
- **Interactive GUI Viewer**: OpenCV-based viewer with keyboard controls
- **Grid-Based Object Placement**: 3×3 grid system for randomized object positioning
- **Gymnasium Compatible**: Full OpenAI Gym/Gymnasium interface compliance
- **MuJoCo Physics**: High-fidelity physics simulation using dm-control

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/gym-soarm.git
cd gym-soarm

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev,test]"
```

### Using pip

```bash
pip install gym-soarm
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
import gym_soarm

# Create environment with human rendering
env = gym.make('SoArm-v0', render_mode='human')

# Reset environment
obs, info = env.reset()

# Run simulation
for _ in range(200):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Grid Position Control

You can specify the initial position of the blue cube using a 3×3 grid system:

```python
import gymnasium as gym
import gym_soarm

env = gym.make('SoArm-v0', render_mode='human')

# Place cube at specific grid position (0-8)
obs, info = env.reset(options={'cube_grid_position': 4})  # Center position

# Use random position (default behavior)
obs, info = env.reset(options={'cube_grid_position': None})
```

**Grid Layout (positions 0-8):**
```
0: (-10cm, -7.5cm)  1: (-10cm,  0cm)   2: (-10cm, +7.5cm)
3: ( 0cm,  -7.5cm)  4: ( 0cm,   0cm)   5: ( 0cm,  +7.5cm)  
6: (+10cm, -7.5cm)  7: (+10cm,  0cm)   8: (+10cm, +7.5cm)
```

The cube will be placed at the specified grid position with a random rotation (0°, 30°, 45°, or 60°).

### Camera Switching

During simulation with `render_mode='human'`, use these keyboard controls:

- **'1'**: Switch to overview camera
- **'2'**: Switch to front camera  
- **'3'**: Switch to wrist camera
- **'q'**: Quit simulation

## Environment Details

### Observation Space

The environment provides rich observations including:

- **Robot State**: Joint positions, velocities, and gripper state (42-dimensional)
- **Camera Images**: RGB images from active camera (480×640×3)
- **Object Information**: Positions and orientations of manipulated objects

```python
obs_space = gym.spaces.Dict({
    'robot_state': gym.spaces.Box(-np.inf, np.inf, shape=(42,)),
    'camera': gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
})
```

### Action Space

6DOF joint position control for the SO-ARM101:

- **Dimensions**: 6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Range**: Joint-specific limits based on hardware specifications
- **Control**: Direct joint position targets

### Workspace Configuration

- **Table Size**: 64cm × 45cm
- **Object Grid**: 3×3 positioning system with ±10cm(X), ±7.5cm(Y) spacing
- **Cube Size**: 3cm × 3cm × 3cm blue cubes
- **Robot Base**: Positioned at (0, 0.15, 0) with 90° rotation

### Camera Specifications

| Camera | Position | Orientation | FOV | Description |
|--------|----------|-------------|-----|-------------|
| Overview | (0, 0.4, 0.8) | Top-down | 90° | Bird's eye view |
| Front | (0, 0.7, 0.25) | Angled forward | 120° | Side perspective |
| Wrist | (0, -0.04, 0) | 30° X-rotation | 110° | First-person view |

## Development

### Project Structure

```
gym-soarm/
├── gym_soarm_aloha/           # Main package
│   ├── __init__.py           # Package initialization
│   ├── env.py               # Main environment class
│   ├── constants.py         # Environment constants
│   ├── assets/             # Robot models and scenes
│   │   ├── so101_new_calib.xml
│   │   ├── so_arm_main_new.xml
│   │   └── assets/         # STL mesh files
│   └── tasks/              # Task implementations
│       ├── __init__.py
│       └── sim.py          # Manipulation task
├── example.py              # Usage examples
├── setup.py               # Package setup
├── pyproject.toml         # Poetry configuration
└── README.md              # This file
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run basic functionality test
python example.py
```

### Code Style

The project uses Ruff for linting and formatting:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check gym_soarm_aloha/

# Auto-format code
ruff format gym_soarm_aloha/
```

## Hardware Requirements

- **Python**: ≥3.10
- **OpenGL**: Required for rendering
- **Memory**: ≥4GB RAM recommended
- **Storage**: ~500MB for assets and dependencies

## Troubleshooting

### Common Issues

1. **MuJoCo Installation**: Ensure MuJoCo ≥2.3.7 is properly installed
2. **OpenGL Context**: On headless systems, use `xvfb-run` for rendering
3. **Asset Loading**: Verify all `.stl` files are present in `assets/assets/`

### Platform-Specific Notes

- **macOS**: May require XQuartz for OpenGL support
- **Linux**: Ensure proper GPU drivers for hardware acceleration
- **Windows**: Use WSL2 for best compatibility

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{gym_soarm,
  title={Gym SO-ARM: A Gymnasium Environment for SO-ARM101 Manipulation},
  author={SO-ARM Development Team},
  version={0.1.0},
  year={2024},
  url={https://github.com/your-org/gym-soarm}
}
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## Support

For questions and support:
- GitHub Issues: [Report bugs or request features](https://github.com/your-org/gym-soarm/issues)
- Discussions: [Community discussions](https://github.com/your-org/gym-soarm/discussions)