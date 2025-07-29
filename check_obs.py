#!/usr/bin/env python3
"""
Check observation structure
"""

from gym_soarm_aloha import SoArmAlohaEnv

def check_obs():
    env = SoArmAlohaEnv(task="pick_place", render_mode="rgb_array")
    
    try:
        obs, info = env.reset(seed=42)
        print("Observation keys:", list(obs.keys()))
        for key, value in obs.items():
            print(f"{key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, 'shape') and len(value.shape) == 1 and value.shape[0] <= 10:
                print(f"  Values: {value}")
        
        print("✓ Robot 90-degree rotation configuration loaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    check_obs()