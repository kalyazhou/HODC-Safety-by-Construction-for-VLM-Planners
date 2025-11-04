#!/usr/bin/env python3
"""
Quick quality check for the nuScenes dataset
"""
import numpy as np
from nuscenes.nuscenes import NuScenes
import sys

def check_scene_quality(nusc, scene, threshold_k=10.0, threshold_v=50.0):
    """
    Check the data quality of a single scene.

    Returns:
        dict: {
            'name': str,
            'is_valid': bool,
            'issues': list[str],
            'stats': dict
        }
    """
    name = scene['name']
    issues = []
    stats = {}

    try:
        # Collect scene data
        sample_token = scene['first_sample_token']
        velocities = []
        curvatures = []

        while sample_token:
            sample = nusc.get('sample', sample_token)

            # Get ego pose
            ego_pose = nusc.get(
                'ego_pose',
                nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']
            )

            # Rough velocity check (sanity on position only)
            if len(velocities) > 0:
                pos = np.array(ego_pose['translation'][:2])
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    issues.append("NaN/Inf in position")

            velocities.append(np.linalg.norm(ego_pose['translation'][:2]))

            # Next sample
            sample_token = sample.get('next', '')

        # Stats
        stats['num_frames'] = len(velocities)

        # Curvature checks would normally use annotations; simplified here

        if len(velocities) < 10:
            issues.append(f"Too few frames: {len(velocities)}")

        if stats['num_frames'] == 0:
            issues.append("No frames found")

    except Exception as e:
        issues.append(f"Exception: {str(e)}")

    return {
        'name': name,
        'is_valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }


def main():
    dataroot = "C:/Users/79120/OpenEMMA/openemma"
    version = "v1.0-mini"

    print(f"Loading NuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    total = len(nusc.scene)
    print(f"Loaded {total} scenes\n")

    print("=" * 80)
    print("DATASET QUALITY CHECK")
    print("=" * 80)

    valid_scenes = []
    invalid_scenes = []

    for scene_idx, scene in enumerate(nusc.scene):
        result = check_scene_quality(nusc, scene)

        status = "OK" if result['is_valid'] else "ISSUES"
        print(
            f"[{scene_idx+1:2d}/{total}] {result['name']:20s} {status:10s} "
            f"frames={result['stats'].get('num_frames', 0):3d}",
            end=""
        )

        if not result['is_valid']:
            print(f"  WARN  {', '.join(result['issues'])}")
            invalid_scenes.append(result)
        else:
            print()
            valid_scenes.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Valid scenes:   {len(valid_scenes)}/{total}")
    print(f"Invalid scenes: {len(invalid_scenes)}/{total}")

    if invalid_scenes:
        print("\nINVALID SCENES:")
        for result in invalid_scenes:
            print(f"  - {result['name']}: {', '.join(result['issues'])}")

    # Special check
    print("\n" + "=" * 80)
    print("SPECIAL CHECK: Known problematic scenes")
    print("=" * 80)

    problematic = ['scene-1100', 'scene-0061']
    for scene_name in problematic:
        found = any(s['name'] == scene_name for s in nusc.scene)
        if found:
            print(f"  {scene_name}: Present in dataset (warning)")
        else:
            print(f"  {scene_name}: Not in dataset (ok)")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if len(invalid_scenes) == 0:
        print("All scenes passed basic checks.")
        print("You can proceed with full evaluation.")
    else:
        print(f"Found {len(invalid_scenes)} scenes with potential issues.")
        print(f"Recommend testing with {len(valid_scenes)} valid scenes first.")
        print("\nTo exclude problematic scenes, add this filter to main.py:")
        print("```python")
        exclude_list = [s['name'] for s in invalid_scenes]
        print(f"if name in {exclude_list}:")
        print("    continue")
        print("```")

    print("\nNote: This is a quick check. Deep issues (like extreme curvatures)")
    print("in scene-1100 can only be detected during actual evaluation.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
