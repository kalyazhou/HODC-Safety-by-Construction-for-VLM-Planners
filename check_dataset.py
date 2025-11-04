#!/usr/bin/env python3
"""
快速检查nuScenes数据集质量
"""
import numpy as np
from nuscenes.nuscenes import NuScenes
import sys

def check_scene_quality(nusc, scene, threshold_k=10.0, threshold_v=50.0):
    """
    检查单个场景的数据质量
    
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
        # 收集场景数据
        sample_token = scene['first_sample_token']
        velocities = []
        curvatures = []
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # 获取ego pose
            ego_pose = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])
            
            # 计算速度（粗略估计）
            if len(velocities) > 0:
                # 简化：只检查位置数据是否合理
                pos = np.array(ego_pose['translation'][:2])
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    issues.append("NaN/Inf in position")
            
            velocities.append(np.linalg.norm(ego_pose['translation'][:2]))
            
            # 下一个sample
            sample_token = sample.get('next', '')
        
        # 统计信息
        stats['num_frames'] = len(velocities)
        
        # 检查曲率（需要从annotation获取，这里简化检查）
        # 实际上我们直接运行时检查observed curvatures
        
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
    print(f"✓ Loaded {len(nusc.scene)} scenes\n")
    
    print("="*80)
    print("DATASET QUALITY CHECK")
    print("="*80)
    
    valid_scenes = []
    invalid_scenes = []
    
    for scene_idx, scene in enumerate(nusc.scene):
        result = check_scene_quality(nusc, scene)
        
        status = "✓ OK" if result['is_valid'] else "✗ ISSUES"
        print(f"[{scene_idx+1:2d}/10] {result['name']:20s} {status:10s} frames={result['stats'].get('num_frames', 0):3d}", end="")
        
        if not result['is_valid']:
            print(f" ⚠️  {', '.join(result['issues'])}")
            invalid_scenes.append(result)
        else:
            print()
            valid_scenes.append(result)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Valid scenes:   {len(valid_scenes)}/10")
    print(f"✗ Invalid scenes: {len(invalid_scenes)}/10")
    
    if invalid_scenes:
        print("\n⚠️  INVALID SCENES:")
        for result in invalid_scenes:
            print(f"  - {result['name']}: {', '.join(result['issues'])}")
    
    # 特殊检查：scene-1100
    print("\n" + "="*80)
    print("SPECIAL CHECK: Known problematic scenes")
    print("="*80)
    
    problematic = ['scene-1100', 'scene-0061']
    for scene_name in problematic:
        found = any(s['name'] == scene_name for s in nusc.scene)
        if found:
            print(f"  {scene_name}: Present in dataset ⚠️")
        else:
            print(f"  {scene_name}: Not in dataset ✓")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if len(invalid_scenes) == 0:
        print("✓ All scenes passed basic checks")
        print("✓ You can proceed with full evaluation")
    else:
        print(f"⚠️  Found {len(invalid_scenes)} scenes with potential issues")
        print(f"✓ Recommend testing with {len(valid_scenes)} valid scenes first")
        print("\nTo exclude problematic scenes, add this filter to main.py:")
        print("```python")
        exclude_list = [s['name'] for s in invalid_scenes]
        print(f"if name in {exclude_list}:")
        print("    continue")
        print("```")
    
    print("\n💡 Note: This is a quick check. Deep issues (like extreme curvatures)")
    print("   in scene-1100 can only be detected during actual evaluation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

