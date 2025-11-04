"""
智能采样模块 - 从nuScenes Full中采样200个代表性场景
用于扩展实验：从10场景(mini) → 200场景(sampled full)
"""

import os
import random
import numpy as np
from nuscenes import NuScenes
from collections import defaultdict


def check_scene_has_images(nusc, scene):
    """检查场景的图片文件是否存在"""
    try:
        first_sample = nusc.get('sample', scene['first_sample_token'])
        cam_front_token = first_sample['data']['CAM_FRONT']
        cam_front_data = nusc.get('sample_data', cam_front_token)
        image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
        return os.path.exists(image_path)
    except:
        return False


def analyze_scene_features(nusc, scene):
    """分析场景特征（用于分类）"""
    scene_token = scene['token']
    first_sample = nusc.get('sample', scene['first_sample_token'])
    
    # 1. 时间特征（白天/夜间）
    timestamp = first_sample['timestamp']
    hour = (timestamp // 1000000) % 86400 // 3600  # 简化版本
    is_night = (hour < 6 or hour > 20)
    
    # 2. 位置特征
    location = nusc.get('log', scene['log_token'])['location']
    
    # 3. 场景描述（包含复杂度信息）
    description = scene['description'].lower()
    
    # 4. 估计复杂度（基于关键词）
    complexity = 'simple'
    complex_keywords = ['turn', 'intersection', 'construction', 'crowded', 'heavy', 'stop']
    medium_keywords = ['lane', 'merge', 'traffic']
    
    if any(kw in description for kw in complex_keywords):
        complexity = 'complex'
    elif any(kw in description for kw in medium_keywords):
        complexity = 'medium'
    
    # 5. 场景长度（帧数）
    frame_count = scene['nbr_samples']
    
    return {
        'name': scene['name'],
        'token': scene['token'],
        'location': location,
        'is_night': is_night,
        'complexity': complexity,
        'frame_count': frame_count,
        'description': scene['description']
    }


def smart_sample_scenes(nusc, n_samples=200, seed=42):
    """
    智能采样nuScenes场景
    
    ⚠️ 重要说明：
    1. 只从 trainval (850场景) 采样，test集标注未公开无法使用
    2. 本项目是**纯推理（零训练）**，不训练模型参数
    3. 因此不存在过拟合问题，trainval全部850场景都是有效评估数据
    4. 不需要区分train/val（这是训练场景才需要的）
    
    nuScenes数据集结构：
    - trainval: 850 scenes (700 train + 150 val，标注公开) ✅ 可用
    - test: 150 scenes (标注未公开，仅用于在线排行榜) ❌ 不可用
    - mini: 10 scenes (trainval的子集，已完成实验)
    
    采样目标分布（200个场景 = trainval的23.5%）：
    - 复杂度: 简单35%, 中等47%, 复杂18%
    - 时间: 白天70%, 夜间30%
    - 位置: Boston 50%, Singapore 50%
    - 避免已知异常场景（极端曲率、静止车辆等）
    
    Args:
        nusc: NuScenes对象 (version='v1.0-trainval')
        n_samples: 采样数量（默认200）
        seed: 随机种子（用于复现，默认42）
    
    Returns:
        list: 采样的场景列表（nuScenes scene对象）
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 已知异常场景（需要过滤）
    problematic_scenes = ['scene-0553', 'scene-0757', 'scene-1100']
    
    # 分析所有场景（同时检查图片是否存在）
    print(f"📊 分析nuScenes场景特征...")
    all_scenes = []
    scenes_without_images = 0
    
    for scene in nusc.scene:
        if scene['name'] in problematic_scenes:
            continue
        
        # 检查图片是否存在（重要：用户可能只下载了部分数据）
        if not check_scene_has_images(nusc, scene):
            scenes_without_images += 1
            continue
        
        features = analyze_scene_features(nusc, scene)
        all_scenes.append(features)
    
    print(f"✅ 共有 {len(all_scenes)} 个有效场景（有图片数据）")
    if scenes_without_images > 0:
        print(f"⚠️  跳过 {scenes_without_images} 个场景（图片文件不存在）")
    
    # 按特征分组
    grouped = {
        'simple_day_boston': [],
        'simple_day_singapore': [],
        'simple_night_boston': [],
        'simple_night_singapore': [],
        'medium_day_boston': [],
        'medium_day_singapore': [],
        'medium_night_boston': [],
        'medium_night_singapore': [],
        'complex_day_boston': [],
        'complex_day_singapore': [],
        'complex_night_boston': [],
        'complex_night_singapore': [],
    }
    
    for scene_info in all_scenes:
        complexity = scene_info['complexity']
        time_of_day = 'night' if scene_info['is_night'] else 'day'
        location = 'boston' if 'boston' in scene_info['location'].lower() else 'singapore'
        
        key = f"{complexity}_{time_of_day}_{location}"
        if key in grouped:
            grouped[key].append(scene_info)
    
    # 打印分组统计
    print("\n📈 场景分布统计:")
    for key, scenes in grouped.items():
        if scenes:
            print(f"  {key}: {len(scenes)} 个场景")
    
    # 目标分配（根据n_samples动态调整，保持比例一致）
    # 基础比例（200场景版本）
    base_distribution = {
        'simple_day_boston': 25/200,
        'simple_day_singapore': 25/200,
        'simple_night_boston': 10/200,
        'simple_night_singapore': 10/200,
        'medium_day_boston': 33/200,
        'medium_day_singapore': 33/200,
        'medium_night_boston': 14/200,
        'medium_night_singapore': 14/200,
        'complex_day_boston': 13/200,
        'complex_day_singapore': 13/200,
        'complex_night_boston': 5/200,
        'complex_night_singapore': 5/200,
    }
    
    # 根据实际n_samples缩放
    target_distribution = {
        key: max(1, int(ratio * n_samples))  # 至少1个
        for key, ratio in base_distribution.items()
    }
    
    # 微调确保总数匹配（由于四舍五入可能有差异）
    total_allocated = sum(target_distribution.values())
    if total_allocated < n_samples:
        # 补充到medium_day_boston（最大类别）
        target_distribution['medium_day_boston'] += (n_samples - total_allocated)
    elif total_allocated > n_samples:
        # 从medium_day_boston减少
        target_distribution['medium_day_boston'] -= (total_allocated - n_samples)
    
    # 执行分层采样
    sampled_scenes = []
    remaining_pool = []  # 未被采样的场景池
    
    for key, target_count in target_distribution.items():
        available = grouped[key]
        
        if len(available) == 0:
            print(f"⚠️  {key}: 没有可用场景，跳过")
            continue
        
        # 如果可用场景少于目标，全部采样
        if len(available) <= target_count:
            sampled = available
            print(f"✅ {key}: 采样 {len(sampled)}/{target_count} (全部可用)")
        else:
            # 随机采样
            sampled = random.sample(available, target_count)
            print(f"✅ {key}: 采样 {len(sampled)}/{target_count}")
            # 未被采样的场景加入剩余池
            remaining = [s for s in available if s not in sampled]
            remaining_pool.extend(remaining)
        
        sampled_scenes.extend(sampled)
    
    # 如果采样数不足，从剩余池补充
    shortage = n_samples - len(sampled_scenes)
    if shortage > 0 and remaining_pool:
        extra_samples = min(shortage, len(remaining_pool))
        extras = random.sample(remaining_pool, extra_samples)
        sampled_scenes.extend(extras)
        print(f"\n➕ 从剩余池补充 {extra_samples} 个场景")
    
    print(f"\n🎯 总共采样 {len(sampled_scenes)} 个场景")
    
    # 转换回nuScenes scene对象
    sampled_tokens = {s['token'] for s in sampled_scenes}
    result_scenes = [s for s in nusc.scene if s['token'] in sampled_tokens]
    
    # 保存采样结果（用于复现）
    sampling_log = {
        'seed': seed,
        'n_samples': len(sampled_scenes),
        'target_distribution': target_distribution,
        'sampled_scene_names': [s['name'] for s in sampled_scenes],
        'complexity_stats': {
            'simple': sum(1 for s in sampled_scenes if s['complexity'] == 'simple'),
            'medium': sum(1 for s in sampled_scenes if s['complexity'] == 'medium'),
            'complex': sum(1 for s in sampled_scenes if s['complexity'] == 'complex'),
        },
        'time_stats': {
            'day': sum(1 for s in sampled_scenes if not s['is_night']),
            'night': sum(1 for s in sampled_scenes if s['is_night']),
        },
        'location_stats': {
            'boston': sum(1 for s in sampled_scenes if 'boston' in s['location'].lower()),
            'singapore': sum(1 for s in sampled_scenes if 'singapore' in s['location'].lower()),
        }
    }
    
    # 根据实际采样数量保存文件
    filename = f'sampled_scenes_{len(sampled_scenes)}.json'
    with open(filename, 'w') as f:
        import json
        json.dump(sampling_log, f, indent=2)
    
    print(f"📄 采样日志已保存到: {filename}")
    
    return result_scenes


def load_sampled_scenes(nusc, sampling_log_path='sampled_scenes_200.json'):
    """从采样日志加载场景（用于复现）"""
    import json
    
    with open(sampling_log_path, 'r') as f:
        log = json.load(f)
    
    scene_names = set(log['sampled_scene_names'])
    result_scenes = [s for s in nusc.scene if s['name'] in scene_names]
    
    print(f"✅ 从日志加载 {len(result_scenes)} 个场景")
    return result_scenes


# 测试代码
if __name__ == "__main__":
    import sys
    import io
    # 设置标准输出编码为UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 命令行参数
    dataroot = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/79120/OpenEMMA/openemma'
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # 默认100场景
    
    print(f"🔄 加载nuScenes trainval数据集...")
    print(f"   数据路径: {dataroot}")
    print(f"   版本: v1.0-trainval (700 train + 150 val = 850 scenes)")
    print(f"   ⚠️  不使用test集(150 scenes)，保留给最终竞赛评估")
    print(f"   目标采样: {n_samples} 个场景\n")
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    
    # 智能采样
    sampled = smart_sample_scenes(nusc, n_samples=n_samples, seed=42)
    
    print(f"\n✅ 采样完成！共 {len(sampled)} 个场景")
    print(f"   采样率: {len(sampled)}/850 = {len(sampled)/850*100:.1f}%")
    print(f"   第一个场景: {sampled[0]['name']}")
    print(f"   最后一个场景: {sampled[-1]['name']}")
    print(f"\n📄 采样配置已保存到: sampled_scenes_{len(sampled)}.json")
    print(f"   可用于复现实验（seed=42）")
    print(f"\n💡 使用方法:")
    print(f"   python main.py --version v1.0-trainval --use-sampled-200 ...")

