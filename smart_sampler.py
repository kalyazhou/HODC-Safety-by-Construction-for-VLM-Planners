#!/usr/bin/env python3
"""
Intelligent sampling module — sample 200 representative scenes from nuScenes Full.
Use case: expand experiments from 10 scenes (mini) → 200 scenes (sampled full)
"""

import os
import json
import random
import numpy as np
from nuscenes.nuscenes import NuScenes


def check_scene_has_images(nusc, scene):
    """Check if required image files exist for the scene."""
    try:
        first_sample = nusc.get('sample', scene['first_sample_token'])
        cam_front_token = first_sample['data']['CAM_FRONT']
        cam_front_data = nusc.get('sample_data', cam_front_token)
        image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
        return os.path.exists(image_path)
    except Exception:
        return False


def analyze_scene_features(nusc, scene):
    """Analyze scene features (for stratified sampling)."""
    first_sample = nusc.get('sample', scene['first_sample_token'])

    # 1) Time-of-day feature (day/night) — rough heuristic from timestamp (microseconds)
    timestamp = first_sample['timestamp']
    hour = (timestamp // 1_000_000) % 86_400 // 3600
    is_night = (hour < 6 or hour > 20)

    # 2) Location feature
    location = nusc.get('log', scene['log_token'])['location']

    # 3) Scene description
    description = (scene.get('description') or "").lower()

    # 4) Complexity estimation via keywords
    complexity = 'simple'
    complex_keywords = ['turn', 'intersection', 'construction', 'crowded', 'heavy', 'stop']
    medium_keywords = ['lane', 'merge', 'traffic']
    if any(kw in description for kw in complex_keywords):
        complexity = 'complex'
    elif any(kw in description for kw in medium_keywords):
        complexity = 'medium'

    # 5) Scene length (frame count)
    frame_count = scene['nbr_samples']

    return:
    return {
        'name': scene['name'],
        'token': scene['token'],
        'location': location,
        'is_night': is_night,
        'complexity': complexity,
        'frame_count': frame_count,
        'description': scene.get('description', '')
    }


def smart_sample_scenes(nusc, n_samples=200, seed=42):
    """
    Intelligent sampling of nuScenes scenes.

    Notes:
      1. Sample only from trainval (850 scenes); test set annotations are not public.
      2. This project runs zero-shot inference (no parameter training).
      3. Overfitting is not a concern; all 850 trainval scenes are valid for evaluation.
      4. No need to split train/val for inference-only evaluation.

    nuScenes structure:
      - trainval: 850 scenes (700 train + 150 val, annotations public)  — usable
      - test:     150 scenes (annotations hidden, leaderboard only)     — not usable
      - mini:     10 scenes (subset of trainval)                         — already used

    Target distribution (example for 200 scenes = 23.5% of trainval):
      - Complexity: simple 35%, medium 47%, complex 18%
      - Time: day 70%, night 30%
      - Location: Boston 50%, Singapore 50%
      - Avoid known problematic scenes (e.g., extreme curvature, stationary vehicle issues)

    Args:
        nusc: NuScenes(version='v1.0-trainval') object
        n_samples: number of scenes to sample (default: 200)
        seed: random seed (for reproducibility, default: 42)

    Returns:
        list: sampled scene dicts (nuScenes "scene" table rows)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Known problematic scenes to exclude
    problematic_scenes = ['scene-0553', 'scene-0757', 'scene-1100']

    # Analyze all scenes and ensure images exist
    print("Analyzing nuScenes scene features...")
    all_scenes = []
    scenes_without_images = 0

    for scene in nusc.scene:
        if scene['name'] in problematic_scenes:
            continue
        if not check_scene_has_images(nusc, scene):
            scenes_without_images += 1
            continue
        features = analyze_scene_features(nusc, scene)
        all_scenes.append(features)

    print(f"Total valid scenes with image data: {len(all_scenes)}")
    if scenes_without_images > 0:
        print(f"Skipped scenes due to missing image files: {scenes_without_images}")

    # Group buckets
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

    # Print distribution
    print("\nScene distribution by bucket:")
    for key, scenes in grouped.items():
        if scenes:
            print(f"  {key}: {len(scenes)} scenes")

    # Base distribution for 200 scenes (fractions)
    base_distribution = {
        'simple_day_boston': 25 / 200,
        'simple_day_singapore': 25 / 200,
        'simple_night_boston': 10 / 200,
        'simple_night_singapore': 10 / 200,
        'medium_day_boston': 33 / 200,
        'medium_day_singapore': 33 / 200,
        'medium_night_boston': 14 / 200,
        'medium_night_singapore': 14 / 200,
        'complex_day_boston': 13 / 200,
        'complex_day_singapore': 13 / 200,
        'complex_night_boston': 5 / 200,
        'complex_night_singapore': 5 / 200,
    }

    # Scale to requested n_samples
    target_distribution = {key: max(1, int(ratio * n_samples))
                           for key, ratio in base_distribution.items()}

    # Adjust to match exact n_samples
    total_allocated = sum(target_distribution.values())
    if total_allocated < n_samples:
        target_distribution['medium_day_boston'] += (n_samples - total_allocated)
    elif total_allocated > n_samples:
        target_distribution['medium_day_boston'] -= (total_allocated - n_samples)

    # Stratified sampling
    sampled_scenes = []
    remaining_pool = []

    for key, target_count in target_distribution.items():
        available = grouped.get(key, [])
        if len(available) == 0:
            print(f"Warning: {key}: no available scenes, skipped")
            continue

        if len(available) <= target_count:
            sampled = available
            print(f"{key}: sampled {len(sampled)}/{target_count} (took all available)")
        else:
            sampled = random.sample(available, target_count)
            print(f"{key}: sampled {len(sampled)}/{target_count}")
            remaining = [s for s in available if s not in sampled]
            remaining_pool.extend(remaining)

        sampled_scenes.extend(sampled)

    # Top up from remaining pool if needed
    shortage = n_samples - len(sampled_scenes)
    if shortage > 0 and remaining_pool:
        extra_samples = min(shortage, len(remaining_pool))
        extras = random.sample(remaining_pool, extra_samples)
        sampled_scenes.extend(extras)
        print(f"\nAdded {extra_samples} scenes from the remaining pool")

    print(f"\nTotal sampled scenes: {len(sampled_scenes)}")

    # Convert back to canonical nuScenes 'scene' rows
    sampled_tokens = {s['token'] for s in sampled_scenes}
    result_scenes = [s for s in nusc.scene if s['token'] in sampled_tokens]

    # Save sampling log for reproducibility
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

    filename = f'sampled_scenes_{len(sampled_scenes)}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sampling_log, f, indent=2, ensure_ascii=False)

    print(f"Sampling log saved to: {filename}")

    return result_scenes


def load_sampled_scenes(nusc, sampling_log_path='sampled_scenes_200.json'):
    """Load scenes from a sampling log (for reproducibility)."""
    with open(sampling_log_path, 'r', encoding='utf-8') as f:
        log = json.load(f)

    scene_names = set(log['sampled_scene_names'])
    result_scenes = [s for s in nusc.scene if s['name'] in scene_names]

    print(f"Loaded {len(result_scenes)} scenes from log")
    return result_scenes


# CLI test
if __name__ == "__main__":
    import sys

    dataroot = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/79120/OpenEMMA/openemma'
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200  
    print("Loading nuScenes trainval dataset...")
    print(f"Data root: {dataroot}")
    print("Version: v1.0-trainval (700 train + 150 val = 850 scenes)")
    print("Not using the test set (150 scenes); reserved for leaderboard evaluation.")
    print(f"Target sample size: {n_samples}\n")

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    sampled = smart_sample_scenes(nusc, n_samples=n_samples, seed=42)

    print(f"\nSampling complete. Total scenes: {len(sampled)}")
    print(f"Sampling rate: {len(sampled)}/850 = {len(sampled)/850*100:.1f}%")
    if sampled:
        print(f"First scene: {sampled[0]['name']}")
        print(f"Last scene:  {sampled[-1]['name']}")
    print(f"\nSampling config saved to: sampled_scenes_{len(sampled)}.json")
    print("Reproducible with seed=42.")
    print("\nUsage:")
