from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import numpy as np
from collections import Counter
import argparse
import math

OBS_RADIUS_M = 35.0        # Radius to count nearby objects (meters)
CURV_WIN = 15              # Window length (frames) for curvature estimation
DT = 0.5                   # Frame interval in seconds (CAM_FRONT is ~0.5 s)

PEDE_LABELS = ('human.pedestrian',)
CYCLE_LABELS = ('vehicle.bicycle', 'vehicle.motorcycle')
VEH_LABEL = 'vehicle.'
TL_LABEL  = 'traffic_light'  # Category name as defined in NuScenes

def est_curvature(poses_xy):
    """Estimate curvature statistics from a sequence of (x, y) ego positions."""
    if len(poses_xy) < 3:
        # Not enough points to estimate curvature; return zeros
        return 0.0, 0.0

    # Quick curvature approximation: use the circumcircle radius of 3 consecutive points
    curvs = []
    for i in range(1, len(poses_xy) - 1):
        x1, y1 = poses_xy[i - 1]
        x2, y2 = poses_xy[i]
        x3, y3 = poses_xy[i + 1]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        s = 0.5 * (a + b + c)
        area = max(1e-6, math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c))))
        R = (a * b * c) / (4 * area) if area > 0 else 1e6
        k = 1.0 / max(R, 1e-6)
        curvs.append(k)
    return float(np.mean(curvs)), float(np.std(curvs))

def score_scene(nusc, scene):
    """Score a scene by traffic density, presence of lights, curvature variation, and night condition."""
    # Collect per-frame ego poses and counts of nearby objects
    token = scene['first_sample_token']
    poses_xy = []
    near_counts = Counter()
    has_tl = False
    timestamps = []

    while token:
        sample = nusc.get('sample', token)
        sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        px, py = pose['translation'][:2]
        poses_xy.append((px, py))
        timestamps.append(pose['timestamp'])

        # Count objects within 35 m (pedestrians, cyclists, vehicles)
        cnt_ped = cnt_cyc = cnt_veh = 0
        for ann_tk in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tk)
            cx, cy = ann['translation'][:2]
            if np.hypot(cx - px, cy - py) <= OBS_RADIUS_M:
                # Safely retrieve category name
                try:
                    if 'category_token' in ann:
                        cat = nusc.get('category', ann['category_token'])['name']
                    elif 'instance_token' in ann:
                        instance = nusc.get('instance', ann['instance_token'])
                        cat = nusc.get('category', instance['category_token'])['name']
                    else:
                        continue
                except (KeyError, TypeError):
                    continue

                if cat.startswith(TL_LABEL):
                    has_tl = True
                if any(cat.startswith(x) for x in PEDE_LABELS):
                    cnt_ped += 1
                elif any(cat.startswith(x) for x in CYCLE_LABELS):
                    cnt_cyc += 1
                elif cat.startswith(VEH_LABEL):
                    cnt_veh += 1
        near_counts['ped'] += cnt_ped
        near_counts['cyc'] += cnt_cyc
        near_counts['veh'] += cnt_veh

        token = sample['next']

    # Curvature features (curve / lane-change tendency)
    curv_mu, curv_std = est_curvature(poses_xy)

    # Simple day/night heuristic: derive from the hour (UTC) of the first timestamp
    # For precise lighting, image data or richer log fields would be required; this is a coarse proxy
    hour_utc = int((timestamps[0] // 1_000_000 // 3600) % 24) if timestamps else 12
    is_night = (hour_utc < 6 or hour_utc >= 18)

    # Composite difficulty score (tune weights according to your focus)
    score = (
        0.9 * min(near_counts['veh'] / 50, 1.0) +   # Vehicle density
        1.0 * min(near_counts['ped'] / 10, 1.0) +   # Pedestrians
        0.6 * min(near_counts['cyc'] / 10, 1.0) +   # Cyclists
        0.8 * (1.0 if has_tl else 0.0) +            # Traffic lights present
        0.8 * min(curv_std * 200, 1.0) +            # Curvature variability
        0.5 * (1.0 if is_night else 0.0)            # Night scenes
    )

    tags = []
    if is_night: tags.append('night')
    if has_tl:   tags.append('traffic_light')
    if curv_std > 0.01: tags.append('curvy')
    if near_counts['ped'] > 0: tags.append('pedestrians')
    if near_counts['cyc'] > 0: tags.append('cyclists')
    if near_counts['veh'] > 60: tags.append('heavy_traffic')

    return dict(
        name=scene['name'],
        description=scene.get('description', '').strip(),
        score=round(float(score), 3),
        veh=near_counts['veh'], ped=near_counts['ped'], cyc=near_counts['cyc'],
        curv_mean=round(curv_mu, 4), curv_std=round(curv_std, 4),
        night=is_night, has_tl=has_tl, tags=tags
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    ap.add_argument("--version", type=str, default="v1.0-mini")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    recs = [score_scene(nusc, s) for s in nusc.scene]
    recs = sorted(recs, key=lambda x: x['score'], reverse=True)

    print("\n=== Top candidates ===")
    for r in recs[:args.topk]:
        print(
            f"{r['name']:>10s}  score={r['score']:.3f}  tags={r['tags']}  "
            f"veh={r['veh']} ped={r['ped']} cyc={r['cyc']}  curv_std={r['curv_std']}  desc={r['description']}"
        )
