#!/usr/bin/env python3
"""
BEV Trajectory Visualization Tool
Generate professional BEV trajectory comparison plots with matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import argparse
import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes  # corrected import


def load_trajectory_data(jsonl_path):
    """Load trajectory data from a JSONL file."""
    trajectories = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            trajectories.append(data)
    return trajectories


def draw_bev_map(ax, extent=50, resolution=0.2):
    """Draw BEV map background with grid and axes."""
    # Grid
    grid_size = 10
    for i in range(-extent, extent + 1, grid_size):
        ax.axvline(i, color='lightgray', alpha=0.3, linewidth=0.5)
        ax.axhline(i, color='lightgray', alpha=0.3, linewidth=0.5)

    # Axes
    ax.axvline(0, color='black', linewidth=1)
    ax.axhline(0, color='black', linewidth=1)

    # Limits & aspect
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')

    # Labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title("Bird's Eye View - Trajectory Planning")


def draw_ego_vehicle(ax, x, y, yaw, color='blue', size=2):
    """Draw ego vehicle as a rotated rectangle with a heading arrow."""
    half_length = size
    half_width = size * 0.6

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Rectangle corners (relative to center)
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])

    # Rotate
    rotated_corners = np.array([
        [cos_yaw * c[0] - sin_yaw * c[1],
         sin_yaw * c[0] + cos_yaw * c[1]]
        for c in corners
    ])

    # Translate to position
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y

    # Vehicle body
    vehicle = patches.Polygon(
        rotated_corners, closed=True,
        facecolor=color, alpha=0.7, edgecolor='black'
    )
    ax.add_patch(vehicle)

    # Heading arrow
    arrow_length = size * 1.5
    ax.arrow(
        x, y, cos_yaw * arrow_length, sin_yaw * arrow_length,
        head_width=0.5, head_length=0.5, fc=color, ec=color
    )


def draw_trajectory(ax, traj_xy, color='red', linewidth=2, label='Predicted', alpha=0.8):
    """Plot a single trajectory polyline with small markers."""
    if len(traj_xy) < 2:
        return

    x_coords = [p[0] for p in traj_xy]
    y_coords = [p[1] for p in traj_xy]

    ax.plot(
        x_coords, y_coords,
        color=color, linewidth=linewidth,
        label=label, alpha=alpha,
        marker='o', markersize=3
    )


def draw_other_vehicles(ax, vehicles, color='orange', alpha=0.6):
    """Draw other vehicles as circles with small labels."""
    for i, vehicle in enumerate(vehicles):
        x, y = vehicle['pos_xy']
        circle = patches.Circle((x, y), 1.5, facecolor=color, alpha=alpha, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y + 2, f'V{i+1}', ha='center', va='bottom', fontsize=8)


def create_bev_visualization(trajectory_data, output_path, scene_name):
    """Create a single-scene BEV visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Map background
    draw_bev_map(ax)

    # Ego vehicle (assumes yaw stored directly; if quaternion, convert before calling)
    if 'ego_pose' in trajectory_data:
        ego_x = trajectory_data['ego_pose']['translation'][0]
        ego_y = trajectory_data['ego_pose']['translation'][1]
        ego_yaw = trajectory_data['ego_pose']['rotation'][2]  # assuming yaw angle
        draw_ego_vehicle(ax, ego_x, ego_y, ego_yaw, color='blue', size=2)

    # Predicted trajectory
    if 'predicted_trajectory' in trajectory_data:
        pred_traj = trajectory_data['predicted_trajectory']
        draw_trajectory(ax, pred_traj, color='red', linewidth=3, label='Predicted Trajectory')

    # Ground-truth trajectory (reference)
    if 'ground_truth_trajectory' in trajectory_data:
        gt_traj = trajectory_data['ground_truth_trajectory']
        draw_trajectory(ax, gt_traj, color='green', linewidth=2, label='Ground Truth', alpha=0.7)

    # Other agents
    if 'other_vehicles' in trajectory_data:
        draw_other_vehicles(ax, trajectory_data['other_vehicles'])

    # Legend
    ax.legend(loc='upper right')

    # Scene info block
    info_text = f"Scene: {scene_name}\n"
    if 'ade' in trajectory_data:
        info_text += f"ADE: {trajectory_data['ade']:.2f} m\n"
    if 'fde' in trajectory_data:
        info_text += f"FDE: {trajectory_data['fde']:.2f} m"

    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BEV visualization saved: {output_path}")


def create_comparison_plot(trajectories, output_path, scene_name):
    """Create a comparison plot showing multiple predicted trajectories in one figure."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Map background
    draw_bev_map(ax, extent=50)

    # Trajectory color/label configs
    trajectory_configs = [
        {'color': 'blue', 'label': 'Baseline (Front View)', 'linewidth': 2, 'alpha': 0.8},
        {'color': 'orange', 'label': 'BEV + Original Prompt', 'linewidth': 2, 'alpha': 0.8},
        {'color': 'red', 'label': 'BEV + New Prompt', 'linewidth': 3, 'alpha': 1.0}
    ]

    # Ego (draw once)
    if trajectories and 'ego_pose' in trajectories[0]:
        ego_x = trajectories[0]['ego_pose']['translation'][0]
        ego_y = trajectories[0]['ego_pose']['translation'][1]
        ego_yaw = trajectories[0]['ego_pose']['rotation'][2]
        draw_ego_vehicle(ax, ego_x, ego_y, ego_yaw, color='black', size=2)

    # Ground truth (reference)
    if trajectories and 'ground_truth_trajectory' in trajectories[0]:
        gt_traj = trajectories[0]['ground_truth_trajectory']
        draw_trajectory(ax, gt_traj, color='green', linewidth=3, label='Ground Truth', alpha=0.9)

    # All predictions
    for i, (traj_data, config) in enumerate(zip(trajectories, trajectory_configs)):
        if 'predicted_trajectory' in traj_data:
            pred_traj = traj_data['predicted_trajectory']
            draw_trajectory(
                ax, pred_traj,
                color=config['color'],
                linewidth=config['linewidth'],
                label=config['label'],
                alpha=config['alpha']
            )

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Performance table (text block)
    if trajectories:
        table_data = []
        for i, traj_data in enumerate(trajectories):
            if 'ade' in traj_data and 'fde' in traj_data:
                model_name = trajectory_configs[i]['label']
                ade = traj_data['ade']
                fde = traj_data['fde']
                table_data.append([model_name, f"{ade:.2f}", f"{fde:.2f}"])

        if table_data:
            table_text = "Performance Comparison:\n"
            table_text += "Model".ljust(25) + "ADE".ljust(10) + "FDE\n"
            table_text += "-" * 45 + "\n"
            for row in table_data:
                table_text += f"{row[0].ljust(25)}{row[1].ljust(10)}{row[2]}\n"

            ax.text(
                0.02, 0.98, table_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            )

    plt.title(f'Trajectory Planning Comparison - Scene {scene_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def create_performance_comparison(data_files, output_dir):
    """Create ADE/FDE bar charts comparing multiple methods across scenes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect all entries
    all_data = []
    for data_file in data_files:
        trajectories = load_trajectory_data(data_file)
        for traj_data in trajectories:
            all_data.append(traj_data)

    # Group by scene (optional; kept for compatibility)
    scenes = {}
    for data in all_data:
        scene_name = data.get('scene_name', 'unknown')
        scenes.setdefault(scene_name, []).append(data)

    # Extract ADE values by ablation
    baseline_ade, bev_original_ade, bev_new_ade = [], [], []
    for scene_name, scene_data in scenes.items():
        for data in scene_data:
            if data.get('ablation') == 'A':
                baseline_ade.append(data['performance']['avg_ade'])
            elif data.get('ablation') == 'B':
                bev_original_ade.append(data['performance']['avg_ade'])
            elif data.get('ablation') == 'C':
                bev_new_ade.append(data['performance']['avg_ade'])

    models = ['Baseline\n(Front View)', 'BEV + Original\nPrompt', 'BEV + New\nPrompt']
    ade_values = [
        np.mean(baseline_ade) if baseline_ade else 0.0,
        np.mean(bev_original_ade) if bev_original_ade else 0.0,
        np.mean(bev_new_ade) if bev_new_ade else 0.0
    ]

    bars1 = ax1.bar(models, ade_values, color=['blue', 'orange', 'red'], alpha=0.7)
    ax1.set_title('Average Displacement Error (ADE) Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ADE (meters)', fontsize=12)
    ade_upper = max(ade_values) * 1.2 if max(ade_values) > 0 else 1.0
    ax1.set_ylim(0, ade_upper)

    for bar, value in zip(bars1, ade_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ade_upper * 0.02,
            f'{value:.2f} m',
            ha='center', va='bottom', fontweight='bold'
        )

    # Extract FDE values by ablation
    baseline_fde, bev_original_fde, bev_new_fde = [], [], []
    for scene_name, scene_data in scenes.items():
        for data in scene_data:
            if data.get('ablation') == 'A':
                baseline_fde.append(data['performance']['avg_fde'])
            elif data.get('ablation') == 'B':
                bev_original_fde.append(data['performance']['avg_fde'])
            elif data.get('ablation') == 'C':
                bev_new_fde.append(data['performance']['avg_fde'])

    fde_values = [
        np.mean(baseline_fde) if baseline_fde else 0.0,
        np.mean(bev_original_fde) if bev_original_fde else 0.0,
        np.mean(bev_new_fde) if bev_new_fde else 0.0
    ]

    bars2 = ax2.bar(models, fde_values, color=['blue', 'orange', 'red'], alpha=0.7)
    ax2.set_title('Final Displacement Error (FDE) Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('FDE (meters)', fontsize=12)
    fde_upper = max(fde_values) * 1.2 if max(fde_values) > 0 else 1.0
    ax2.set_ylim(0, fde_upper)

    for bar, value in zip(bars2, fde_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + fde_upper * 0.02,
            f'{value:.2f} m',
            ha='center', va='bottom', fontweight='bold'
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='BEV Trajectory Visualization')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing trajectory data')
    parser.add_argument('--scene', type=str, help='Specific scene name substring to visualize')
    parser.add_argument('--comparison', action='store_true', help='Create performance comparison chart')
    parser.add_argument('--output-dir', type=str, default='bev_visualizations', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find data files
    data_files = list(Path(args.data_dir).glob('**/nuboard_data.jsonl'))
    if not data_files:
        print("No nuboard_data.jsonl files found!")
        return

    if args.comparison:
        # Create performance comparison charts
        create_performance_comparison(data_files, args.output_dir)
    else:
        # Create per-scene BEV figures
        for data_file in data_files:
            trajectories = load_trajectory_data(data_file)

            for traj_data in trajectories:
                scene_name = traj_data.get('scene_name', 'unknown')

                if args.scene and args.scene not in scene_name:
                    continue

                output_path = os.path.join(args.output_dir, f'{scene_name}_bev.png')
                create_bev_visualization(traj_data, output_path, scene_name)

    print(f"\nBEV visualizations saved in: {args.output_dir}")
    print("You can now view the generated PNG files!")


if __name__ == '__main__':
    main()
