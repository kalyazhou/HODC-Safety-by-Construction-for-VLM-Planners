#!/usr/bin/env python3
"""
BEV轨迹可视化工具
使用matplotlib生成专业的BEV轨迹对比图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import argparse
import os
from pathlib import Path

def load_trajectory_data(jsonl_path):
    """加载轨迹数据"""
    trajectories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            trajectories.append(data)
    return trajectories

def draw_bev_map(ax, extent=50, resolution=0.2):
    """绘制BEV地图背景"""
    # 绘制网格
    grid_size = 10
    for i in range(-extent, extent + 1, grid_size):
        ax.axvline(i, color='lightgray', alpha=0.3, linewidth=0.5)
        ax.axhline(i, color='lightgray', alpha=0.3, linewidth=0.5)
    
    # 绘制坐标轴
    ax.axvline(0, color='black', linewidth=1)
    ax.axhline(0, color='black', linewidth=1)
    
    # 设置范围
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    
    # 添加标签
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Bird\'s Eye View - Trajectory Planning')

def draw_ego_vehicle(ax, x, y, yaw, color='blue', size=2):
    """绘制自车"""
    # 计算车辆四个角点
    half_length = size
    half_width = size * 0.6
    
    # 车辆矩形
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 四个角点（相对于车辆中心）
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    # 旋转
    rotated_corners = np.array([
        [cos_yaw * corner[0] - sin_yaw * corner[1],
         sin_yaw * corner[0] + cos_yaw * corner[1]]
        for corner in corners
    ])
    
    # 平移到实际位置
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # 绘制车辆
    vehicle = patches.Polygon(rotated_corners, closed=True, 
                             facecolor=color, alpha=0.7, edgecolor='black')
    ax.add_patch(vehicle)
    
    # 绘制方向箭头
    arrow_length = size * 1.5
    arrow_x = x + cos_yaw * arrow_length
    arrow_y = y + sin_yaw * arrow_length
    ax.arrow(x, y, cos_yaw * arrow_length, sin_yaw * arrow_length,
             head_width=0.5, head_length=0.5, fc=color, ec=color)

def draw_trajectory(ax, traj_xy, color='red', linewidth=2, label='Predicted', alpha=0.8):
    """绘制轨迹"""
    if len(traj_xy) < 2:
        return
    
    x_coords = [point[0] for point in traj_xy]
    y_coords = [point[1] for point in traj_xy]
    
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, 
            label=label, alpha=alpha, marker='o', markersize=3)

def draw_other_vehicles(ax, vehicles, color='orange', alpha=0.6):
    """绘制其他车辆"""
    for i, vehicle in enumerate(vehicles):
        x, y = vehicle['pos_xy']
        # 简单绘制为圆形
        circle = patches.Circle((x, y), 1.5, facecolor=color, alpha=alpha, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y+2, f'V{i+1}', ha='center', va='bottom', fontsize=8)

def create_bev_visualization(trajectory_data, output_path, scene_name):
    """创建BEV可视化"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制地图背景
    draw_bev_map(ax)
    
    # 绘制自车初始位置
    if 'ego_pose' in trajectory_data:
        ego_x = trajectory_data['ego_pose']['translation'][0]
        ego_y = trajectory_data['ego_pose']['translation'][1]
        ego_yaw = trajectory_data['ego_pose']['rotation'][2]  # 假设是yaw角
        draw_ego_vehicle(ax, ego_x, ego_y, ego_yaw, color='blue', size=2)
    
    # 绘制预测轨迹
    if 'predicted_trajectory' in trajectory_data:
        pred_traj = trajectory_data['predicted_trajectory']
        draw_trajectory(ax, pred_traj, color='red', linewidth=3, label='Predicted Trajectory')
    
    # 绘制真实轨迹
    if 'ground_truth_trajectory' in trajectory_data:
        gt_traj = trajectory_data['ground_truth_trajectory']
        draw_trajectory(ax, gt_traj, color='green', linewidth=2, label='Ground Truth', alpha=0.7)
    
    # 绘制其他车辆
    if 'other_vehicles' in trajectory_data:
        draw_other_vehicles(ax, trajectory_data['other_vehicles'])
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 添加场景信息
    info_text = f"Scene: {scene_name}\n"
    if 'ade' in trajectory_data:
        info_text += f"ADE: {trajectory_data['ade']:.2f}m\n"
    if 'fde' in trajectory_data:
        info_text += f"FDE: {trajectory_data['fde']:.2f}m"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BEV visualization saved: {output_path}")

def create_comparison_plot(trajectories, output_path, scene_name):
    """创建对比图 - 在同一个图上显示所有轨迹"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制地图背景
    draw_bev_map(ax, extent=50)
    
    # 轨迹颜色和标签
    trajectory_configs = [
        {'color': 'blue', 'label': 'Baseline (Front View)', 'linewidth': 2, 'alpha': 0.8},
        {'color': 'orange', 'label': 'BEV + Original Prompt', 'linewidth': 2, 'alpha': 0.8},
        {'color': 'red', 'label': 'BEV + New Prompt', 'linewidth': 3, 'alpha': 1.0}
    ]
    
    # 绘制自车（只画一次）
    if trajectories and 'ego_pose' in trajectories[0]:
        ego_x = trajectories[0]['ego_pose']['translation'][0]
        ego_y = trajectories[0]['ego_pose']['translation'][1]
        ego_yaw = trajectories[0]['ego_pose']['rotation'][2]
        draw_ego_vehicle(ax, ego_x, ego_y, ego_yaw, color='black', size=2)
    
    # 绘制真实轨迹（参考线）
    if trajectories and 'ground_truth_trajectory' in trajectories[0]:
        gt_traj = trajectories[0]['ground_truth_trajectory']
        draw_trajectory(ax, gt_traj, color='green', linewidth=3, 
                       label='Ground Truth', alpha=0.9)
    
    # 绘制所有预测轨迹
    for i, (traj_data, config) in enumerate(zip(trajectories, trajectory_configs)):
        if 'predicted_trajectory' in traj_data:
            pred_traj = traj_data['predicted_trajectory']
            draw_trajectory(ax, pred_traj, 
                          color=config['color'], 
                          linewidth=config['linewidth'],
                          label=config['label'], 
                          alpha=config['alpha'])
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加性能对比表格
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
            
            ax.text(0.02, 0.98, table_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.title(f'Trajectory Planning Comparison - Scene {scene_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {output_path}")

def create_performance_comparison(data_files, output_dir):
    """创建性能对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 收集所有数据
    all_data = []
    for data_file in data_files:
        trajectories = load_trajectory_data(data_file)
        for traj_data in trajectories:
            all_data.append(traj_data)
    
    # 按场景分组
    scenes = {}
    for data in all_data:
        scene_name = data.get('scene_name', 'unknown')
        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append(data)
    
    # 绘制ADE对比
    scene_names = list(scenes.keys())
    baseline_ade = []
    bev_original_ade = []
    bev_new_ade = []
    
    for scene_name in scene_names:
        scene_data = scenes[scene_name]
        for data in scene_data:
            if data.get('ablation') == 'A':  # Baseline
                baseline_ade.append(data['performance']['avg_ade'])
            elif data.get('ablation') == 'B':  # BEV + Original
                bev_original_ade.append(data['performance']['avg_ade'])
            elif data.get('ablation') == 'C':  # BEV + New
                bev_new_ade.append(data['performance']['avg_ade'])
    
    # ADE柱状图
    models = ['Baseline\n(Front View)', 'BEV + Original\nPrompt', 'BEV + New\nPrompt']
    ade_values = [np.mean(baseline_ade) if baseline_ade else 0,
                  np.mean(bev_original_ade) if bev_original_ade else 0,
                  np.mean(bev_new_ade) if bev_new_ade else 0]
    
    bars1 = ax1.bar(models, ade_values, color=['blue', 'orange', 'red'], alpha=0.7)
    ax1.set_title('Average Displacement Error (ADE) Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ADE (meters)', fontsize=12)
    ax1.set_ylim(0, max(ade_values) * 1.2)
    
    # 添加数值标签
    for bar, value in zip(bars1, ade_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    # FDE对比
    baseline_fde = []
    bev_original_fde = []
    bev_new_fde = []
    
    for scene_name in scene_names:
        scene_data = scenes[scene_name]
        for data in scene_data:
            if data.get('ablation') == 'A':  # Baseline
                baseline_fde.append(data['performance']['avg_fde'])
            elif data.get('ablation') == 'B':  # BEV + Original
                bev_original_fde.append(data['performance']['avg_fde'])
            elif data.get('ablation') == 'C':  # BEV + New
                bev_new_fde.append(data['performance']['avg_fde'])
    
    fde_values = [np.mean(baseline_fde) if baseline_fde else 0,
                  np.mean(bev_original_fde) if bev_original_fde else 0,
                  np.mean(bev_new_fde) if bev_new_fde else 0]
    
    bars2 = ax2.bar(models, fde_values, color=['blue', 'orange', 'red'], alpha=0.7)
    ax2.set_title('Final Displacement Error (FDE) Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('FDE (meters)', fontsize=12)
    ax2.set_ylim(0, max(fde_values) * 1.2)
    
    # 添加数值标签
    for bar, value in zip(bars2, fde_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='BEV Trajectory Visualization')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing trajectory data')
    parser.add_argument('--scene', type=str, help='Specific scene to visualize')
    parser.add_argument('--comparison', action='store_true', help='Create comparison plot')
    parser.add_argument('--output-dir', type=str, default='bev_visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找数据文件
    data_files = list(Path(args.data_dir).glob('**/nuboard_data.jsonl'))
    
    if not data_files:
        print("No nuboard_data.jsonl files found!")
        return
    
    if args.comparison:
        # 创建性能对比图
        create_performance_comparison(data_files, args.output_dir)
    else:
        # 创建单个场景的BEV图
        for data_file in data_files:
            trajectories = load_trajectory_data(data_file)
            
            for traj_data in trajectories:
                scene_name = traj_data.get('scene_name', 'unknown')
                
                if args.scene and args.scene not in scene_name:
                    continue
                
                # 创建单个场景的BEV图
                output_path = os.path.join(args.output_dir, f'{scene_name}_bev.png')
                create_bev_visualization(traj_data, output_path, scene_name)
    
    print(f"\nBEV visualizations saved in: {args.output_dir}")
    print("You can now view the generated PNG files!")

if __name__ == '__main__':
    main()
