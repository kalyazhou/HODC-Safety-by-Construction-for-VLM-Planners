import numpy as np
from typing import Dict, Any

def calculate_trajectory_smoothness(trajectory: np.ndarray) -> Dict[str, float]:
    """
    Calculates various smoothness metrics for a given trajectory.
    Trajectory is expected to be a (N, 3) or (N, 2) array of (x, y, z) or (x, y) waypoints.
    """
    if trajectory.shape[0] < 3: # Need at least 3 points for acceleration/jerk
        return {
            "velocity_cv": np.nan,
            "acceleration_mean": np.nan,
            "acceleration_std": np.nan,
            "jerk_mean": np.nan,
            "jerk_std": np.nan,
            "curvature_mean": np.nan,
            "curvature_std": np.nan,
            "smoothness_score": np.nan # A composite score
        }

    # Calculate velocities
    velocities = np.diff(trajectory[:, :2], axis=0) # (N-1, 2)
    speed = np.linalg.norm(velocities, axis=1) # (N-1,)

    # Calculate accelerations
    accelerations = np.diff(speed, axis=0) # (N-2,)

    # Calculate jerk (rate of change of acceleration)
    jerk = np.diff(accelerations, axis=0) # (N-3,)

    # Calculate curvature (simplified for 2D points)
    # This is a rough approximation, more accurate curvature needs more complex geometry
    dx = np.gradient(trajectory[:, 0])
    dy = np.gradient(trajectory[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = curvature[~np.isnan(curvature)] # Remove NaNs from division by zero

    # Metrics
    velocity_cv = np.std(speed) / np.mean(speed) if np.mean(speed) > 1e-6 else 0.0
    acceleration_mean = np.mean(np.abs(accelerations))
    acceleration_std = np.std(accelerations)
    jerk_mean = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0.0
    jerk_std = np.std(jerk) if len(jerk) > 0 else 0.0
    curvature_mean = np.mean(np.abs(curvature)) if len(curvature) > 0 else 0.0
    curvature_std = np.std(curvature) if len(curvature) > 0 else 0.0

    # A simple composite smoothness score (lower is better)
    # This can be weighted differently based on importance
    smoothness_score = (
        velocity_cv * 0.5 +
        acceleration_std * 0.3 +
        jerk_std * 0.2 +
        curvature_std * 0.1 # Add curvature variation
    )

    return {
        "velocity_cv": float(velocity_cv),
        "acceleration_mean": float(acceleration_mean),
        "acceleration_std": float(acceleration_std),
        "jerk_mean": float(jerk_mean),
        "jerk_std": float(jerk_std),
        "curvature_mean": float(curvature_mean),
        "curvature_std": float(curvature_std),
        "smoothness_score": float(smoothness_score)
    }

def compare_trajectory_smoothness(
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    gt_name: str = "Ground Truth",
    pred_name: str = "Prediction"
) -> Dict[str, Any]:
    """
    Compares smoothness metrics between ground truth and predicted trajectories.
    """
    gt_metrics = calculate_trajectory_smoothness(gt_trajectory)
    pred_metrics = calculate_trajectory_smoothness(pred_trajectory)

    comparison = {
        gt_name: gt_metrics,
        pred_name: pred_metrics,
        "improvement": {}
    }

    for key in gt_metrics:
        if gt_metrics[key] is not np.nan and pred_metrics[key] is not np.nan and gt_metrics[key] > 1e-6:
            improvement = ((gt_metrics[key] - pred_metrics[key]) / gt_metrics[key]) * 100
            comparison["improvement"][key] = float(improvement)
        else:
            comparison["improvement"][key] = np.nan # Cannot calculate improvement

    return comparison

def print_smoothness_analysis(comparison_results: Dict[str, Any]):
    """
    Prints the smoothness analysis results in a readable format.
    """
    keys = list(comparison_results.keys())
    if len(keys) < 2:
        # Single trajectory analysis
        gt_name = keys[0] if keys else "Trajectory"
        pred_name = None
    else:
        gt_name = keys[0]
        pred_name = keys[1]

    print(f"\n{gt_name}:")
    print("-" * 60)
    for key, value in comparison_results[gt_name].items():
        print(f"{key:<20}: {value:.4f}")

    if pred_name:
        print(f"\n{pred_name}:")
        print("-" * 60)
        for key, value in comparison_results[pred_name].items():
            print(f"{key:<20}: {value:.4f}")

        print("\nIMPROVEMENT ANALYSIS (lower is better, so positive % is improvement):")
        print("-" * 60)
        has_significant_improvement = False
        for key, value in comparison_results["improvement"].items():
            if not np.isnan(value):
                print(f"{key:<20}: {value:+.2f}% {'↓' if value > 0 else '↑' if value < 0 else ''}")
                if key == "smoothness_score" and value > 5: # Example threshold for "significant"
                    has_significant_improvement = True
            else:
                print(f"{key:<20}: N/A")

        if has_significant_improvement:
            print("\n SIGNIFICANT SMOOTHNESS IMPROVEMENT!")
        else:
            print("\nNo significant smoothness improvement observed.")
    else:
        print("\nSingle trajectory analysis completed.")
