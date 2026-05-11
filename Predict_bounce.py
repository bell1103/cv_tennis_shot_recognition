# code from past project 

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import PchipInterpolator
from tabulate import tabulate
import os

# -----------------------
# data cleaing functions 
#------------------------

def remove_coordinate_outliers(data, z_thresh=2.5):
    """
    Removes outlier coordinates (extreme x/y positions) based on z-scores.
    Ignores (0, 0) entries which likely represent missing data.
    """
    coords = np.array([[x, y] for _, x, y in data if not (x == 0 and y == 0)])
    if len(coords) < 5:
        return data  # Too few points to filter
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    z_scores = np.abs((coords - mean) / std)
    keep_indices = np.all(z_scores < z_thresh, axis=1)

    # Keep only non-outlier points
    filtered_data = []
    j = 0
    for frame, x, y in data:
        if not (x == 0 and y == 0):
            if keep_indices[j]:
                filtered_data.append([frame, x, y])
            j += 1
        else:
            filtered_data.append([frame, x, y])
    return filtered_data


def interpolate_missing_points(data):
    """
    Fills missing (0,0) points by interpolating between valid data using PCHIP interpolation.
    Creates smooth and realistic trajectories for ball motion.
    """
    valid_data = [(frame, x, y) for frame, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 3:
        return data  # Not enough points to interpolate
    frames = np.array([frame for frame, _, _ in valid_data])
    x_coords = np.array([x for _, x, _ in valid_data])
    y_coords = np.array([y for _, _, y in valid_data])

    # PCHIP preserves shape and avoids overshooting
    f_x = PchipInterpolator(frames, x_coords)
    f_y = PchipInterpolator(frames, y_coords)

    all_frames = list(range(int(frames.min()), int(frames.max()) + 1))
    interpolated_data = []
    for frame in all_frames:
        if frame in frames:
            idx = np.where(frames == frame)[0][0]
            interpolated_data.append([frame, x_coords[idx], y_coords[idx]])
        else:
            interpolated_data.append([frame, float(f_x(frame)), float(f_y(frame))])
    return interpolated_data


# ---------------------------
# Bounce detection Functions
# ---------------------------

def detect_local_maxima_bounces(data, distance=20, prominence=15, max_x_velocity=5.0):
    """
    Detects bounces by finding local maxima in the smoothed Y trajectory (ball height).
    A bounce typically corresponds to a peak when the ball rebounds upward.
    """
    valid_data = [(f, x, y) for f, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 5:
        return []

    frames = np.array([f for f, _, _ in valid_data])
    x_vals = np.array([x for _, x, _ in valid_data])
    y_vals = np.array([y for _, _, y in valid_data])

    # Smooth data to reduce noise
    window = min(7, len(y_vals) if len(y_vals) % 2 == 1 else len(y_vals) - 1)
    x_smooth = savgol_filter(x_vals, window, polyorder=2)
    y_smooth = savgol_filter(y_vals, window, polyorder=2)

    # Find peaks in Y (local maxima)
    peaks, _ = find_peaks(y_smooth, distance=distance, prominence=prominence)
    bounces = [(frames[i], valid_data[i][1], valid_data[i][2]) for i in peaks]
    return bounces


def detect_bounces_by_slope_sign_change(data, distance=20):
    """
    Detects bounces by identifying sign changes in vertical velocity (dy/dt).
    A bounce occurs when the Y velocity changes direction (up ↔ down).
    """
    valid_data = [(f, x, y) for f, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 5:
        return []

    frames = np.array([f for f, _, _ in valid_data])
    y_vals = np.array([y for _, _, y in valid_data])

    # Smooth y-values before calculating velocity
    window = min(7, len(y_vals) if len(y_vals) % 2 == 1 else len(y_vals) - 1)
    y_smooth = savgol_filter(y_vals, window, polyorder=2)
    velocity = np.gradient(y_smooth)

    bounces = []
    for i in range(1, len(velocity)):
        sign_change = ((velocity[i-1] > 0 and velocity[i] <= 0) or
                       (velocity[i-1] < 0 and velocity[i] >= 0))
        if sign_change:
            # Avoid detecting too-close bounces
            if len(bounces) == 0 or (frames[i] - bounces[-1][0]) >= distance:
                bounces.append((frames[i], valid_data[i][1], valid_data[i][2]))
    return bounces


def detect_bounces_velocity_threshold(data, min_frames_between_bounces=50, velocity_threshold=3.0):
    """
    Detects bounces when Y velocity sharply changes direction beyond a threshold.
    Used to identify the first toss and subsequent bounces.
    """
    bounces = []
    valid_data = [(frame, x, y) for frame, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 3:
        return bounces

    # Compute frame-to-frame vertical velocity
    velocities = [valid_data[i][2] - valid_data[i - 1][2] for i in range(1, len(valid_data))]

    # Smooth velocity signal
    if len(velocities) > 5:
        smoothed_velocities = savgol_filter(velocities, 5, 2)
    else:
        smoothed_velocities = velocities

    last_bounce_frame = -min_frames_between_bounces
    for i in range(1, len(smoothed_velocities)):
        current_frame = valid_data[i + 1][0]
        prev_vel = smoothed_velocities[i - 1]
        curr_vel = smoothed_velocities[i]

        # Detect sharp positive-to-negative velocity switch
        if (prev_vel > velocity_threshold and curr_vel < -velocity_threshold):
            if (current_frame - last_bounce_frame) >= min_frames_between_bounces:
                if abs(curr_vel - prev_vel) > velocity_threshold * 2.0:
                    bounces.append((current_frame, valid_data[i + 1][1], valid_data[i + 1][2]))
                    last_bounce_frame = current_frame
    return bounces


def combine_bounce_detections(bounces1, bounces2, frame_tolerance=20):
    """
    Merges bounce detections from two different methods if their frame indices
    are within a small tolerance. Produces an averaged, unified bounce list.
    """
    combined = []
    used_indices = set()
    for i1, (f1, x1, y1) in enumerate(bounces1):
        for i2, (f2, x2, y2) in enumerate(bounces2):
            if i2 in used_indices:
                continue
            if abs(f1 - f2) <= frame_tolerance:
                avg_frame = int(round((f1 + f2) / 2))
                avg_x = (x1 + x2) / 2
                avg_y = (y1 + y2) / 2
                combined.append((avg_frame, avg_x, avg_y))
                used_indices.add(i2)
                break
    combined.sort(key=lambda x: x[0])
    return combined



# -------------------------
#  Main pipeline function 
# -------------------------

def detect_bounces_pipeline(data, visualize=False, save_path=None):
    """
    Main callable function that runs:
      1. Outlier removal
      2. Missing data interpolation
      3. Bounce detection using multiple methods
      4. Optional visualization + saving
    """
    filtered_data = remove_coordinate_outliers(data)
    interpolated_data = interpolate_missing_points(filtered_data)

    # Detect first toss/bounce using velocity-based method
    all_bounces = detect_bounces_velocity_threshold(interpolated_data, 50, 2.5)
    if not all_bounces:
        print("No toss (first bounce) detected.")
        return []

    # Trim data starting from first bounce
    first_bounce_frame = all_bounces[0][0]
    trimmed_data = interpolated_data

    # Run multiple bounce detection methods
    bounces_maxima = detect_local_maxima_bounces(trimmed_data, 10, 5)
    bounces_direction = detect_bounces_by_slope_sign_change(trimmed_data, 10)

    # Combine methods for robust results
    combined_bounces = combine_bounce_detections(bounces_maxima, bounces_direction, 5)

    # Visualization and debugging
    if visualize:
        visualize_analysis(trimmed_data,
                           [bounces_maxima, bounces_direction, combined_bounces],
                           ['Local Maxima', 'Direction Change', 'Combined'])
        print_velocity_table(trimmed_data)

    # Save combined results to file
    if save_path:
        with open(save_path, "w") as out_file:
            json.dump(combined_bounces, out_file, indent=2)
        print(f"Saved combined bounces to {save_path}")

    return combined_bounces


# -------------------------------------
# Command-line interface (CLI) 
# --------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bounces from tennis tracking data")
    parser.add_argument("input_json", type=str, help="Path to input JSON file with ball coordinates")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save detected bounces")
    parser.add_argument("--visualize", action="store_true", help="Show bounce detection plots")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_json):
        print(f"Error: File {args.input_json} does not exist.")
        exit(1)

    # Load JSON data and run pipeline
    with open(args.input_json, "r") as f:
        data = json.load(f)

    detect_bounces_pipeline(data, visualize=args.visualize, save_path=args.output)
