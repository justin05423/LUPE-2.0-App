import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.classification import load_data

def behavior_distance_traveled_heatmaps(project_name, selected_groups, selected_conditions):
    """
    Generate distance-traveled statistics and heatmaps for each group and condition.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.

    Returns:
        figs (list): A list of matplotlib Figure objects (one for each group-condition combination).
    """

    # Define the base directory using os.path.join for cross-platform compatibility
    base_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name)
    poses_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")

    poses = load_data(poses_file)

    # Define the directory path for saving figures and CSVs
    directory_path = os.path.join(base_dir, "figures", "behavior_distance-traveled")

    # Conversion factor from pixels to units
    pixels_to_units = 0.0330828  # meters = 0.000330708, cm = 0.0330828
    unit = 'cm'
    bodypart_idx = 38  # Index of the body part to track (e.g., center of mass)

    # Fixed max_count for heatmap scaling
    max_count = 5000  # Arbitrary number to reflect pixel intensity differences

    # List to store figures
    figs = []

    # Calculate distances and generate heatmaps
    for group in selected_groups:
        for condition in selected_conditions:
            poses_selected = poses[group][condition]

            distances_traveled = []
            cumulative_distance_traveled = 0.0

            for file_key in poses_selected:
                pose_data = poses_selected[file_key]
                total_distance_pixels = 0.0
                for frame in range(1, len(pose_data)):
                    # Calculate Euclidean distance between consecutive frames in pixels
                    distance_pixels = np.linalg.norm(
                        pose_data[frame][bodypart_idx:bodypart_idx + 2] - pose_data[frame - 1][
                                                                          bodypart_idx:bodypart_idx + 2])
                    total_distance_pixels += distance_pixels
                # Convert total distance from pixels to units
                total_distance = total_distance_pixels * pixels_to_units
                # Append to list and update cumulative distance traveled
                distances_traveled.append(total_distance)
                cumulative_distance_traveled += total_distance

            distances_traveled = np.array(distances_traveled)

            # Calculate statistics
            average_distance = np.mean(distances_traveled)
            standard_deviation = np.std(distances_traveled)
            sem = standard_deviation / np.sqrt(len(distances_traveled))

            # Save statistics to CSV using pandas (cross-platform compatible)
            stats_data = {
                'Statistic': [
                    'Average distance traveled',
                    'Standard deviation',
                    'Standard error of the mean (SEM)',
                    'Cumulative distance traveled'
                ],
                'Value': [
                    f'{average_distance:.2f} {unit}',
                    f'{standard_deviation:.2f} {unit}',
                    f'{sem:.2f} {unit}',
                    f'{cumulative_distance_traveled:.2f} {unit}'
                ]
            }
            df = pd.DataFrame(stats_data)

            # Ensure directory exists before each write
            os.makedirs(directory_path, exist_ok=True)

            output_filename = os.path.join(directory_path, f"behavior_distance_stats-{unit}_{group}_{condition}.csv")
            df.to_csv(output_filename, index=False)

            # Generate heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            heatmap, xedges, yedges = np.histogram2d(
                np.hstack([poses_selected[file_key][:, bodypart_idx] for file_key in poses_selected]),
                np.hstack([poses_selected[file_key][:, bodypart_idx + 1] for file_key in poses_selected]),
                bins=50
            )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', interpolation='nearest',
                           vmax=max_count)
            plt.colorbar(im, ax=ax, label='Counts')
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_title(f'{group}_{condition}')

            # Save the figure
            save_path = os.path.join(directory_path, f"behavior_distance-heatmap_{group}_{condition}.svg")
            plt.savefig(save_path, format='svg', bbox_inches='tight')

            # Add the figure to the list
            figs.append(fig)

    return figs
