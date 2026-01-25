# utils/analysis_scripts/behavior_bout_durations.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors  # Assumes these are defined in meta.py


def behavior_bout_durations(project_name, selected_groups, selected_conditions, framerate=60):
    """
    Generate a boxplot figure showing the durations (in seconds) of behavioral bouts
    for each selected group and condition. Also computes and saves per-file summary statistics.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.
        framerate (int): Frame rate for duration conversion (default is 60).

    Returns:
        fig (matplotlib.figure.Figure): The generated figure containing the boxplots.
    """
    # Use the same base directory as your app
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    behaviors = load_behaviors(behaviors_file)

    # Define the directory to save CSVs and figures
    directory_path = os.path.join(base_dir, "figures", "behavior_instance-durations")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Helper function to compute bout durations (in seconds)
    def get_duration_bouts(predict, behavior_classes, framerate=60):
        # Find indices where the behavior label changes
        bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
        # Compute durations for each bout (in frames) and then convert to seconds
        bout_durations = np.hstack([np.diff(bout_start_idx), len(predict) - np.max(bout_start_idx)])
        bout_start_label = predict[bout_start_idx]
        behav_durations = []
        for b, _ in enumerate(behavior_classes):
            idx_b = np.where(bout_start_label == int(b))[0]
            if len(idx_b) > 0:
                behav_durations.append(bout_durations[idx_b] / framerate)
            else:
                behav_durations.append(np.array([np.nan]))
        return behav_durations

    # Determine grid size based on selected groups and conditions
    rows = len(selected_groups)
    cols = len(selected_conditions)
    fig, ax = plt.subplots(rows, cols, figsize=(12, rows * 3), sharex=False, sharey=True)

    # Wrap axes into a 2D array if necessary
    if rows == 1 and cols == 1:
        ax = np.array([[ax]])
    elif rows == 1:
        ax = np.array([ax])
    elif cols == 1:
        ax = np.array([[a] for a in ax])

    # Loop over each group (row) and condition (column)
    for row in range(rows):
        for col in range(cols):
            selected_group = selected_groups[row]
            selected_condition = selected_conditions[col]

            durations_ = []
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())

                # Compute bout durations for each file
                for file_name in file_keys:
                    durations_.append(get_duration_bouts(behaviors[selected_group][selected_condition][file_name],
                                                         behavior_names, framerate))

                # Create a dictionary to build a DataFrame for plotting
                try:
                    durations_dict = {
                        'behavior': np.hstack([
                            np.hstack([np.repeat(behavior_names[i], len(durations_[f][i]))
                                       for i in range(len(durations_[f]))])
                            for f in range(len(durations_))
                        ]),
                        'duration': np.hstack([np.hstack(durations_[f]) for f in range(len(durations_))])
                    }
                except Exception as e:
                    raise ValueError(f"Error processing durations for {selected_group} - {selected_condition}: {e}")

                durations_df = pd.DataFrame(durations_dict)
                # Save the per–group/condition CSV file
                csv_filename = os.path.join(
                    directory_path,
                    f"behavior_durations_{selected_group}_{selected_condition}.csv"
                )
                durations_df.to_csv(csv_filename, index=False)

                # Create a horizontal boxplot of durations using Seaborn
                sns.boxplot(data=durations_df, x='duration', y='behavior', hue='behavior',
                            orient='h', width=0.8, palette=behavior_colors, showfliers=False, ax=ax[row, col])
                ax[row, col].set_ylabel('')
                ax[row, col].set_xlabel('')
                if row == rows - 1:
                    ax[row, col].set_xlabel('Behavior duration (s)')
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')
                ax[row, col].set_xlim(0, 6)
                # Remove the legend if it exists
                legend = ax[row, col].get_legend()
                if legend is not None:
                    legend.remove()
            else:
                ax[row, col].text(0.5, 0.5,
                                  f"Data not found for\n{selected_group} - {selected_condition}",
                                  horizontalalignment='center', verticalalignment='center')
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Save as SVG file
    svg_path = os.path.join(directory_path, f"behavior_durations_{project_name}.svg")
    fig.savefig(svg_path, dpi=600, bbox_inches='tight')

    # --- Additional Analysis: Average, Total, and Std Durations Per File ---
    all_file_durations = []
    for row in range(len(selected_groups)):
        for col in range(len(selected_conditions)):
            selected_group = selected_groups[row]
            selected_condition = selected_conditions[col]
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())
                for file_name in file_keys:
                    file_durations = get_duration_bouts(behaviors[selected_group][selected_condition][file_name],
                                                        behavior_names, framerate)
                    total_durations = [np.nansum(d) for d in file_durations]
                    mean_durations = [np.nanmean(d) if not np.isnan(np.nanmean(d)) else 0 for d in file_durations]
                    std_durations = [np.nanstd(d) if not np.isnan(np.nanstd(d)) else 0 for d in file_durations]

                    record = {
                        'group': selected_group,
                        'condition': selected_condition,
                        'file_name': file_name
                    }
                    for i, bname in enumerate(behavior_names):
                        record[f'{bname}_total_duration_s'] = total_durations[i]
                        record[f'{bname}_average_duration_s'] = mean_durations[i]
                        record[f'{bname}_std_duration_s'] = std_durations[i]
                    all_file_durations.append(record)

    all_file_durations_df = pd.DataFrame(all_file_durations)
    output_csv_path = os.path.join(directory_path, f"total_average_std_durations_per_file_{project_name}.csv")
    all_file_durations_df.to_csv(output_csv_path, index=False)

    return fig