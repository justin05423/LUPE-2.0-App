import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors  # Assumes meta.py defines these

def behavior_bout_counts(project_name, selected_groups, selected_conditions):
    """
    Generate a figure showing the bout counts for each behavior as a horizontal bar chart
    for each selected group and condition. For each group-condition combination, the function
    calculates the mean and standard deviation of bout counts and saves the data as CSV.

    Additionally, it performs a raw frequency analysis of behavioral bouts and saves CSVs:
      - One CSV per group/condition (raw bout counts per file)
      - One combined CSV aggregating all bout counts.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure containing the bar charts.
    """

    base_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    behaviors = load_behaviors(behaviors_file)

    directory_path = os.path.join(base_dir, "figures", "behavior_instance-counts")
    os.makedirs(directory_path, exist_ok=True)

    # Helper function: compute number of bouts for each behavior from a prediction vector.
    def get_num_bouts(predict, behavior_classes):
        bout_counts = []
        # Find indices where the predicted label changes
        bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
        bout_start_label = predict[bout_start_idx]
        for b, _ in enumerate(behavior_classes):
            idx_b = np.where(bout_start_label == int(b))[0]
            if len(idx_b) > 0:
                bout_counts.append(len(idx_b))
            else:
                bout_counts.append(np.nan)
        return bout_counts

    ### Part 1: Main Analysis – Horizontal Bar Charts ###
    rows = len(selected_groups)
    cols = len(selected_conditions)
    fig, ax = plt.subplots(rows, cols, figsize=(11, 20), sharex=False, sharey=True)

    if rows == 1 and cols == 1:
        ax = np.array([[ax]])
    elif rows == 1:
        ax = np.array([ax])
    elif cols == 1:
        ax = np.array([[a] for a in ax])

    for row in range(rows):
        for col in range(cols):
            selected_group = selected_groups[row]
            selected_condition = selected_conditions[col]

            bout_counts = []
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())

                for file_name in file_keys:
                    counts = get_num_bouts(behaviors[selected_group][selected_condition][file_name], behavior_names)
                    bout_counts.append(counts)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bout_mean = np.nanmean(bout_counts, axis=0)
                    bout_std = np.nanstd(bout_counts, axis=0)

                behavior_instance_dict = {
                    'mean_counts': bout_mean,
                    'std_counts': bout_std,
                    'labels': behavior_names,
                    'colors': behavior_colors,
                }
                behavior_instance_df = pd.DataFrame(behavior_instance_dict)

                csv_filename = os.path.join(
                    directory_path,
                    f"behavior_instance_counts_{selected_group}-{selected_condition}.csv"
                )
                behavior_instance_df.to_csv(csv_filename, index=False)

                behavior_instance_df.plot.barh(
                    y='mean_counts',
                    x='labels',
                    xerr='std_counts',
                    color=behavior_colors,
                    legend=False,
                    ax=ax[row, col],
                    zorder=3
                )
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')
                ax[row, col].grid(True, zorder=0)
                for spine in ax[row, col].spines.values():
                    spine.set_color('#D3D3D3')
                if row == rows - 1:
                    ax[row, col].set_xlabel('Instance counts / 30min')
            else:
                ax[row, col].text(
                    0.5, 0.5,
                    f"Data not found for\n{selected_group} - {selected_condition}",
                    horizontalalignment='center',
                    verticalalignment='center'
                )
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_svg = os.path.join(directory_path, "behavior_counts.svg")
    fig.savefig(save_path_svg, dpi=600, bbox_inches='tight')

    ### Part 2: Additional Analysis – Raw Frequency CSVs ###
    raw_directory_path = os.path.join(directory_path, "behavior_instance-counts_raw")
    os.makedirs(raw_directory_path, exist_ok=True)

    all_data = []
    for selected_group in selected_groups:
        for selected_condition in selected_conditions:
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())
                bout_counts = []
                for file_name in file_keys:
                    per_file_bout_counts = get_num_bouts(behaviors[selected_group][selected_condition][file_name],
                                                         behavior_names)
                    bout_counts.append(per_file_bout_counts)
                    for behavior, count in zip(behavior_names, per_file_bout_counts):
                        all_data.append({
                            'Group': selected_group,
                            'Condition': selected_condition,
                            'File': file_name,
                            'Behavior': behavior,
                            'Count': count
                        })

                raw_bout_counts_df = pd.DataFrame(bout_counts, columns=behavior_names, index=file_keys)
                raw_csv_filename = os.path.join(
                    raw_directory_path,
                    f"behavior_instance_counts_raw_{selected_group}-{selected_condition}.csv"
                )
                raw_bout_counts_df.to_csv(raw_csv_filename)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _ = np.nanmean(bout_counts, axis=0)
                    _ = np.nanstd(bout_counts, axis=0)

    all_data_df = pd.DataFrame(all_data)
    all_data_csv = os.path.join(raw_directory_path, "behavior_instance_counts_raw_all.csv")
    all_data_df.to_csv(all_data_csv, index=False)

    return fig
