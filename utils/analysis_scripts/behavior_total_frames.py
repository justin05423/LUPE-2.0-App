# utils/analysis_scripts/behavior_total_frames.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors


def behavior_total_frames(project_name, selected_groups, selected_conditions):
    """
    Create a pie chart showing the total number of frames per behavior for each
    selected group and condition. For each group-condition combination, the function:
      - Aggregates the behavior predictions from all files.
      - Saves a CSV file summarizing the counts.
      - Creates a pie chart subplot.
    Finally, the overall figure is saved as an SVG and returned.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure containing the pie charts.
    """
    # Use the app's base directory
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    behaviors = load_behaviors(behaviors_file)

    # Directory to save CSV files and the overall figure.
    directory_path = os.path.join(base_dir, "figures", "behavior_total-frames")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Determine grid dimensions based on selected groups and conditions.
    rows = len(selected_groups)
    cols = len(selected_conditions)
    fig, ax = plt.subplots(rows, cols, figsize=(10, 11))

    # Ensure ax is always a 2D array for uniform indexing.
    if rows == 1 and cols == 1:
        ax = np.array([[ax]])
    elif rows == 1:
        ax = np.array([ax])
    elif cols == 1:
        ax = np.array([[a] for a in ax])

    # Loop over each group and condition.
    for row in range(rows):
        # (Optional) For rows > 4, you might want to limit the number of columns.
        current_cols = cols
        if row > 4 and cols > 1:
            for extra in range(1, cols):
                fig.delaxes(ax[row, extra])
            current_cols = 1

        for col in range(current_cols):
            selected_group = selected_groups[row]
            selected_condition = selected_conditions[col]

            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())

                # Aggregate the behavior predictions for all files.
                # 'condition' is repeated for each frame, and 'behavior' is a horizontal stack of predictions.
                predict_dict = {
                    'condition': np.repeat(
                        selected_condition,
                        len(np.hstack([
                            behaviors[selected_group][selected_condition][file_name]
                            for file_name in file_keys
                        ]))
                    ),
                    'behavior': np.hstack([
                        behaviors[selected_group][selected_condition][file_name]
                        for file_name in file_keys
                    ])
                }
                df_raw = pd.DataFrame(data=predict_dict)

                # Count the total frames per behavior without sorting.
                vc = df_raw['behavior'].value_counts(sort=False)
                labels_indices = vc.index
                values = vc.values

                # Create a summary DataFrame.
                df = pd.DataFrame()
                behavior_labels = [behavior_names[int(l)] for l in labels_indices]
                df["values"] = values
                df["labels"] = behavior_labels
                df["colors"] = df["labels"].apply(lambda x: behavior_colors[behavior_names.index(x)])

                # Save the summary CSV for this group-condition.
                csv_filename = os.path.join(
                    directory_path,
                    f"behavior_total_frames_{project_name}_{selected_group}-{selected_condition}.csv"
                )
                df.to_csv(csv_filename, index=False)

                # Create the pie chart.
                ax[row, col].pie(
                    df['values'],
                    colors=df['colors'],
                    labels=df['labels'],
                    autopct='%1.1f%%',
                    pctdistance=0.85
                )
                # Draw a white circle at the center to give a donut look.
                centre_circle = plt.Circle((0, 0), 0.50, fc='white')
                ax[row, col].add_artist(centre_circle)
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')
            else:
                ax[row, col].text(
                    0.5, 0.5,
                    f"Data not found for\n{selected_group} - {selected_condition}",
                    horizontalalignment='center',
                    verticalalignment='center'
                )
                ax[row, col].set_title(f'{selected_group} - {selected_condition}')

    # Save the overall figure as an SVG.
    svg_filename = os.path.join(directory_path, f"behavior_total-frames_{project_name}.svg")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(svg_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return fig