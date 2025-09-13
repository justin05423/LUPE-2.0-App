# utils/analysis_scripts/behavior_binned_ratio.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.classification import load_behaviors

def behavior_binned_ratio_timeline(project_name, selected_groups, selected_conditions, num_min):
    """
    Generate a binned-ratio timeline plot and save the results as CSV and SVG files.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.
        num_min (int): Number of minutes per time bin.

    Returns:
        figs (list): A list of matplotlib Figure objects (one for each group).
    """
    # Define the base directory
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Load behaviors
    behaviors = load_behaviors(behaviors_file)

    # Parameters
    time_bin_size = 60 * 60 * num_min

    # Define the directory path for saving figures and CSVs
    directory_path = os.path.join(base_dir, "figures/behavior_binned-ratio-timeline")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Check if behaviors is None or empty
    if behaviors is None or not behaviors:
        raise ValueError("Failed to load behaviors or empty dataset.")

    # Behavior names and colors
    behavior_names = ['still', 'walking', 'rearing', 'grooming', 'licking hindpaw L', 'licking hindpaw R']
    behavior_colors = ['crimson', 'darkcyan', 'goldenrod', 'royalblue', 'rebeccapurple', 'mediumorchid']

    # List to store figures
    figs = []

    # Generate plots and save data
    for selected_group in selected_groups:
        fig, axes = plt.subplots(nrows=len(selected_conditions), figsize=(8, 2.5 * len(selected_conditions) + 1))
        fig.suptitle(f'Group: {selected_group}', fontsize=16)

        for idx, selected_condition in enumerate(selected_conditions):
            ax = axes[idx] if len(selected_conditions) > 1 else axes  # Handle single subplot case

            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())
                n_bins = len(behaviors[selected_group][selected_condition][file_keys[0]]) // time_bin_size

                behavior_ratios_files = {key: np.nan for key in file_keys}

                for file_name in file_keys:
                    binned_behaviors = []
                    for bin_n in range(int(n_bins)):
                        behavior_ratios = {key: 0 for key in range(len(behavior_names))}
                        values, counts = np.unique(behaviors[selected_group][selected_condition][file_name][time_bin_size * bin_n:time_bin_size * (bin_n + 1)], return_counts=True)
                        for i, value in enumerate(values):
                            behavior_ratios[value] = counts[i] / sum(counts)
                        binned_behaviors.append(behavior_ratios)
                    behavior_ratios_files[file_name] = binned_behaviors

                data_to_save = {'Time_bin': np.arange(int(n_bins))}

                for b in range(len(behavior_names)):
                    y_files = []
                    for file_name in file_keys:
                        y_files.append(np.hstack([behavior_ratios_files[file_name][bin][b] for bin in range(len(behavior_ratios_files[file_name]))]))
                    y = np.mean(y_files, axis=0)
                    x = np.arange(int(n_bins))
                    y_sem = np.std(y_files, axis=0) / np.sqrt(len(behavior_ratios_files))

                    ax.plot(x, y, color=behavior_colors[b], label=behavior_names[b])
                    ax.fill_between(x, y - y_sem, y + y_sem, color=behavior_colors[b], alpha=0.2)

                    data_to_save[behavior_names[b]] = y
                    data_to_save[f'{behavior_names[b]}_SEM'] = y_sem

                ax.set_title(f'{selected_condition}')
                ax.set_xlabel(f'Time bin = {num_min} min')
                ax.set_ylabel('Percent')
                ax.spines[['top', 'right']].set_visible(False)
                df = pd.DataFrame(data_to_save)

                # Save DataFrame to CSV (one file per group-condition combination)
                output_filename = os.path.join(directory_path, f"behavior_binned-ratio-timeline__{project_name}_{selected_group}-{selected_condition}.csv")
                df.to_csv(output_filename, index=False)
                print(f"Data saved to {output_filename}.")

            else:
                raise ValueError(f"Selected group '{selected_group}' or condition '{selected_condition}' not found in the dataset.")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title

        # Create a single legend below all subplots
        handles, labels = [], []
        for color, name in zip(behavior_colors, behavior_names):
            handles.append(plt.Line2D([0], [0], color=color, lw=4))
            labels.append(name)

        fig.legend(handles, labels, loc='lower center', ncol=len(behavior_names), bbox_to_anchor=(0.5, -0.05))

        # Save the figure (one file per group)
        save_path = os.path.join(directory_path, f"behavior_binned-ratio-timeline_{project_name}_{selected_group}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')

        # Add the figure to the list
        figs.append(fig)

    return figs