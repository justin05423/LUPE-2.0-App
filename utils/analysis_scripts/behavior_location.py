# utils/analysis_scripts/behavior_location.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from utils.classification import load_behaviors, load_data
from utils.meta import behavior_names, behavior_colors  # Make sure these are defined in meta.py


def behavior_location(project_name, selected_groups, selected_conditions):
    """
    Generate figures showing the arena location of a specific behavior performed,
    with one figure per behavior. Each figure contains subplots for each combination
    of selected groups and conditions.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.

    Returns:
        figs (list): A list of matplotlib Figure objects (one per behavior).
    """
    # Update file paths to use the app's base directory
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    poses_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")

    # Load behaviors and poses
    behaviors = load_behaviors(behaviors_file)
    poses = load_data(poses_file)

    # Define the directory to save figures
    directory_path = os.path.join(base_dir, "figures", "behavior_location")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Parameters for plotting
    bodypart_idx = 38  # tail-base as position indicator
    center = (768 / 2, 770 / 2)
    radius = 768 / 2 + 20
    h = '00FEFF'  # cyan-like color for the circle

    rows = len(selected_groups)
    cols = len(selected_conditions)

    figs = []  # list to hold the figures per behavior

    # Loop over each behavior
    for b, behav_name in enumerate(behavior_names):
        count = 0
        # Create a figure with a black background
        fig = plt.figure(facecolor='#000000', figsize=(10, 11))

        # Loop over each group and condition combination
        for row in range(rows):
            for col in range(cols):
                # Create subplot for the current group-condition combination
                ax = fig.add_subplot(rows, cols, count + 1)
                ax.set_facecolor(None)
                selected_group = selected_groups[row]
                selected_condition = selected_conditions[col]

                # Convert the hex string to an RGB tuple
                rgb_val = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
                # Create a circle patch to indicate arena border
                circle = Circle(center, radius, color=rgb_val, linewidth=3, fill=False)
                hist2d_all = []
                # Create a colormap from black to the behavior color
                colors = ['#000000', behavior_colors[b]]
                cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
                # (Optional) Preallocate a dummy array (not used further)
                heatmaps = np.empty((38, 38))

                if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                    file_keys = list(behaviors[selected_group][selected_condition].keys())

                    # Loop over each file to compute a heatmap of positions when the behavior occurred
                    for file_name in file_keys:
                        idx_b = np.where(behaviors[selected_group][selected_condition][file_name] == b)[0]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            heatmap, xedges, yedges = np.histogram2d(
                                poses[selected_group][selected_condition][file_name][idx_b, bodypart_idx],
                                poses[selected_group][selected_condition][file_name][idx_b, bodypart_idx + 1],
                                bins=[np.arange(0, 768, 20), np.arange(0, 770, 20)],
                                density=True)
                        heatmap[heatmap == 0] = np.nan
                        hist2d_all.append(heatmap)

                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Plot the mean heatmap (transposed for correct orientation)
                        ax.imshow(np.nanmean(hist2d_all, axis=0).T,
                                  extent=extent, origin='lower', cmap=cm)

                # Draw a legend indicating the behavior name
                patches = [mpatches.Patch(color=behavior_colors[b], label=behav_name)]
                lgd = ax.legend(handles=patches, facecolor="#000000", frameon=False, prop={"size": 11},
                                ncol=1, bbox_to_anchor=(0.9, 0.9), loc='lower center', edgecolor='w')
                for text in lgd.get_texts():
                    text.set_color("#FFFFFF")
                # Add the circle patch to the axis
                ax.add_patch(circle)
                ax.set_aspect('equal')
                ax.invert_yaxis()  # invert y-axis for proper orientation
                plt.axis('off')
                plt.axis('equal')
                ax.set_title(f'{selected_group} - {selected_condition}', color='white', fontsize=10)
                count += 1
                # Optional: remove extra axes for specific positions if needed.
                if (row, col) in [(5, 1), (5, 2), (6, 1), (6, 2)]:
                    fig.delaxes(ax)

        # Save the figure for the current behavior
        save_path = os.path.join(directory_path, f"behavior_location_{behav_name}_{project_name}.svg")
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        # Append the figure to the list
        figs.append(fig)

    return figs