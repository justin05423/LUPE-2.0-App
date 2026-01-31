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
    # Update file paths to use the app's base directory with cross-platform path handling
    base_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    poses_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")

    behaviors = load_behaviors(behaviors_file)
    poses = load_data(poses_file)

    directory_path = os.path.join(base_dir, "figures", "behavior_location")
    os.makedirs(directory_path, exist_ok=True)

    bodypart_idx = 38  # tail-base as position indicator
    center = (768 / 2, 770 / 2)
    radius = 768 / 2 + 20
    h = '00FEFF'  # cyan-like color for the circle

    rows = len(selected_groups)
    cols = len(selected_conditions)

    figs = []  # list to hold the figures per behavior

    for b, behav_name in enumerate(behavior_names):
        count = 0

        fig = plt.figure(facecolor='#000000', figsize=(10, rows * 2.5 + 1))
        fig.suptitle(behav_name, color=behavior_colors[b], fontsize=16, fontweight='bold', y=0.98)

        for row in range(rows):
            for col in range(cols):
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

                ax.add_patch(circle)
                ax.set_aspect('equal')
                ax.invert_yaxis()  # invert y-axis for proper orientation
                plt.axis('off')
                plt.axis('equal')
                ax.set_title(f'{selected_group} - {selected_condition}', color='white', fontsize=10)
                count += 1

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.subplots_adjust(top=0.88, hspace=0.3)

        save_path_svg = os.path.join(directory_path, f"behavior_location_{behav_name}.svg")
        fig.savefig(save_path_svg, dpi=600, bbox_inches='tight')

        figs.append(fig)

    return figs
