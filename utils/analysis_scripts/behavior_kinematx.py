# utils/analysis_scripts/behavior_kinematx.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import os
import sys
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

# Ensure the project root is in sys.path if needed
if not os.path.join(os.path.abspath(''), '../') in sys.path:
    sys.path.append(os.path.join(os.path.abspath(''), '../'))

from utils.classification import load_model, load_features, load_data, weighted_smoothing, load_behaviors
from utils.feature_utils import get_avg_kinematics
from utils.meta import keypoints, behavior_names  # 'keypoints' should be a list of bodypart names


def behavior_kinematx(project_name, selected_group, selected_conditions, bp_selects):
    """
    Calculate and plot the average displacement of a selected bodypart across behaviors.

    Parameters:
        project_name (str): The project name.
        selected_group (str): The group name to analyze (only one).
        selected_conditions (list): A list of condition names to analyze.
        bp_selects (str): The bodypart of interest (e.g., 'genitalia').

    Returns:
        fig (matplotlib.figure.Figure): The generated heatmap figure.
    """
    # Define paths using the app's directory structure
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    model_path = os.path.join("model", "model.pkl")
    data_path = os.path.join(base_dir, f"raw_data_{project_name}.pkl")
    features_path = os.path.join(base_dir, f"binned_features_{project_name}.pkl")
    behaviors_path = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Load data
    model = load_model(model_path)
    poses = load_data(data_path)
    features = load_features(features_path)
    behaviors = load_behaviors(behaviors_path)

    # Define directory to save figures and CSV files
    figure_dir = os.path.join(base_dir, "figures", "behavior_kinematx")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    movement_by_condition = {}

    # For each condition, process all files in the selected group
    for selected_condition in selected_conditions:
        # Determine the index of the selected bodypart from keypoints
        try:
            bodypart = keypoints.index(bp_selects)
        except ValueError:
            raise ValueError(f"Bodypart '{bp_selects}' not found in keypoints.")

        bout_disp_all = []
        # Instead of assuming keys like 'file0', iterate over the actual keys
        files_dict = behaviors[selected_group][selected_condition]
        for file_key in files_dict.keys():
            # get_avg_kinematics returns multiple outputs; we need the 'bout_disp' output.
            # The expected output order: behavior, behavioral_start_time, behavior_duration, bout_disp, bout_duration, bout_avg_speed
            outputs = get_avg_kinematics(files_dict[file_key],
                                         poses[selected_group][selected_condition][file_key],
                                         bodypart, framerate=60)
            bout_disp = outputs[3]
            bout_disp_all.append(bout_disp)

        # At this point, bout_disp_all is a list (one per file) of dictionaries keyed by behavior names.
        # Aggregate displacement data for each behavior:
        behavioral_sums = {}
        for behav in behavior_names:
            # For each file, if the dictionary contains data for this behavior, include it.
            data_list = [file_dict[behav] for file_dict in bout_disp_all if behav in file_dict]
            if data_list:
                behavioral_sums[behav] = np.hstack(data_list)
            else:
                behavioral_sums[behav] = np.array([])

        # Determine the maximum displacement among behaviors with enough data
        max_perb = []
        for beh in behavioral_sums:
            if len(behavioral_sums[beh]) > 10:
                max_perb.append(np.percentile(behavioral_sums[beh], 95))
        max_all = np.max(max_perb) if max_perb else 1  # fallback to 1 if no data

        # Define the number of movement bins
        movement_n_bins = 10
        pre_alloc_movement = np.zeros((len(behavior_names), movement_n_bins))
        label_encoder = LabelEncoder()

        # Create histogram of displacements for each behavior
        for b, behav in enumerate(behavior_names):
            df_bp = pd.DataFrame(data=behavioral_sums[behav], columns=['bp_movement'])
            n_bins = np.linspace(0, max_all, movement_n_bins)
            # Use pd.cut to bin the data
            cat, bins = pd.cut(df_bp['bp_movement'], n_bins, retbins=True)
            y = label_encoder.fit_transform(cat)
            pre_alloc_movement[b, :] = np.histogram(y, bins=np.arange(0, movement_n_bins + 1))[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_sums = pre_alloc_movement.sum(axis=1)
            movement_by_behav = pre_alloc_movement / row_sums[:, np.newaxis]

        movement_by_condition[selected_condition] = movement_by_behav

    # Save the aggregated movement data to a CSV file
    csv_data = []
    for condition, movement_data in movement_by_condition.items():
        for beh_index, behavior in enumerate(behavior_names):
            for bin_index, value in enumerate(movement_data[beh_index]):
                csv_data.append({
                    'Condition': condition,
                    'Behavior': behavior,
                    'Bin': bin_index,
                    'Probability': value
                })
    df_csv = pd.DataFrame(csv_data)
    csv_path = os.path.join(figure_dir, f"avg_displacement_conditions_{selected_group}_{bp_selects}.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # Reload CSV and fill missing values with 0.0
    df_csv = pd.read_csv(csv_path)
    df_csv['Probability'].fillna(0.0, inplace=True)
    df_csv.to_csv(csv_path, index=False)

    # Reorganize data for heatmap plotting
    for condition in selected_conditions:
        movement_by_condition[condition] = np.zeros((len(behavior_names), movement_n_bins))
        for beh_index, behavior in enumerate(behavior_names):
            for bin_index in range(movement_n_bins):
                value = df_csv[(df_csv['Condition'] == condition) &
                               (df_csv['Behavior'] == behavior) &
                               (df_csv['Bin'] == bin_index)]['Probability']
                if not value.empty:
                    movement_by_condition[condition][beh_index, bin_index] = value.values[0]

    # Plot: one subplot per condition (vertical stack)
    n_conditions = len(selected_conditions)
    if n_conditions > 1:
        fig, axes = plt.subplots(nrows=n_conditions, ncols=1, figsize=(10, 10), sharex=True)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes = [axes]

    for i, condition in enumerate(selected_conditions):
        ax = axes[i]
        heatmap = ax.imshow(movement_by_condition[condition], aspect='auto', cmap="viridis")
        ax.set_title(f'{selected_group} - {condition}')
        ax.set_xlabel('Avg Displacement (pixels/frame)')
        ax.set_ylabel('Behaviors')
        ax.set_xticks(np.arange(movement_n_bins))
        ax.set_yticks(np.arange(len(behavior_names)))
        ax.set_yticklabels(behavior_names)
        fig.colorbar(heatmap, ax=ax, label='Probability of Displacement')

    plt.suptitle(f'Average Displacement for Bodypart - {bp_selects}', fontsize=14)
    plt.tight_layout()

    # Save the figure as an SVG file
    fig_path = os.path.join(figure_dir, f"avg_displacement_conditions_{selected_group}_{bp_selects}.svg")
    plt.savefig(fig_path, format='svg', dpi=600, bbox_inches='tight')
    plt.show()

    return fig