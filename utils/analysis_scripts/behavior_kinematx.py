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
    Calculate and plot the average displacement of a selected bodypart across behaviors,
    saving one CSV per condition and returning a single figure with one subplot per condition.

    Parameters:
        project_name (str): The project name.
        selected_group (str): The group name to analyze (only one).
        selected_conditions (list): A list of condition names to analyze.
        bp_selects (str): The bodypart of interest (e.g., 'l_hindpaw').

    """
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    model_path = os.path.join("model", "model.pkl")
    data_path = os.path.join(base_dir, f"raw_data_{project_name}.pkl")
    features_path = os.path.join(base_dir, f"binned_features_{project_name}.pkl")
    behaviors_path = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    model = load_model(model_path)
    poses = load_data(data_path)
    features = load_features(features_path)
    behaviors = load_behaviors(behaviors_path)

    figure_dir = os.path.join(base_dir, "figures", "behavior_kinematx")
    bp_dir = os.path.join(figure_dir, bp_selects)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    if not os.path.exists(bp_dir):
        os.makedirs(bp_dir)

    perfile_dir = os.path.join(bp_dir, "per_file_avg_displacement")
    if not os.path.exists(perfile_dir):
        os.makedirs(perfile_dir)

    movement_by_condition = {}
    bin_centers_by_condition = {}

    for selected_condition in selected_conditions:
        try:
            bodypart = keypoints.index(bp_selects)
        except ValueError:
            raise ValueError(f"Bodypart '{bp_selects}' not found in keypoints.")

        all_files_bout_disp = {}
        files_dict = behaviors[selected_group][selected_condition]
        for file_key in files_dict.keys():
            _, _, _, bout_disp_dict, _, _ = get_avg_kinematics(
                files_dict[file_key],
                poses[selected_group][selected_condition][file_key],
                bodypart,
                framerate=60
            )

            for beh in bout_disp_dict:
                bout_disp_dict[beh] = np.asarray(bout_disp_dict[beh])

            all_files_bout_disp[file_key] = bout_disp_dict

            perfile_avg_rows = []
            for beh in behavior_names:
                arr = bout_disp_dict.get(beh, np.array([]))
                avg_disp = float(np.mean(arr)) if (arr.ndim == 1 and arr.size > 0) else 0.0
                perfile_avg_rows.append({
                    'File': file_key,
                    'Behavior': beh,
                    'AvgDisplacement': avg_disp
                })

            df_perfile_avg = pd.DataFrame(perfile_avg_rows)
            perfile_avg_path = os.path.join(
                perfile_dir,
                f"{selected_group}_{selected_condition}_{file_key}_avg_displacement.csv"
            )
            df_perfile_avg.to_csv(perfile_avg_path, index=False)
            print(f"Saved per-file average displacement: {perfile_avg_path}")

        behavioral_sums = {}
        for beh in behavior_names:
            concatenated = []
            for file_key, disp_dict in all_files_bout_disp.items():
                arr = disp_dict.get(beh, np.array([]))
                if arr.ndim == 1 and arr.size > 0:
                    concatenated.append(arr)
            behavioral_sums[beh] = np.hstack(concatenated) if concatenated else np.array([])

        desc_rows = []
        for beh, arr in behavioral_sums.items():
            if arr.ndim == 1 and arr.size > 0:
                mean_disp = float(np.mean(arr))
                median_disp = float(np.median(arr))
                std_disp = float(np.std(arr))
                count_n = int(arr.size)
            else:
                mean_disp = median_disp = std_disp = 0.0
                count_n = 0
            desc_rows.append({
                'Condition': selected_condition,
                'Behavior': beh,
                'MeanDisplacement': mean_disp,
                'MedianDisplacement': median_disp,
                'StdDisplacement': std_disp,
                'Count': count_n
            })
        df_desc = pd.DataFrame(desc_rows)
        desc_csv_path = os.path.join(
            bp_dir,
            f"behavior_kinematx_descriptive_stats_{selected_group}_{selected_condition}_{bp_selects}.csv"
        )
        df_desc.to_csv(desc_csv_path, index=False)
        print(f"Saved descriptive stats CSV: {desc_csv_path}")

        global_max_perb = [
            np.percentile(arr, 95)
            for arr in behavioral_sums.values()
            if arr.ndim == 1 and arr.size > 10
        ]
        max_all = np.max(global_max_perb) if global_max_perb else 1.0

        movement_n_bins = 10
        global_bin_edges = np.linspace(0, max_all, movement_n_bins + 1)
        global_bin_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
        bin_centers_by_condition[selected_condition] = global_bin_centers

        pre_alloc_movement = np.zeros((len(behavior_names), movement_n_bins))
        for b, beh in enumerate(behavior_names):
            arr = behavioral_sums.get(beh, np.array([]))
            if arr.ndim != 1 or arr.size == 0:
                counts = np.zeros(movement_n_bins)
            else:
                df_temp = pd.DataFrame({'bp_movement': arr})
                cats = pd.cut(
                    df_temp['bp_movement'],
                    bins=global_bin_edges,
                    labels=False,
                    include_lowest=True
                )
                counts, _ = np.histogram(cats, bins=np.arange(-0.5, movement_n_bins + 0.5, 1))
            pre_alloc_movement[b, :] = counts

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_sums = pre_alloc_movement.sum(axis=1)
            movement_by_behav = pre_alloc_movement / row_sums[:, np.newaxis]

        movement_by_condition[selected_condition] = movement_by_behav

    for condition in selected_conditions:
        csv_rows = []
        centers = bin_centers_by_condition[condition]
        mov_data = movement_by_condition[condition]
        for beh_idx, beh in enumerate(behavior_names):
            for bin_idx, prob in enumerate(mov_data[beh_idx]):
                csv_rows.append({
                    'Condition': condition,
                    'Behavior': beh,
                    'Bin': bin_idx,
                    'BinCenter': float(centers[bin_idx]),
                    'Probability': float(prob)
                })

        df_condition = pd.DataFrame(csv_rows)
        condition_csv_path = os.path.join(
            bp_dir,
            f"plot_avg_displacement_{selected_group}_{condition}_{bp_selects}.csv"
        )
        df_condition.to_csv(condition_csv_path, index=False)
        print(f"Saved pooled displacement CSV for '{condition}': {condition_csv_path}")

    n_conditions = len(selected_conditions)
    n_behaviors = len(behavior_names)
    n_bins = movement_n_bins  # == 10

    fig, axes = plt.subplots(
        nrows=n_conditions,
        ncols=1,
        figsize=(10, 4 * n_conditions),
        sharex=True
    )
    if n_conditions == 1:
        axes = [axes]

    for i, condition in enumerate(selected_conditions):
        ax = axes[i]
        heatmap = ax.imshow(
            movement_by_condition[condition],
            aspect='auto',
            cmap='magma'
        )
        ax.set_title(f"{selected_group} – {condition}")
        ax.set_ylabel("Behaviors")
        ax.set_yticks(np.arange(n_behaviors))
        ax.set_yticklabels(behavior_names)
        ax.set_xlabel("Displacement Bin Center (pixels/frame)")

        # Use that condition’s bin centers for x‐tick labels
        centers = bin_centers_by_condition[condition]
        ax.set_xticks(np.arange(n_bins))
        ax.set_xticklabels([f"{c:.2f}" for c in centers], rotation=45, ha='right')

        fig.colorbar(heatmap, ax=ax, label="Probability")

    plt.suptitle(f"Average Displacement Distribution for Bodypart: {bp_selects}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(
        bp_dir,
        f"avg_displacement_conditions_{selected_group}_{bp_selects}.svg"
    )
    plt.savefig(fig_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"Saved combined heatmap figure: {fig_path}")

    # 9) Return that one Figure so that st.pyplot(fig) works
    return fig