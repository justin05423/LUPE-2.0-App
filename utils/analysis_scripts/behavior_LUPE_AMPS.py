import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.ndimage import label
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist
import seaborn as sns
import streamlit as st
import itertools


def behavior_LUPE_AMPS(project_name, selected_groups, selected_conditions):
    # Base directory for processed datasets
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    # Directory for saving figures/CSVs from the behavior LUPE-AMPS analysis
    directory_path = os.path.join(base_dir, "figures", "behavior_LUPE-AMPS")
    # Base directory for CSV files from prior analysis
    novel_data_base_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "frames")

    # Section 1: Preprocessing and File Analysis
    dtB = 60  # Original sampling rate (fps)
    dt = 20  # Desired sampling rate (fps)
    recLength = 30  # Recording length (minutes)
    original_length = 60 * dtB * recLength
    n_downsampled = int(original_length * (dt / dtB))

    novel_occ = []      # Fraction occupancy vector (6 states) per file.
    novel_nBouts = []   # Number of bouts per state (6 values per file).
    novel_boutDur = []  # Mean bout duration (seconds, 6 values per file).
    anOrder_novel = []  # CSV file names (animal IDs).
    group_labels = []   # Group for each file.
    condition_labels = []  # Condition for each file.
    total_files = 0

    for group in selected_groups:
        group_dir = os.path.join(novel_data_base_dir, group)
        print(f"Processing group directory: {group_dir}")
        for condition in selected_conditions:
            condition_dir = os.path.join(group_dir, condition)
            print(f"  Processing condition directory: {condition_dir}")
            csv_files = glob.glob(os.path.join(condition_dir, "*.csv"))
            csv_files.sort()
            print(f"    Found {len(csv_files)} CSV files.")
            for file in csv_files:
                baseFileName = os.path.basename(file)
                try:
                    data = pd.read_csv(file)
                except Exception as e:
                    print(f"      Skipping {baseFileName} due to error: {e}")
                    continue

                behav_col = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy()

                if behav_col.shape[0] < original_length:
                    behav = np.concatenate([
                        behav_col,
                        np.zeros(original_length - behav_col.shape[0], dtype=behav_col.dtype),
                    ])
                else:
                    behav = behav_col[:original_length]

                # Downsample (group every 3 frames: 60 -> 20 fps) using NumPy instead of scipy.stats.mode
                downsampled = []
                window_size = int(1 / (dt / dtB))  # equals 3
                for i in range(0, len(behav), window_size):
                    window = behav[i:i + window_size]
                    if window.size == 0:
                        break
                    window = window[~np.isnan(window)]
                    if window.size == 0:
                        continue
                    values, counts = np.unique(window.astype(int), return_counts=True)
                    value = values[np.argmax(counts)]
                    downsampled.append(value)
                downsampled = np.asarray(downsampled, dtype=int)

                if len(downsampled) < n_downsampled:
                    downsampled = np.pad(downsampled, (0, n_downsampled - len(downsampled)), mode="constant")
                elif len(downsampled) > n_downsampled:
                    downsampled = downsampled[:n_downsampled]

                occ_vector = [np.sum(downsampled == state) / len(downsampled) for state in range(6)]
                novel_occ.append(occ_vector)
                nBouts_vector = []
                boutDur_vector = []
                for state in range(6):
                    state_bin = (downsampled == state).astype(int)
                    labeled, num_features = label(state_bin)
                    nBouts_vector.append(num_features)
                    durations = [np.sum(labeled == i_feat) / dt for i_feat in range(1, num_features + 1)]
                    boutDur_vector.append(np.mean(durations) if durations else 0)
                novel_nBouts.append(nBouts_vector)
                novel_boutDur.append(boutDur_vector)
                anOrder_novel.append(baseFileName)
                group_labels.append(group)
                condition_labels.append(condition)
                total_files += 1

    print("Total files processed:", total_files)
    novel_occ = np.array(novel_occ)
    novel_nBouts = np.array(novel_nBouts)
    novel_boutDur = np.array(novel_boutDur)
    print("novel_occ shape:", novel_occ.shape)
    if novel_occ.size == 0 or novel_occ.ndim != 2:
        raise ValueError("novel_occ is empty or not 2D. Check your input data and file paths.")
    print("### Section 1 complete: Preprocessing and File Analysis")

    # Section 2: PCA Projection and Scatter Plot
    print("Starting Section 2: PCA Projection and Scatter Plot")
    pca_model_path = os.path.join("model", "LUPE-AMPS", "model_LUPE-AMPS.pkl")
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    novel_projection = pca.transform(novel_occ)
    novel_projection_2 = novel_projection[:, :2]
    pc1_novel = novel_projection_2[:, 0]
    pc2_novel = novel_projection_2[:, 1]
    output_dir_sec2 = os.path.join(base_dir, "figures", "behavior_LUPE-AMPS", "Section2")
    os.makedirs(output_dir_sec2, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Dynamically assign colors for any number of conditions
    prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    colors_for_conds = list(itertools.islice(itertools.cycle(prop_cycle_colors), len(selected_conditions)))
    condition_colors = dict(zip(selected_conditions, colors_for_conds))

    for condition in selected_conditions:
        idx = [i for i, c in enumerate(condition_labels) if c == condition]
        if len(idx) == 0:
            continue
        plt.scatter(
            pc1_novel[idx], pc2_novel[idx],
            color=condition_colors.get(condition, 'gray'),
            label=condition, s=50
        )
    plt.xlabel("PC1 (Generalized Behavior Scale)")
    plt.ylabel("PC2 (Pain Behavior Scale)")
    plt.title("Novel Data PCA Projection by Condition")
    plt.legend()
    plt.tight_layout()
    novel_fig_png = os.path.join(output_dir_sec2, f"{project_name}_novel_data_projection.png")
    plt.savefig(novel_fig_png)
    novel_fig_svg = os.path.join(output_dir_sec2, f"{project_name}_novel_data_projection.svg")
    plt.savefig(novel_fig_svg, format="svg")
    fig_scatter = plt.gcf()
    st.markdown("#### PCA Projection Scatter Plot")
    st.pyplot(fig_scatter)
    plt.clf()
    df_novel_projection = pd.DataFrame(novel_projection_2, columns=["PC1", "PC2"])
    df_novel_projection["Animal"] = anOrder_novel
    df_novel_projection["Condition"] = condition_labels
    csv_proj_path = os.path.join(output_dir_sec2, f"{project_name}_novel_data_projection.csv")
    df_novel_projection.to_csv(csv_proj_path, index=False)
    print("### Section 2 complete: PCA Projection and Scatter Plot")

    # Section 3: Multi-graph Line Plots for Metrics
    print("Starting Section 3: Multi-graph Line Plots for Metrics")
    output_dir_sec3 = os.path.join(base_dir, "figures", "behavior_LUPE-AMPS", "Section3")
    os.makedirs(output_dir_sec3, exist_ok=True)
    states = [f"State {i + 1}" for i in range(6)]
    # Human-readable labels per state category
    category_map = {
        1: "Pain Suppressed",
        2: "Non-pain State",
        3: "Pain Enhanced",
        4: "Pain Enhanced",
        5: "Non-pain State",
        6: "Non-pain State",
    }
    states_display = [f"State {i} (" + category_map[i] + ")" for i in range(1, 7)]

    # ----- Fraction Occupancy Line Graph -----
    df_occ_animals = pd.DataFrame(novel_occ, columns=states)
    df_occ_animals["Condition"] = condition_labels
    df_occ_animals["Group"] = group_labels
    df_occ_animals["Animal"] = anOrder_novel
    fig_occ, ax_occ = plt.subplots(figsize=(10, 6))
    for group in selected_groups:
        for condition in selected_conditions:
            cond_data = df_occ_animals[(df_occ_animals["Group"] == group) & (df_occ_animals["Condition"] == condition)]
            if cond_data.empty:
                continue
            means = cond_data[states].mean()
            sem = cond_data[states].std(ddof=1) / np.sqrt(len(cond_data))
            xvals = np.arange(len(states))
            label_str = f"{group} - {condition}"
            ax_occ.plot(xvals, means, marker='o', label=label_str)
            ax_occ.fill_between(xvals, means - sem, means + sem, alpha=0.2)
    ax_occ.set_xticks(xvals)
    ax_occ.set_xticklabels(states_display, rotation=45, ha='right')
    ax_occ.set_xlabel("State")
    ax_occ.set_ylabel("Fraction Occupancy")
    ax_occ.set_title("Fraction Occupancy by Condition and State (Line Graph)")
    ax_occ.legend()
    plt.tight_layout()
    # Save Fraction Occupancy Line Graph as PNG and SVG
    fig_occ_png = os.path.join(output_dir_sec3, f"{project_name}_fraction_occupancy_line_graph.png")
    fig_occ_svg = os.path.join(output_dir_sec3, f"{project_name}_fraction_occupancy_line_graph.svg")
    fig_occ.savefig(fig_occ_png)
    fig_occ.savefig(fig_occ_svg)
    st.markdown("#### Fraction Occupancy Line Graph")
    st.pyplot(fig_occ)
    plt.clf()

    # ----- Number of Bouts Line Graph -----
    df_bouts_animals = pd.DataFrame(novel_nBouts, columns=states)
    df_bouts_animals["Condition"] = condition_labels
    df_bouts_animals["Group"] = group_labels
    df_bouts_animals["Animal"] = anOrder_novel
    fig_bouts, ax_bouts = plt.subplots(figsize=(10, 6))
    for group in selected_groups:
        for condition in selected_conditions:
            cond_data = df_bouts_animals[(df_bouts_animals["Group"] == group) & (df_bouts_animals["Condition"] == condition)]
            if cond_data.empty:
                continue
            means = cond_data[states].mean()
            sem = cond_data[states].std(ddof=1) / np.sqrt(len(cond_data))
            xvals = np.arange(len(states))
            label_str = f"{group} - {condition}"
            ax_bouts.plot(xvals, means, marker='o', label=label_str)
            ax_bouts.fill_between(xvals, means - sem, means + sem, alpha=0.2)
    ax_bouts.set_xticks(xvals)
    ax_bouts.set_xticklabels(states_display, rotation=45, ha='right')
    ax_bouts.set_xlabel("State")
    ax_bouts.set_ylabel("Number of Bouts")
    ax_bouts.set_title("Number of Bouts by Condition and State (Line Graph)")
    ax_bouts.legend()
    plt.tight_layout()
    # Save Number of Bouts Line Graph as PNG and SVG
    fig_bouts_png = os.path.join(output_dir_sec3, f"{project_name}_number_of_bouts_line_graph.png")
    fig_bouts_svg = os.path.join(output_dir_sec3, f"{project_name}_number_of_bouts_line_graph.svg")
    fig_bouts.savefig(fig_bouts_png)
    fig_bouts.savefig(fig_bouts_svg)
    st.markdown("#### Number of Bouts Line Graph")
    st.pyplot(fig_bouts)
    plt.clf()

    # ----- Bout Duration Line Graph -----
    df_boutdur_animals = pd.DataFrame(novel_boutDur, columns=states)
    df_boutdur_animals["Condition"] = condition_labels
    df_boutdur_animals["Group"] = group_labels
    df_boutdur_animals["Animal"] = anOrder_novel
    fig_boutdur, ax_boutdur = plt.subplots(figsize=(10, 6))
    for group in selected_groups:
        for condition in selected_conditions:
            cond_data = df_boutdur_animals[(df_boutdur_animals["Group"] == group) & (df_boutdur_animals["Condition"] == condition)]
            if cond_data.empty:
                continue
            means = cond_data[states].mean()
            sem = cond_data[states].std(ddof=1) / np.sqrt(len(cond_data))
            xvals = np.arange(len(states))
            label_str = f"{group} - {condition}"
            ax_boutdur.plot(xvals, means, marker='o', label=label_str)
            ax_boutdur.fill_between(xvals, means - sem, means + sem, alpha=0.2)
    ax_boutdur.set_xticks(xvals)
    ax_boutdur.set_xticklabels(states_display, rotation=45, ha='right')
    ax_boutdur.set_xlabel("State")
    ax_boutdur.set_ylabel("Bout Duration (s)")
    ax_boutdur.set_title("Bout Duration by Condition and State (Line Graph)")
    ax_boutdur.legend()
    plt.tight_layout()
    # Save Bout Duration Line Graph as PNG and SVG
    fig_boutdur_png = os.path.join(output_dir_sec3, f"{project_name}_bout_duration_line_graph.png")
    fig_boutdur_svg = os.path.join(output_dir_sec3, f"{project_name}_bout_duration_line_graph.svg")
    fig_boutdur.savefig(fig_boutdur_png)
    fig_boutdur.savefig(fig_boutdur_svg)
    st.markdown("#### Bout Duration Line Graph")
    st.pyplot(fig_boutdur)
    plt.clf()

    csv_occ_path = os.path.join(output_dir_sec3, f"{project_name}_novel_fraction_occupancy_by_condition.csv")
    csv_bouts_path = os.path.join(output_dir_sec3, f"{project_name}_novel_number_of_bouts_by_condition.csv")
    csv_boutdur_path = os.path.join(output_dir_sec3, f"{project_name}_novel_bout_duration_by_condition.csv")
    df_occ_animals.to_csv(csv_occ_path, index=False)
    df_bouts_animals.to_csv(csv_bouts_path, index=False)
    df_boutdur_animals.to_csv(csv_boutdur_path, index=False)
    print("### Section 3 complete: Multi-graph Line Plots for Metrics")

    # Section 4: Model Fit Analysis
    print("Starting Section 4: Model Fit Analysis")
    animal_list = []
    for group in selected_groups:
        for cond in selected_conditions:
            animal_list.append({"animal_id": f"{group}_{cond}", "group": group, "condition": cond})
    animal_info = pd.DataFrame(animal_list)
    num_animals = animal_info.shape[0]  # Total number of animals

    # PARAMETERS AND WINDOWING SETUP
    nSecs = 30       # window length (sec)
    toSlide = 10     # sliding step (sec)
    nBeh = 6         # number of behaviors
    dt = 20          # frames per sec
    recLength = 30   # recording length in minutes
    winSize = dt * nSecs             # e.g., 600 frames
    winSlide = dt * toSlide          # e.g., 200 frames
    total_frames = dt * recLength * 60  # e.g., 108000 frames

    wins = np.arange(1, total_frames + 1, winSlide)
    wins = wins[wins + winSize + 1 <= total_frames]  # exclude windows that extend beyond total_frames
    nWins = len(wins)

    phase = 0  # 0 = whole session; adjust if needed.
    winsP1 = np.arange(1, dt * 10 * 60 + 1, winSlide)
    winsP1 = winsP1[winsP1 + winSize + 1 <= dt * 10 * 60]
    nWinsP1 = len(winsP1)
    if phase == 1:
        times = np.arange(0, nWinsP1)
    elif phase == 2:
        times = np.arange(nWinsP1, nWins)
    else:
        times = np.arange(0, nWins)

    nVals_global = 100
    resampled_length = toSlide * dt * (nWins + 1)
    behState = np.zeros((resampled_length, num_animals, 9), dtype=int)

    # DEFINE INDEX SUBSETS (converted from MATLAB)
    subsetStill = np.array([0, 1, 2, 3, 4, 5, 6, 12, 18, 24, 30])
    subsetWalk  = np.array([1, 6, 7, 8, 9, 10, 11, 13, 19, 25, 31])
    subsetRear  = np.array([2, 8, 12, 13, 14, 15, 16, 17, 20, 26, 32])
    subsetGroom = np.array([3, 9, 15, 18, 19, 20, 21, 22, 23, 27, 33])
    subsetLick  = np.array([4, 10, 16, 22, 24, 25, 26, 27, 28, 29, 34])
    subsetRight = np.array([5, 11, 17, 23, 29, 30, 31, 32, 33, 34, 35])
    subsets = [subsetStill, subsetWalk, subsetRear, subsetGroom, subsetLick, subsetRight]
    group1 = np.arange(2, 7) - 1
    group2 = np.concatenate(([1], np.arange(3, 7))) - 1
    group3 = np.concatenate((np.arange(1, 3), np.arange(4, 7))) - 1
    group4 = np.concatenate((np.arange(1, 4), np.array([5, 6]))) - 1
    group5 = np.concatenate((np.arange(1, 5), [6])) - 1
    group6 = np.arange(1, 6) - 1
    subset2 = np.concatenate((group1, group2, group3, group4, group5, group6))

    # PREALLOCATE OUTCOME VARIABLES
    ditMeans = np.zeros((num_animals, 9, nVals_global))
    match = np.zeros((num_animals, 9, nVals_global))
    nFeatures_full = nBeh ** 2
    transUnfolded = np.random.rand(nWins, nFeatures_full, num_animals)
    k = nBeh
    c = np.random.rand(k, nFeatures_full)

    # MAIN LOOP: COMPUTE MODEL-FIT METRICS
    for l in range(1, 10):  # l = 1,...,9
        if l < 9:
            nVals = 1
        else:
            nVals = nVals_global
        for v in range(nVals):
            for a in range(num_animals):
                data_animal = transUnfolded[times, :, a].copy()  # shape: (len(times), nFeatures_full)
                centroids = c.copy()  # copy centroids
                if l == 1:
                    # Full model: no modifications.
                    pass
                elif l == 2:
                    data_animal = data_animal[:, subset2]
                    centroids = centroids[:, subset2]
                elif l == 9:
                    perm = np.random.permutation(nFeatures_full)
                    centroids = centroids[:, perm]
                else:
                    data_animal = np.delete(data_animal, subsets[l - 3], axis=1)
                    centroids = np.delete(centroids, subsets[l - 3], axis=1)
                distances = cdist(data_animal, centroids, metric='euclidean')
                d = np.min(distances, axis=1)
                idx_min = np.argmin(distances, axis=1)
                n_idx = len(idx_min)
                scaled_length = toSlide * dt * n_idx
                scaledIdx = np.zeros(scaled_length, dtype=int)
                for t in range(n_idx):
                    start_idx = toSlide * dt * t
                    end_idx = toSlide * dt * (t + 2)
                    end_idx = min(end_idx, scaled_length)
                    # Add 1 to mimic MATLAB 1-indexing
                    scaledIdx[start_idx:end_idx] = idx_min[t] + 1
                behState[:len(scaledIdx), a, l - 1] = scaledIdx
                ditMeans[a, l - 1, v] = np.mean(d)
                if l == 1:
                    full_state = scaledIdx.copy()
                else:
                    full_state = behState[:len(scaledIdx), a, 0]
                match[a, l - 1, v] = np.mean(full_state == scaledIdx)

    print("ditMeans shape:", ditMeans.shape)
    print("match shape:", match.shape)
    avg_dit = np.mean(ditMeans, axis=2)
    df_all = animal_info.copy()
    model_condition_names = ["Full Model", "No Self-Transition", "No Still", "No Walk",
                             "No Rear", "No Groom", "No Left Lick", "No Right Lick", "Shuffled"]
    for i, cond_name in enumerate(model_condition_names):
        df_all[cond_name] = avg_dit[:, i]
    output_dir_csv = os.path.join(base_dir, "figures", "behavior_LUPE-AMPS", "Section4")
    os.makedirs(output_dir_csv, exist_ok=True)
    csv_summary = os.path.join(output_dir_csv, "ditMeans_Summary_allConditions.csv")
    df_all.to_csv(csv_summary, index=False)
    print("Aggregated summary CSV saved at:", csv_summary)
    output_dir_sec4 = os.path.join(base_dir, "figures", "behavior_LUPE-AMPS", "Section4")
    os.makedirs(output_dir_sec4, exist_ok=True)
    overall_mean = np.mean(avg_dit, axis=0)
    overall_sem = np.std(avg_dit, axis=0, ddof=1) / np.sqrt(avg_dit.shape[0])
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_condition_names))
    ax1.plot(x, overall_mean, marker='o', linestyle='-', color='skyblue', label='All Animals')
    ax1.fill_between(x, overall_mean - overall_sem, overall_mean + overall_sem, color='skyblue', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_condition_names, rotation=45, ha='right')
    ax1.set_ylabel("Mean Euclidean Distance")
    ax1.set_title("Overall Model Fit (Average Across All Animals)")
    ax1.legend()
    plt.tight_layout()
    fig1_png = os.path.join(output_dir_sec4, "model_Section4_fit_line_overall.png")
    fig1_svg = os.path.join(output_dir_sec4, "model_Section4_fit_line_overall.svg")
    plt.savefig(fig1_png)
    plt.savefig(fig1_svg)
    st.markdown("#### Overall Model Fit (Average Across All Animals)")
    st.pyplot(fig1)
    plt.clf()
    print("Overall Model Fit line graph saved as PNG at:", fig1_png)
    print("Overall Model Fit line graph saved as SVG at:", fig1_svg)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_condition_names))
    for group in selected_groups:
        for cond in selected_conditions:
            indices = np.where((df_all['group'] == group) & (df_all['condition'] == cond))[0]
            if len(indices) == 0:
                continue
            sub_data = avg_dit[indices, :]
            group_mean = np.mean(sub_data, axis=0)
            group_sem = np.std(sub_data, axis=0, ddof=1) / np.sqrt(len(indices))
            label_str = f"{group}_{cond}"
            ax2.plot(x, group_mean, marker='o', linestyle='-', label=label_str)
            ax2.fill_between(x, group_mean - group_sem, group_mean + group_sem, alpha=0.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_condition_names, rotation=45, ha='right')
    ax2.set_ylabel("Mean Euclidean Distance")
    ax2.set_title("Model Fit by Experimental Condition and Group")
    ax2.legend()
    plt.tight_layout()
    fig2_png = os.path.join(output_dir_sec4, "model_Section4_fit_line_by_experimental_condition.png")
    fig2_svg = os.path.join(output_dir_sec4, "model_Section4_fit_line_by_experimental_condition.svg")
    plt.savefig(fig2_png)
    plt.savefig(fig2_svg)
    st.markdown("#### Model Fit by Experimental Condition and Group")
    st.pyplot(fig2)
    plt.clf()
    print("Group comparison line graph saved as PNG at:", fig2_png)
    print("Group comparison line graph saved as SVG at:", fig2_svg)
    print("### Section 4 complete: Model Fit Analysis")