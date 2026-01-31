import os
import pandas as pd

def behavior_timepoint_comparison(project_name, selected_groups, selected_conditions, time_ranges):
    """
    Compare behavior metrics across different time ranges and generate cohort summaries.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.
        time_ranges (list): List of tuples representing time ranges in seconds (e.g., [(0, 600), (600, 1800)]).
    """
    if len(time_ranges) < 2:
        raise ValueError("At least two time ranges are required for comparison.")

    behavior_labels = ['still', 'walking', 'rearing', 'grooming', 'licking hindpaw L', 'licking hindpaw R']

    time_labels = [f"{start // 60}-{end // 60} min" for start, end in time_ranges]

    input_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name, "figures", "behaviors_csv_raw-classification", "seconds")

    analysis_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name, "figures", "behavior_timepoint_comparison")
    os.makedirs(analysis_dir, exist_ok=True)

    def calculate_behavior_metrics(data, frame_rate=60):
        metrics = {}
        unique_behaviors = data['behavior'].unique()

        for behavior in unique_behaviors:
            behavior_data = data[data['behavior'] == behavior]

            fraction_time = len(behavior_data) / len(data)

            bout_starts = (behavior_data.index.to_series().diff() > 1).cumsum()
            bouts = behavior_data.groupby(bout_starts)

            bouts_per_minute = len(bouts) / (len(data) / frame_rate / 60)

            mean_bout_duration = bouts.size().mean() / frame_rate

            metrics[behavior] = {
                'Fraction Time': fraction_time,
                'Bouts per Minute': bouts_per_minute,
                'Mean Bout Duration (s)': mean_bout_duration
            }
        return metrics

    for group in selected_groups:
        for condition in selected_conditions:
            group_cond_dir = os.path.join(input_dir, group, condition)
            if not os.path.isdir(group_cond_dir):
                print(f"No directory found for group '{group}' and condition '{condition}'")
                continue
            for file_name in os.listdir(group_cond_dir):
                if not file_name.endswith('.csv'):
                    continue
                file_path = os.path.join(group_cond_dir, file_name)
                df = pd.read_csv(file_path)
                max_time = df['time_seconds'].max()
                bins = [start for start, end in time_ranges] + [time_ranges[-1][1]]
                if max_time < bins[-1]:
                    print(f"Warning: Maximum time ({max_time}s) in {file_name} is less than the final bin end ({bins[-1]}s).")
                    bins[-1] = max_time
                try:
                    df['time_group'] = pd.cut(df['time_seconds'], bins=bins, labels=time_labels, right=False)
                except ValueError as e:
                    print(f"Error in pd.cut for file {file_name}: {e}")
                    continue
                all_metrics = []
                for time_group, group_data in df.groupby('time_group', observed=False):
                    if not group_data.empty:
                        metrics = calculate_behavior_metrics(group_data)
                        for behavior, behavior_metrics in metrics.items():
                            all_metrics.append({
                                'Group': group,
                                'Condition': condition,
                                'Time Group': time_group,
                                'Behavior': behavior,
                                'Behavior Label': behavior_labels[int(behavior)],
                                **behavior_metrics
                            })
                analysis_df = pd.DataFrame(all_metrics)
                analysis_file_name = f'analysis_{file_name}'
                analysis_file_path = os.path.join(analysis_dir, analysis_file_name)
                analysis_df.to_csv(analysis_file_path, index=False)
                print(f"Saved analysis for {file_name} to {analysis_file_path}")

    print('Behavior analysis completed for all files.')

    cohort_summary_dir = os.path.join(analysis_dir, "cohort_summaries")
    os.makedirs(cohort_summary_dir, exist_ok=True)

    def aggregate_cohort_data(group_name, condition_list):
        all_metrics = []

        for file_name in os.listdir(analysis_dir):
            if file_name.endswith('.csv'):
                if any(condition in file_name for condition in condition_list):
                    file_path = os.path.join(analysis_dir, file_name)
                    file_data = pd.read_csv(file_path)
                    all_metrics.append(file_data)

        if not all_metrics:
            print(f"No matching files found for group '{group_name}' with conditions {condition_list}")
            return None

        combined_data = pd.concat(all_metrics, ignore_index=True)

        if 'Time Group' not in combined_data.columns or 'Behavior' not in combined_data.columns:
            raise ValueError("The analysis files are missing required columns ('Time Group' or 'Behavior').")

        summary = combined_data.groupby(['Time Group', 'Behavior', 'Behavior Label']).agg({
            'Fraction Time': ['mean', 'std'],  # Mean and standard deviation
            'Bouts per Minute': ['mean', 'std'],
            'Mean Bout Duration (s)': ['mean', 'std']
        }).reset_index()

        summary.columns = ['Time Group', 'Behavior', 'Behavior Label',
                           'Fraction Time (mean)', 'Fraction Time (std)',
                           'Bouts per Minute (mean)', 'Bouts per Minute (std)',
                           'Mean Bout Duration (mean)', 'Mean Bout Duration (std)']

        summary = summary.dropna(subset=[
            'Fraction Time (mean)',
            'Bouts per Minute (mean)',
            'Mean Bout Duration (mean)'
        ], how='all')

        return summary

    for group_name in selected_groups:
        summary = aggregate_cohort_data(group_name, selected_conditions)
        if summary is not None:
            summary_file_path = os.path.join(cohort_summary_dir, f'{group_name}_cohort_summary.csv')
            summary.to_csv(summary_file_path, index=False)
            print(f"Saved cohort summary for group '{group_name}' to {summary_file_path}")

    print("Cohort summaries created.")
