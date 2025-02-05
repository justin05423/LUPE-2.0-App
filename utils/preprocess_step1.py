import os
import pickle
import numpy as np
import pandas as pd
from utils.feature_utils import filter_pose_noise
from tqdm import tqdm


def update_meta_file(project_name, groups, conditions):
    """
    Updates the meta.py file with project-specific groups and conditions.

    Parameters:
        project_name (str): The name of the project.
        groups (list): List of group names.
        conditions (list): List of condition names.
    """
    meta_file_path = './utils/meta.py'

    groups_var = f"groups_{project_name} = {groups}"
    conditions_var = f"conditions_{project_name} = {conditions}"

    # Read the current contents of the meta file
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = []

    # Check if the variables are already defined and update them if necessary
    groups_defined = False
    conditions_defined = False
    for i, line in enumerate(lines):
        if line.startswith(f"groups_{project_name} ="):
            lines[i] = groups_var + '\n'
            groups_defined = True
        elif line.startswith(f"conditions_{project_name} ="):
            lines[i] = conditions_var + '\n'
            conditions_defined = True

    # If the variables are not defined, add them to the end of the file
    if not groups_defined:
        lines.append(groups_var + '\n')
    if not conditions_defined:
        lines.append(conditions_var + '\n')

    # Write the updated contents back to the meta file
    with open(meta_file_path, 'w') as file:
        file.writelines(lines)

    print(f'Updated {meta_file_path} with project-specific groups and conditions.')


def preprocess_data_step1(project_name, uploaded_files, groups, conditions):
    """
    Preprocesses uploaded DLC-analyzed CSV data and saves the data dictionary as a pickle file.

    Parameters:
        project_name (str): The name of the project.
        uploaded_files (dict): Dictionary of uploaded files grouped by groups and conditions.
        groups (list): List of group names.
        conditions (list): List of condition names.

    Returns:
        str: Path to the saved pickle file.
    """
    # Update meta.py with project-specific information
    update_meta_file(project_name, groups, conditions)

    # Initialize the data structure
    data = {group: {condition: {} for condition in conditions} for group in groups}

    print("Processing and filtering pose data...")
    for group in tqdm(groups, desc="Groups"):
        for condition in tqdm(conditions, desc=f"Conditions in {group}"):
            if group in uploaded_files and condition in uploaded_files[group]:
                for uploaded_file in tqdm(uploaded_files[group][condition],
                                          desc=f"Processing files in {group}/{condition}"):
                    try:
                        # Read CSV file into DataFrame
                        temp_df = pd.read_csv(uploaded_file, header=[0, 1, 2, 3], sep=",", index_col=0)

                        # Extract pose indices and filter pose noise
                        selected_pose_idx = np.arange(temp_df.shape[1])
                        idx_llh = selected_pose_idx[2::3]
                        idx_selected = [i for i in selected_pose_idx if i not in idx_llh]

                        currdf_filt, _ = filter_pose_noise(
                            temp_df,
                            idx_selected=idx_selected,
                            idx_llh=idx_llh,
                            llh_value=0.1
                        )

                        # Save filtered data into the dictionary
                        file_name = uploaded_file.name
                        data[group][condition][file_name] = currdf_filt

                    except Exception as e:
                        print(f"Error processing file {uploaded_file.name}: {e}")
                        raise ValueError(f"Failed to process file: {uploaded_file.name}")

    # Save the data dictionary to a pickle file
    output_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"raw_data_{project_name}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Processed data saved to: {output_file_path}")
    return output_file_path