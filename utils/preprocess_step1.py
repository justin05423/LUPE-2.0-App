import os
import datetime
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO

# Import the CSV reading helper and noise filtering function
from utils.feature_utils import filter_pose_noise


# Import the update_meta_file function from this same module (or ensure it is accessible)
def update_meta_file(project_name, groups, conditions):
    """
    Update the meta.py file with project-specific groups and conditions.
    """
    meta_file_path = os.path.join(os.path.dirname(__file__), 'meta.py')

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


def preprocess_step1(project_name, groups, conditions, uploaded_files):
    """
    Preprocess the uploaded CSV files and store the data in a dictionary.

    Parameters:
        project_name (str): Name of the project.
        groups (list): List of group names.
        conditions (list): List of condition names.
        uploaded_files (dict): Dictionary of uploaded files, organized by group and condition.

    Returns:
        str: Path to the saved raw data file.
    """
    print(f"\n🔍 DEBUG: Starting Step 1 for Project: {project_name}\n")

    # Automatically update the meta file with the provided groups and conditions
    update_meta_file(project_name, groups, conditions)

    # Initialize the data dictionary
    data = {group: {condition: {} for condition in conditions} for group in groups}

    # Process each uploaded file
    for group in tqdm(groups, desc="Processing groups"):
        for condition in tqdm(conditions, desc=f"Processing conditions in {group}"):
            if group in uploaded_files and condition in uploaded_files[group]:
                for uploaded_file in uploaded_files[group][condition]:
                    try:
                        print(f"\n📂 Processing file: {uploaded_file.name} (Size: {uploaded_file.size} bytes)\n")

                        # Convert the uploaded file's bytes to a string and then to a file-like object
                        file_content = uploaded_file.getvalue().decode("utf-8")
                        file_buffer = StringIO(file_content)

                        # Read the CSV file
                        temp_df = pd.read_csv(file_buffer, header=[0, 1, 2, 3], sep=",", index_col=0)

                        print(f"✅ Successfully loaded {uploaded_file.name} | Shape: {temp_df.shape}")

                        # Filter pose noise
                        selected_pose_idx = np.arange(temp_df.shape[1])
                        idx_llh = selected_pose_idx[2::3]
                        idx_selected = [i for i in selected_pose_idx if i not in idx_llh]
                        currdf_filt, _ = filter_pose_noise(temp_df, idx_selected=idx_selected, idx_llh=idx_llh,
                                                           llh_value=0.1)

                        # Use the file name (without extension) as the key
                        file_name = os.path.splitext(uploaded_file.name)[0]
                        data[group][condition][file_name] = currdf_filt

                        print(f"✅ Processed & stored: {file_name} under [{group} -> {condition}]")

                    except Exception as e:
                        print(f"🚨 Error processing file {uploaded_file.name}: {e}")

    # Save the processed data as a pickle file
    directory = f"./LUPEAPP_processed_dataset/{project_name}/"
    os.makedirs(directory, exist_ok=True)

    # Save project info to a text file for later reference
    project_info_filename = os.path.join(directory, f"project_info_{project_name}.txt")
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(project_info_filename, "w") as f_info:
        f_info.write(f"project_name = {project_name}\n")
        f_info.write(f"groups = {groups}\n")
        f_info.write(f"conditions = {conditions}\n")
        f_info.write(f"analysis_date = {analysis_time}\n")

    raw_data_pkl_filename = os.path.join(directory, f"raw_data_{project_name}.pkl")
    with open(raw_data_pkl_filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ {raw_data_pkl_filename} is created and saved!\n")
    return raw_data_pkl_filename