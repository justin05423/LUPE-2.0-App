import os
import pickle
import importlib
from tqdm import tqdm
from utils.classification import load_data, load_features
from utils.feature_utils import feature_extraction

def preprocess_get_features(project_name):
    """
    Extract features for each group and condition from the raw data.

    Parameters:
        project_name (str): The name of the project.

    Returns:
        str: Path to the saved pickle file containing features.
    """
    # Dynamically load group and condition data from meta.py
    try:
        meta_module = importlib.import_module("utils.meta")
        groups = getattr(meta_module, f"groups_{project_name}")
        conditions = getattr(meta_module, f"conditions_{project_name}")
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Error loading meta information for project '{project_name}': {e}")

    # Paths for input and output
    data_path = f"./LUPEAPP_processed_dataset/{project_name}/raw_data_{project_name}.pkl"
    output_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    output_file_path = os.path.join(output_dir, f"binned_features_{project_name}.pkl")

    # Ensure raw data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    # Load raw data
    print("Loading raw data...")
    data = load_data(data_path)

    # Check if features already exist
    if os.path.exists(output_file_path):
        print("Features already exist. Skipping extraction.")
        return output_file_path

    # Extract features
    print("Extracting features...")
    features = {group: {} for group in groups}
    for group in tqdm(groups, desc="Groups"):
        for condition in tqdm(conditions, desc=f"Conditions in {group}", leave=False):
            if condition not in data[group]:
                raise ValueError(f"Condition '{condition}' not found in raw data for group '{group}'.")
            features[group][condition] = {}

            for file_name in tqdm(data[group][condition], desc=f"Processing {group}/{condition}", leave=False):
                features[group][condition][file_name] = feature_extraction(
                    [data[group][condition][file_name]],
                    num_train=1,
                    framerate=60,
                )

    # Save features to a pickle file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"Features saved to: {output_file_path}")
    return output_file_path