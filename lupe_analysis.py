import base64
import os
from pathlib import Path
import extra_streamlit_components as stx
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import streamlit as st
from PIL import Image
import pickle
import importlib
from app_pages import analysis, behavior_analysis
from utils.preprocess_step1 import preprocess_step1
from utils.preprocess_step2 import preprocess_get_features
from utils.preprocess_step3 import preprocess_get_behaviors
from utils.analysis_scripts.behavior_binned_ratio import behavior_binned_ratio_timeline
from utils.analysis_scripts.behavior_distance_traveled import behavior_distance_traveled_heatmaps
from utils.analysis_scripts.behavior_csv_classification import behavior_csv_classification
from utils.analysis_scripts.behavior_bout_counts import behavior_bout_counts
from utils.analysis_scripts.behavior_bout_durations import behavior_bout_durations
from utils.analysis_scripts.behavior_location import behavior_location
from utils.analysis_scripts.behavior_total_frames import behavior_total_frames
from utils.analysis_scripts.behavior_transitions import behavior_transitions
from utils.analysis_scripts.behavior_timepoint_comparison import behavior_timepoint_comparison
from utils.analysis_scripts.behavior_kinematx import behavior_kinematx

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import streamlit as st


def run_notebook(notebook_path, output_path):
    try:
        # Load the notebook
        with open(notebook_path, "r") as f:
            notebook = nbformat.read(f, as_version=4)

        # Create an execution preprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        # Execute the notebook
        ep.preprocess(notebook, {"metadata": {"path": "./"}})

        # Save the executed notebook
        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        # Extract cell outputs
        outputs = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and "outputs" in cell:
                for output in cell["outputs"]:
                    outputs.append(output)

        return outputs
    except Exception as e:
        st.error(f"Failed to execute notebook: {e}")
        return None

# Analysis Options Dictionary
ANALYSIS_DETAILS = {
    "Behavior Binned Ratio": {
        "module": "behavior_binned_ratio",
        "description": "Analyze behavior binned ratios across groups and conditions.",
    },
    "Another Analysis": {
        "module": "another_analysis_script",
        "description": "Description of another analysis.",
    },
    # Add more analyses as needed
}
# Helper functions
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=500):
    img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}' width='{width}px' class='img-fluid'>"
    return img_html


# Constants
HERE = Path(__file__).parent.resolve()
icon_fname = HERE.joinpath("images/logo_mouse.png")
icon_img = Image.open(icon_fname)

# Set Streamlit configurations
st.set_page_config(
    page_title="LUPE",
    page_icon=icon_img,
    layout="centered",
    menu_items={}
)

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        min-width: 320px;
        max-width: 320px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -320px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

logo_fname = HERE.joinpath("images/logo.png")
st.markdown("<p style='text-align: center; color: grey; '>" +
            img_to_html(logo_fname, width=250) + "</p>",
            unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    if 'classifier' not in st.session_state:
        model_name = HERE.joinpath('model/model.pkl')
        with open(model_name, 'rb') as fr:
            st.session_state['classifier'] = pickle.load(fr)

    if 'annotated_behaviors' not in st.session_state:
        st.session_state['annotated_behaviors'] = ['still',
                                                   'walking',
                                                   'rearing',
                                                   'grooming',
                                                   'licking hindpaw L',
                                                   'licking hindpaw R']

    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = {}

    if 'group_names' not in st.session_state:
        st.session_state['group_names'] = {}

    if 'condition_names' not in st.session_state:
        st.session_state['condition_names'] = []

    # Configure Groups
    st.markdown("### Configure Groups")
    num_groups = st.number_input(
        "Number of Groups:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="num_groups"
    )

    while len(st.session_state["group_names"]) < num_groups:
        new_group_idx = len(st.session_state["group_names"]) + 1
        st.session_state["group_names"][f"group_{new_group_idx}"] = f"Group {new_group_idx}"

    while len(st.session_state["group_names"]) > num_groups:
        removed_group_key = f"group_{len(st.session_state['group_names'])}"
        st.session_state["group_names"].pop(removed_group_key, None)
        st.session_state["uploaded_files"].pop(removed_group_key, None)

    for group_idx in range(1, num_groups + 1):
        group_key = f"group_{group_idx}"
        group_name = st.text_input(
            f"Enter Name for Group {group_idx}:",
            value=st.session_state["group_names"][group_key],
            key=f"group_key_{group_idx}"
        )
        st.session_state["group_names"][group_key] = group_name

    # Configure Conditions
    st.markdown("### Configure Conditions")
    num_conditions = st.number_input(
        "Number of Conditions:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="num_conditions"
    )

    while len(st.session_state["condition_names"]) < num_conditions:
        st.session_state["condition_names"].append(f"Condition {len(st.session_state['condition_names']) + 1}")

    while len(st.session_state["condition_names"]) > num_conditions:
        st.session_state["condition_names"].pop()

    for condition_idx in range(num_conditions):
        st.session_state["condition_names"][condition_idx] = st.text_input(
            f"Enter Name for Condition {condition_idx + 1}:",
            value=st.session_state["condition_names"][condition_idx],
            key=f"condition_name_{condition_idx + 1}"
        )

    # Clear any previously stored uploaded files so we can rebuild the dictionary with the updated keys.
    st.session_state["uploaded_files"] = {}

    st.markdown("### Upload Files")
    for group_key, group_name in st.session_state["group_names"].items():
        st.markdown(f"#### {group_name}")
        for condition_name in st.session_state["condition_names"]:
            # Use the user-defined group name as the key in the file uploader's session state.
            uploaded = st.file_uploader(
                f"Upload Files for {group_name} - {condition_name}:",
                accept_multiple_files=True,
                type=["csv"],
                key=f"file_{group_name}_{condition_name}"
            )

            if uploaded:
                # Use the actual group name (user-defined) as the key
                if group_name not in st.session_state["uploaded_files"]:
                    st.session_state["uploaded_files"][group_name] = {}
                st.session_state["uploaded_files"][group_name][condition_name] = uploaded

            # If there are uploaded files for this group and condition, show them.
            if (group_name in st.session_state["uploaded_files"] and
                    condition_name in st.session_state["uploaded_files"][group_name]):
                st.markdown("Uploaded Files:")
                st.dataframe([file.name for file in st.session_state["uploaded_files"][group_name][condition_name]])

def preprocess_workflow():
    st.markdown("## Preprocessing Workflow")

    # Input for project name
    project_name = st.text_input("Enter Project Name:", key="project_name")

    if not project_name:
        st.warning("Please enter a project name to proceed.")
        return

    # Save project name to session state for analyses
    if not st.session_state.get("current_project") or st.session_state["current_project"] != project_name:
        st.session_state["current_project"] = project_name  # Save the project name

    # Paths to expected files
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    raw_data_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")
    features_file = os.path.join(base_dir, f"binned_features_{project_name}.pkl")
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Determine workflow progress
    raw_data_exists = os.path.exists(raw_data_file)
    features_exist = os.path.exists(features_file)
    behaviors_exist = os.path.exists(behaviors_file)

    st.markdown("### Workflow Progress")
    st.markdown(f"**Step 1: Preprocess Data** - {'Completed ✅' if raw_data_exists else 'Pending ⏳'}")
    st.markdown(f"**Step 2: Extract Features** - {'Completed ✅' if features_exist else 'Pending ⏳'}")
    st.markdown(f"**Step 3: Predict Behaviors** - {'Completed ✅' if behaviors_exist else 'Pending ⏳'}")

    # Step 1: Preprocess Data
    if not raw_data_exists:
        st.markdown("### Step 1: Preprocess Data")
        uploaded_files = st.session_state.get("uploaded_files", {})

        # Debug: Print the uploaded_files dictionary
        st.write("📋 Checkpoint!: Confirm Your Uploaded Files Dictionary")
        st.write(uploaded_files)

        if not uploaded_files:
            st.warning("No files uploaded. Please upload files in the sidebar.")
            return

        if st.button("Run Preprocessing Step 1"):
            try:
                processed_file_path = preprocess_step1(
                    project_name,
                    list(st.session_state["group_names"].values()),
                    st.session_state["condition_names"],
                    uploaded_files
                )
                st.success(f"Step 1 completed! Data saved at {processed_file_path}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in Step 1: {e}")

    # Step 2: Extract Features
    elif not features_exist:
        st.markdown("### Step 2: Extract Features")
        if st.button("Run Preprocessing Step 2"):
            try:
                features_file_path = preprocess_get_features(project_name)
                st.success(f"Step 2 completed! Features saved at {features_file_path}")
                # Comment out automatic rerun so you can see debug info:
                # st.experimental_rerun()
                if st.button("Continue to Step 3"):
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in Step 2: {e}")

    # Step 3: Predict Behaviors
    elif not behaviors_exist:
        st.markdown("### Step 3: Predict Behaviors")
        if st.button("Run Preprocessing Step 3"):
            try:
                # Only pass project_name, since groups and conditions are handled inside preprocess_get_behaviors
                behaviors_file_path = preprocess_get_behaviors(project_name)
                st.success(f"Step 3 completed! Behaviors saved at {behaviors_file_path}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in Step 3: {e}")

    else:
        st.success("All preprocessing steps are complete! 🚀")

# Analysis Workflow
def analysis_workflow():
    st.markdown("## LUPE Analysis Pipeline")

    if "current_project" not in st.session_state or not st.session_state["current_project"]:
        st.warning("Please select or create a project in the Preprocessing Workflow tab first.")
        return

    project_name = st.session_state["current_project"]
    processed_data_dir = f"./LUPEAPP_processed_dataset/{project_name}/"

    if not os.path.exists(processed_data_dir):
        st.error(f"Processed data for project '{project_name}' not found. Please run preprocessing first.")
        return

    st.markdown("### Select Groups and Conditions")
    st.markdown("###### Enter groups and conditions in the sidebar if not already present.")
    selected_groups = st.multiselect(
        "Select Groups:",
        options=list(st.session_state["group_names"].values()),
        default=list(st.session_state["group_names"].values())[:1],
        key="general_groups"
    )
    selected_conditions = st.multiselect(
        "Select Conditions:",
        options=st.session_state["condition_names"],
        default=st.session_state["condition_names"][:1],
        key="general_conditions"
    )

    # Analysis selection
    analysis_type = st.radio(
        "Select Analysis:",
        options=[
            "Behavior Binned-Ratio Timeline",
            "Distance Traveled Heatmaps",
            "Behavior CSV Classification",
            "Behavior Bout Counts",
            "Behavior Bout Durations",
            "Behavior Location",
            "Behavior Total Frames",
            "Behavior Transitions",
            "Behavior Timepoint Comparison",
            "Behavior Kinematx"
        ],
        index=0
    )

    num_min = None
    if analysis_type == "Behavior Binned-Ratio Timeline":
        num_min = st.number_input(
            "Select Time Bin Size (minutes):",
            min_value=1,
            max_value=60,
            value=1,
            step=1,
            key="binned_ratio_timebin"
        )

    time_ranges = []
    if analysis_type == "Behavior Timepoint Comparison":
        st.markdown("### Define Time Ranges (in minutes)")
        num_timepoints = st.number_input(
            "Enter the number of time ranges you want to compare (e.g., 2, 3, etc.):",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key="timepoint_number"
        )
        for i in range(num_timepoints):
            time_range = st.text_input(f"Time range {i + 1} (e.g., 0-10):", key=f"time_range_{i}")
            try:
                start_min, end_min = map(int, time_range.split('-'))
                if start_min >= end_min:
                    st.error(f"Error: Start time ({start_min}) must be less than end time ({end_min}).")
                    return
                start_sec, end_sec = start_min * 60, end_min * 60
                time_ranges.append((start_sec, end_sec))
            except ValueError:
                st.error("Invalid input format. Please enter the time range as 'start-end' (e.g., 0-10).")
                return

    # Additional inputs for Behavior Kinematx analysis
    if analysis_type == "Behavior Kinematx":
        st.markdown("### Behavior Kinematx Settings")
        # Choose a single group for this analysis
        selected_group = st.selectbox(
            "Select Group (only one):",
            options=list(st.session_state["group_names"].values()),
            key="kinematx_group"
        )

        selected_conditions = st.multiselect(
            "Select Conditions:",
            options=st.session_state["condition_names"],
            default=st.session_state["condition_names"][:1],
            key="kinematx_conditions"
        )
        # Provide a selectbox for the bodypart of interest
        bp_options = ['nose', 'mouth', 'l_forepaw', 'l_forepaw_digit', 'r_forepaw',
                      'r_forepaw_digit', 'l_hindpaw', 'l_hindpaw_digit1', 'l_hindpaw_digit2',
                      'l_hindpaw_digit3', 'l_hindpaw_digit4', 'l_hindpaw_digit5', 'r_hindpaw',
                      'r_hindpaw_digit1', 'r_hindpaw_digit2', 'r_hindpaw_digit3', 'r_hindpaw_digit4',
                      'r_hindpaw_digit5', 'genitalia', 'tail_base']
        bp_selects = st.selectbox("Select Bodypart:", bp_options, key="kinematx_bp")

    if st.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            try:
                if analysis_type == "Behavior Binned-Ratio Timeline":
                    figs = behavior_binned_ratio_timeline(project_name, selected_groups, selected_conditions, num_min)
                elif analysis_type == "Distance Traveled Heatmaps":
                    figs = behavior_distance_traveled_heatmaps(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Location":
                    figs = behavior_location(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Transitions":
                    figs = behavior_transitions(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior CSV Classification":
                    behavior_csv_classification(project_name)
                    st.success("CSV files generated successfully!")
                    return  # No figures to display
                elif analysis_type == "Behavior Bout Counts":
                    fig = behavior_bout_counts(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Bout Durations":
                    fig = behavior_bout_durations(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Total Frames":
                    fig = behavior_total_frames(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Timepoint Comparison":
                    behavior_timepoint_comparison(project_name, selected_groups, selected_conditions, time_ranges)
                    st.success("Timepoint comparison analysis completed!")
                    return  # No figures to display for this analysis
                elif analysis_type == "Behavior Kinematx":
                    fig = behavior_kinematx(project_name, selected_group, selected_conditions, bp_selects)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return
        st.success("Analysis completed!")

        # Display the resulting figure(s)
        if analysis_type in [
            "Behavior Bout Counts",
            "Behavior Bout Durations",
            "Behavior Total Frames",
            "Behavior Kinematx"
        ]:
            st.pyplot(fig)
        elif analysis_type in [
            "Behavior Binned-Ratio Timeline",
            "Distance Traveled Heatmaps",
            "Behavior Location",
            "Behavior Transitions"
        ]:
            for f in figs:
                st.pyplot(f)

# Main Function
def main():
    st.markdown(f" <h1 style='text-align: left; color: #FFFFFF; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"Introducing LUPE, the innovative no code website predicting pain behavior in mice. "
                f"With our platform, you can input pose and classify mice behavior inside the LUPE Box. "
                f"LUPE can further summarize a composite pain score for mice behavior. "
                f"Best of all, LUPE runs without the need for a GPU. "
                f"With in-depth analysis and interactive visuals, "
                f"as well as downloadable CSVs for easy integration into your existing workflow. "
                f"Try LUPE today and unlock a new level of insights into animal behavior."
                , unsafe_allow_html=True)
    st.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>Select an example behavior</h1> "
                , unsafe_allow_html=True)
    selected_behavior = st.radio('Behaviors',
                                 options=st.session_state['annotated_behaviors'],
                                 index=0,
                                 horizontal=True,
                                 label_visibility='collapsed')

    _, mid_col, _ = st.columns([0.5, 1.5, 0.5])
    try:
        behav_viddir = HERE.joinpath('behavior_videos')
        mid_col.image(f'{behav_viddir}/{selected_behavior}.gif')
    except:
        pass
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE is developed by Justin James and Alexander Hsu</h1> "
                    , unsafe_allow_html=True)

# Navigation Logic
page_names = ['Home', 'Preprocessing Workflow', 'Analysis Pipeline']
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title=page_names[0], description="Model description"),
    stx.TabBarItemData(id=2, title=page_names[1], description="Run preprocessing"),
    stx.TabBarItemData(id=3, title=page_names[2], description="Run Analysis"),
], default=1)

if page_names[int(chosen_id) - 1] == 'Home':
    main()
elif page_names[int(chosen_id) - 1] == 'Preprocessing Workflow':
    preprocess_workflow()
elif page_names[int(chosen_id) - 1] == 'Analysis Pipeline':
    analysis_workflow()