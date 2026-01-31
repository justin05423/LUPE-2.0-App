import base64
import os
from pathlib import Path
import extra_streamlit_components as stx
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import streamlit as st
import re
import glob
import pandas as pd
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
from utils.analysis_scripts.behavior_binned_mouse_screening import behavior_binned_mouse_screening
from utils.analysis_scripts.behavior_LUPE_AMPS import behavior_LUPE_AMPS

def run_notebook(notebook_path, output_path):
    try:
        with open(notebook_path, "r") as f:
            notebook = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        ep.preprocess(notebook, {"metadata": {"path": "./"}})

        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        outputs = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and "outputs" in cell:
                for output in cell["outputs"]:
                    outputs.append(output)

        return outputs
    except Exception as e:
        st.error(f"Failed to execute notebook: {e}")
        return None

# Analysis Options
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


# Helper to parse project metadata file for groups and conditions
def parse_project_info_file(meta_path: str):
    """Parse a project_info_<project>.txt file and return (groups, conditions)."""
    groups, conditions = [], []
    section = None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('Groups:'):
                    section = 'groups'
                    continue
                if line.startswith('Conditions:'):
                    section = 'conditions'
                    continue
                if line.startswith('Files by Group and Condition'):
                    section = 'files'
                    continue
                if line.startswith('[') and line.endswith(']'):
                    section = 'files'
                    continue
                if line.endswith(':') and section == 'files':
                    continue
                if line.startswith('- '):
                    item = line[2:].strip()
                    if section == 'groups':
                        groups.append(item)
                    elif section == 'conditions':
                        conditions.append(item)
    except Exception:
        pass
    return groups, conditions

# Constants
HERE = Path(__file__).parent.resolve()
icon_fname = HERE.joinpath("images/logo_mouse.png")
icon_img = Image.open(icon_fname)

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
    if "classifier" not in st.session_state:
        model_dir = HERE / "model"

        candidate_names = ["model.pkl", "model"]

        classifier_model_path = None
        for name in candidate_names:
            p = model_dir / name
            if p.exists():
                classifier_model_path = p
                break

        if classifier_model_path is None:
            st.error(
                "No model file found. Please check folder named 'model'. Expected one of: "
                "model.pkl or model"
            )
        else:
            with open(classifier_model_path, "rb") as fr:
                st.session_state["classifier"] = pickle.load(fr)

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

    # Apply any pending values loaded from project metadata BEFORE widgets are instantiated
    if st.session_state.get("apply_meta_on_sidebar", False):
        if "pending_num_groups" in st.session_state:
            st.session_state["num_groups"] = st.session_state.pop("pending_num_groups")
        if "pending_num_conditions" in st.session_state:
            st.session_state["num_conditions"] = st.session_state.pop("pending_num_conditions")
        st.session_state["apply_meta_on_sidebar"] = False

    # Configure Groups
    st.markdown("### Configure Groups")
    st.caption("Tip: Limit spaces in group names to prevent confusion with coding language.")
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
    st.caption("Tip: Limit spaces in condition names to prevent confusion with coding language.")
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
            uploaded = st.file_uploader(
                f"Upload Files for {group_name} - {condition_name}:",
                accept_multiple_files=True,
                type=["csv"],
                key=f"file_{group_name}_{condition_name}"
            )

            if uploaded:
                if group_name not in st.session_state["uploaded_files"]:
                    st.session_state["uploaded_files"][group_name] = {}
                st.session_state["uploaded_files"][group_name][condition_name] = uploaded

            if (group_name in st.session_state["uploaded_files"] and
                    condition_name in st.session_state["uploaded_files"][group_name]):
                st.markdown("Uploaded Files:")
                st.dataframe([file.name for file in st.session_state["uploaded_files"][group_name][condition_name]])

# Preprocess Workflow
def preprocess_workflow():
    st.markdown("## Preprocessing Workflow")

    # Input for project name (existing dropdown + create new)
    data_root = "./LUPEAPP_processed_dataset"
    os.makedirs(data_root, exist_ok=True)
    existing_projects = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    selector_options = ["➕ Create New Project…"] + existing_projects
    selected_option = st.selectbox(
        "Select an existing project or create a new one:",
        options=selector_options,
        index=0,
        key="project_selector",
        help=(
            "Choose an existing project to continue where you left off, "
            "or select 'Create New Project…' to start a new project."
        ),
    )

    if selected_option == "➕ Create New Project…":
        project_name = st.text_input(
            "Enter Project Name:",
            key="project_name",
            help=(
                "Tip: If you're creating a new project, avoid spaces in the project name to prevent "
                "confusion with code and file paths. Use underscores (_) instead (e.g., 'Formalin_Test_01')."
            ),
        )
    else:
        project_name = selected_option
        st.info(f"Using existing project: {project_name}")

    if not project_name:
        st.warning("Please enter a project name to proceed.")
        return

    if not st.session_state.get("current_project") or st.session_state["current_project"] != project_name:
        st.session_state["current_project"] = project_name  # Save the project name

    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    raw_data_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")
    features_file = os.path.join(base_dir, f"binned_features_{project_name}.pkl")
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # If an existing project is selected, try to auto-populate groups/conditions from its metadata file.
    # Ensure we only do this once per selected project to avoid loops.
    if selected_option != "➕ Create New Project…" and st.session_state.get("meta_loaded_for_project") != project_name:
        meta_path = os.path.join(base_dir, f"project_info_{project_name}.txt")
        if os.path.exists(meta_path):
            g_list, c_list = parse_project_info_file(meta_path)
            updated = False
            if g_list:
                st.session_state["group_names"] = {f"group_{i+1}": g for i, g in enumerate(g_list)}
                st.session_state["pending_num_groups"] = len(g_list)
                updated = True
            if c_list:
                st.session_state["condition_names"] = list(c_list)
                st.session_state["pending_num_conditions"] = len(c_list)
                updated = True
            if updated:
                st.session_state["meta_loaded_for_project"] = project_name
                st.session_state["apply_meta_on_sidebar"] = True
                st.experimental_rerun()

    # (Project directory and metadata file for new projects are now handled after Step 1 completes.)
    raw_data_exists = os.path.exists(raw_data_file)
    features_exist = os.path.exists(features_file)
    behaviors_exist = os.path.exists(behaviors_file)

    st.markdown("### Workflow Progress")
    st.markdown(f"**Step 1: Preprocess Data** - {'Completed ✅' if raw_data_exists else 'Pending ⏳... Begin by configuring Groups/Conditions and adding DLC output .csv files.'}")
    st.markdown(f"**Step 2: Extract Features** - {'Completed ✅' if features_exist else 'Pending ⏳'}")
    st.markdown(f"**Step 3: Predict Behaviors** - {'Completed ✅' if behaviors_exist else 'Pending ⏳'}")

    # Step 1: Preprocess Data
    if not raw_data_exists:
        st.markdown("### Step 1: Preprocess Data")
        uploaded_files = st.session_state.get("uploaded_files", {})

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
                os.makedirs(base_dir, exist_ok=True)
                meta_path = os.path.join(base_dir, f"project_info_{project_name}.txt")
                try:
                    from datetime import datetime
                    with open(meta_path, "w", encoding="utf-8") as f:
                        f.write(f"Project: {project_name}\n")
                        f.write(f"Created/Updated: {datetime.now().isoformat()}\n\n")
                        f.write("Groups:\n")
                        for g in list(st.session_state.get("group_names", {}).values()):
                            f.write(f"- {g}\n")
                        f.write("\nConditions:\n")
                        for c in st.session_state.get("condition_names", []):
                            f.write(f"- {c}\n")
                        f.write("\nFiles by Group and Condition:\n")
                        up = uploaded_files if isinstance(uploaded_files, dict) else {}
                        for gname, cond_map in up.items():
                            f.write(f"\n[{gname}]\n")
                            for cname, files in (cond_map or {}).items():
                                f.write(f"  {cname}:\n")
                                for uf in (files or []):
                                    try:
                                        fname = getattr(uf, 'name', str(uf))
                                    except Exception:
                                        fname = str(uf)
                                    f.write(f"    - {fname}\n")
                    st.info(f"Project metadata file saved: {meta_path}")
                except Exception as meta_e:
                    st.warning(f"Could not write project metadata file: {meta_e}")
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

                if st.button("Continue to Step 3"):
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in Step 2: {e}")

    # Step 3: Predict Behaviors
    elif not behaviors_exist:
        st.markdown("### Step 3: Predict Behaviors")
        if st.button("Run Preprocessing Step 3"):
            try:
                behaviors_file_path = preprocess_get_behaviors(project_name)
                st.success(f"Step 3 completed! Behaviors saved at {behaviors_file_path}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error in Step 3: {e}")

    else:
        st.success("All preprocessing steps are complete! 🚀")

# LUPE Analysis Workflow
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
            "Behavior Kinematx",
            "Behavior Binned Mouse Screening"
        ],
        index=0
    )

    # Dynamic analysis description (shows context for the selected analysis)
    analysis_help = {
        "Behavior Binned-Ratio Timeline": (
            "This analysis auto-detects your project’s groups/conditions, computes mean ± SEM behavior ratios "
            "across consecutive bins, and saves one SVG per group plus one CSV per group–condition. "
            "Select a time-bin size (minutes), then run."
        ),
        "Distance Traveled Heatmaps": (
            "This analysis computes total and average distance traveled (mean, SD, SEM, cumulative) from pose trajectories "
            "and renders a 2D position heatmap for each group–condition. The app auto-detects "
            "groups/conditions and saves one SVG heatmap per group–condition plus a CSV of summary statistics."
        ),
        "Behavior CSV Classification": (
            "This analysis exports raw behavior classifications as CSVs at per-frame and per-second resolution for each file, organized by group/condition."
        ),
        "Behavior Bout Counts": (
            "This analysis counts behavior bouts (transitions) per file and aggregates them by group–condition to compute mean ± SD, rendering horizontal bar charts."
        ),
        "Behavior Bout Durations": (
            "This analysis parses contiguous behavior bouts from your predictions, converts their lengths to seconds (60 fps by default), computes per-bout durations for each behavior across your project’s groups/conditions, and visualizes the distributions as horizontal boxplots for each group–condition."
        ),
        "Behavior Location": (
            "This analysis maps where each behavior occurs in the arena by aggregating tail-base positions during that behavior into 2D density heatmaps for each selected group–condition (with the arena outline overlaid). It uses your project’s pose + behavior data and saves one SVG per behavior (per group–condition)."
        ),
        "Behavior Total Frames": (
            "This analysis calculates the percent of total frames spent in each behavior for your selected group–condition(s) by pooling all files, and visualizes the distribution as donut pie charts."
        ),
        "Behavior Transitions": (
            "This analysis builds behavior→behavior transition matrices for each selected group–condition by aggregating across files, zeroing self-transitions, and row-normalizing to probabilities."
        ),
        "Behavior Timepoint Comparison": (
            "This analysis compares behaviors across user-defined time windows by binning the per-second classification CSVs into those ranges and computing, for each behavior, Fraction Time, Bouts per Minute, and Mean Bout Duration (s), saving one analysis CSV per file. Inputs required: the number of time ranges (≥2) and each range in minutes entered as 'start-end' (e.g., 0-10, 11-30)."
        ),
        "Behavior Kinematx": (
            "This analysis measures how far the selected body part moves during each behavior, aggregates across files into 10-bin displacement distributions, and visualizes per-condition heatmaps while saving per-file average-displacement and descriptive-stats CSVs. Inputs: choose one group, one or more conditions, and a body part (bp_selects); everything else loads from the current project."
        ),
        "Behavior Binned Mouse Screening": (
            "This analysis aggregates per-frame behavior labels into 1-minute bins for each mouse, producing per-mouse × time heatmaps (frames/min) and saving one CSV + one SVG per behavior. The app auto-detects mice from your project’s per-frame CSVs; optionally set a fixed heatmap maximum to standardize color scaling across plots."
        ),
        # You can add more entries here for other analyses later.
    }
    desc = analysis_help.get(analysis_type)
    if desc:
        st.info(desc)

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
        st.caption(
            "Use contiguous windows like 0-10, 10-20, 20-30. "
            "Windows are interpreted as [start, end): start is included, end is excluded."
        )

        num_timepoints = st.number_input(
            "How many time windows do you want to compare?",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key="timepoint_number"
        )

        parsed_min = []
        valid = True

        for i in range(int(num_timepoints)):
            st.markdown(f"**Time window {i + 1}**")
            c1, c2 = st.columns(2)

            start_min = c1.number_input(
                "Start (min)",
                min_value=0,
                value=i * 10,
                step=1,
                key=f"time_range_start_{i}"
            )
            end_min = c2.number_input(
                "End (min)",
                min_value=0,
                value=(i + 1) * 10,
                step=1,
                key=f"time_range_end_{i}"
            )

            if end_min <= start_min:
                st.error(f"Window {i + 1}: end must be greater than start.")
                valid = False
                break

            parsed_min.append((int(start_min), int(end_min)))

        # Enforce contiguity: end of one == start of next
        if valid:
            for i in range(1, len(parsed_min)):
                prev_start, prev_end = parsed_min[i - 1]
                cur_start, cur_end = parsed_min[i]

                if cur_start != prev_end:
                    st.error(
                        f"Time ranges must be contiguous. "
                        f"Expected window {i + 1} to start at {prev_end} min, got {cur_start} min."
                    )
                    valid = False
                    break

        if valid:
            # Convert minutes → seconds
            time_ranges = [(s * 60, e * 60) for s, e in parsed_min]
            st.success("Time ranges set: " + ", ".join([f"{s}-{e}" for s, e in parsed_min]))

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

    # Additional inputs for Behavior Binned Mouse Screening analysis
    if analysis_type == "Behavior Binned Mouse Screening":
        st.markdown("### Heatmap Settings")
        set_heatmap_max = st.checkbox(
            "Do you want to set a fixed maximum for heatmaps? (Recommended: 1000, 2000, or 3000)",
            key="heatmap_max_toggle")
        if set_heatmap_max:
            heatmap_max = st.number_input("Enter maximum value for heatmaps:", min_value=0, value=3000, step=100,
                                          key="heatmap_max_input")
        else:
            heatmap_max = None

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
                elif analysis_type == "Behavior Bout Counts":
                    fig = behavior_bout_counts(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Bout Durations":
                    fig = behavior_bout_durations(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Total Frames":
                    fig = behavior_total_frames(project_name, selected_groups, selected_conditions)
                elif analysis_type == "Behavior Timepoint Comparison":
                    # Ensure we have valid, parsed time ranges before running
                    num_timepoints = st.session_state.get("timepoint_number", 0)
                    if not time_ranges or len(time_ranges) != num_timepoints:
                        st.error("Please enter all time ranges in 'start-end' format (e.g., 0-10, 11-30) before running.")
                    else:
                        behavior_timepoint_comparison(project_name, selected_groups, selected_conditions, time_ranges)
                elif analysis_type == "Behavior Kinematx":
                    fig = behavior_kinematx(project_name, selected_group, selected_conditions, bp_selects)
                elif analysis_type == "Behavior Binned Mouse Screening":
                    heatmap_files = behavior_binned_mouse_screening(project_name, heatmap_max_value=heatmap_max)
                    for behavior_name, svg_path in heatmap_files.items():
                        st.markdown(f"**{behavior_name.capitalize()} Heatmap**")
                        with open(svg_path, "r") as f:
                            svg_content = f.read()
                        st.components.v1.html(svg_content, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return
        st.success("Analysis completed!")
        st.info(f"All output files, including CSVs and figures, are saved under: {processed_data_dir}")

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

# LUPE-AMPS Analysis Workflow
def pain_state_model_analysis():
    st.markdown("## LUPE-Affective-Motivational Pain Scale (AMPS)")
    st.write(
        """\
LUPE-AMPS is a dedicated module within the LUPE platform that focuses on uncovering latent behavioral states in mice — those subtle patterns that indicate pain or the effect of analgesia. This module uses a combination of behavioral metrics derived from video-based pose estimation and classification.

The ultimate goal of LUPE-AMPS is to translate these behavioral signatures into a quantitative index known as the Affective-Motivational Pain Scale (AMPS). This index helps researchers measure pain in a more nuanced and objective way by correlating the observed behaviors with the subjective experience of pain.
"""
    )

    st.markdown("### Requirements")
    st.markdown(
        """\
The model is specifically trained to analyze localized hindpaw injuries. It leverages behavioral data collected from formalin, capsaicin, and spared nerve injuries to the hindpaw. For more details on the LUPE-AMPS module, please refer to our recent [Nature publication](https://www.nature.com/articles/s41586-025-09908-w).
"""
    )

    st.markdown("### How to Begin?")
    st.markdown(
        """\
1. **Complete LUPE Analysis Pipeline:** Ensure that you complete the entire pipeline, with particular attention to the *Behavior CSV Classification* step.
2. **Organize/Upload CSVs:** Arrange and upload your CSV files according to the experimental conditions that need to be compared, including data from the uninjured/control group.
3. **Configure Project and Conditions:** Name your project (using the dataset from which the LUPE pipeline was run) and set up your groups and conditions. (Enter groups and conditions in the sidebar if not already present.)
"""
    )

    # -----------------------------
    # Model option selection
    # -----------------------------
    st.markdown("### Select Model Option")
    st.markdown(
        "For novel analysis, you can choose to either use the original LUPE-AMPS model (see "
        "[Nature publication](https://www.nature.com/articles/s41586-025-09908-w)) "
        "or create a new LUPE-AMPS model on novel data."
    )
    model_option = st.radio(
        "Choose Model Option",
        options=[
            "Use original LUPE-AMPS model for novel analysis (see [recent publication](https://www.nature.com/articles/s41586-025-09908-w))",
            "Create new LUPE-AMPS model on novel data",
        ],
        index=0,
        key="amps_model_option",
    )

    if model_option == "Create new LUPE-AMPS model on novel data":
        st.markdown("### Create LUPE-AMPS Model")
        st.markdown("🚧 **Under Construction:** Please check back later or contact the developers/authors for more information.")
        return

    # -----------------------------
    # Must have project selected
    # -----------------------------
    if "current_project" not in st.session_state or not st.session_state["current_project"]:
        st.warning("Please select or create a project in the Preprocessing Workflow tab first.")
        return

    st.markdown("### LUPE-AMPS Analysis 🚀")

    # -----------------------------
    # LUPE-AMPS model paths (separate from the sidebar classifier model)
    # -----------------------------
    amps_model_dir = HERE / "model" / "LUPE-AMPS"
    amps_centroids_path = amps_model_dir / "panpainmodel30.mat"
    amps_pca_params_path = amps_model_dir / "pain_scale_params.mat"

    st.session_state["amps_model_dir"] = str(amps_model_dir)
    st.session_state["amps_centroids_path"] = str(amps_centroids_path)
    st.session_state["amps_pca_params_path"] = str(amps_pca_params_path)

    with st.expander("LUPE-AMPS model paths", expanded=False):
        st.write(f"AMPS model dir: {amps_model_dir}")
        st.write(f"Centroids/state model: {amps_centroids_path}  (exists={amps_centroids_path.exists()})")
        st.write(f"Pain-scale params: {amps_pca_params_path}  (exists={amps_pca_params_path.exists()})")

    # -----------------------------
    # Groups / Conditions selection
    # -----------------------------
    st.markdown("### Select Groups and Conditions")
    st.markdown("###### Enter groups and conditions in the sidebar if not already present.")

    group_options = list(st.session_state.get("group_names", {}).values())
    condition_options = st.session_state.get("condition_names", [])

    if len(group_options) == 0:
        st.warning("No groups found. Please define groups in the sidebar / project setup first.")
        return
    if len(condition_options) == 0:
        st.warning("No conditions found. Please define conditions in the sidebar / project setup first.")
        return

    selected_groups = st.multiselect(
        "Select Groups:",
        options=group_options,
        default=group_options[:1],
        key="amps_selected_groups",
    )

    selected_conditions = st.multiselect(
        "Select Conditions:",
        options=condition_options,
        default=condition_options[:1],
        key="amps_selected_conditions",
    )

    # -----------------------------
    # Injury site input
    # -----------------------------
    st.markdown("### Injury Site")
    st.caption(
        "LUPE-AMPS expects the *injured paw lick* for all mice to be treated consistently. "
        "If injury was on the RIGHT hindpaw, the analysis will swap left/right lick labels so that "
        "the injured paw is always handled as the injured channel downstream."
    )

    injury_site = st.radio(
        "Which hindpaw was injured?",
        options=[0, 1],
        format_func=lambda x: "0 — Left hindpaw" if x == 0 else "1 — Right hindpaw",
        index=0,
        key="amps_injury_site",
    )

    # -----------------------------
    # TIMEPOINT COMPARISON (Optional)
    # -----------------------------
    st.markdown("### Optional - Timepoint Comparison")
    st.caption(
        "If enabled, LUPE-AMPS will compute the same cohort-level outputs "
        "(AMPS, state statistics, behavior statistics, transitions) "
        "within user-defined time windows also."
    )

    # Infer available recording length (minutes) from the first readable behavior CSV.
    # This avoids hard-coding a fixed duration (e.g., 30 min) and supports variable-length recordings.
    # Assumes: 1 row = 1 frame at 60 fps in the raw-classification CSVs.
    project_name = st.session_state["current_project"]
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    base_data_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "frames")

    # Try to infer recording length (minutes) from the first readable behavior CSV
    # (assumes 1 row = 1 frame at 60 fps in raw-classification CSVs).
    sampling_rate_fps = 60.0
    recording_length_min_available = None

    try:
        candidate_csvs = []
        for g in selected_groups:
            for c in selected_conditions:
                candidate_csvs.extend(
                    glob.glob(os.path.join(base_data_dir, g, c, "*.csv"))
                )

        candidate_csvs = sorted(candidate_csvs)
        for fp in candidate_csvs:
            try:
                df0 = pd.read_csv(fp)
                n_frames = int(df0.shape[0])
                if n_frames > 0:
                    recording_length_min_available = n_frames / sampling_rate_fps / 60.0
                    break
            except Exception:
                continue
    except Exception:
        recording_length_min_available = None

    if recording_length_min_available is None:
        # Fallback if we can't infer it yet
        recording_length_min_available = float(st.session_state.get("amps_recording_length_min_available", 30))
        st.warning(
            "Could not infer recording length from behavior CSVs yet. "
            "Using fallback duration for time-window setup."
        )

    # Persist for downstream UI defaults
    st.session_state["amps_recording_length_min_available"] = float(recording_length_min_available)

    use_timepoint_comparison = st.toggle(
        "Would you like to compare specific timepoints within your dataset?",
        value=False,
        key="amps_use_timepoint_comparison",
    )

    time_ranges_min = []
    time_labels = []
    timepoint_error = None

    if use_timepoint_comparison:
        with st.expander("Configure time windows", expanded=True):
            st.caption(
                f"Define one or more time windows in minutes (0–{recording_length_min_available:.2f}). "
                "Windows are clipped to the recording length if needed."
            )

            st.info(
                "Tip: Use contiguous windows like 0-10, 10-20, 20-30 for full coverage. "
                "Windows are interpreted as start-inclusive and end-exclusive in slicing (standard Python)."
            )

            num_timepoints = st.number_input(
                "How many time windows do you want to compare?",
                min_value=1,
                max_value=12,
                value=2,
                step=1,
                key="amps_num_timepoints",
            )

            for i in range(int(num_timepoints)):
                st.markdown(f"**Time window {i + 1}**")

                c1, c2 = st.columns(2)
                start_min = c1.number_input(
                    "Start (min)",
                    min_value=0.0,
                    max_value=float(recording_length_min_available),
                    value=float(min(i * 10, recording_length_min_available)),
                    step=1.0,
                    key=f"amps_tp_start_{i}",
                )
                end_min = c2.number_input(
                    "End (min)",
                    min_value=0.0,
                    max_value=float(recording_length_min_available),
                    value=float(min((i + 1) * 10, recording_length_min_available)),
                    step=1.0,
                    key=f"amps_tp_end_{i}",
                )

                # Validation
                if start_min < 0:
                    timepoint_error = f"Window {i + 1}: start must be ≥ 0."
                    continue

                if end_min <= start_min:
                    timepoint_error = f"Window {i + 1}: end must be greater than start."
                    continue

                # Clip to available recording length
                if end_min > recording_length_min_available:
                    end_min = float(recording_length_min_available)

                # Re-check after clipping
                if end_min <= start_min:
                    timepoint_error = f"Window {i + 1}: window becomes empty after clipping."
                    continue

                time_ranges_min.append((start_min, end_min))
                time_labels.append(f"{start_min:g}-{end_min:g} min")

            # Optional: lightweight “coverage” check (not enforced)
            if len(time_ranges_min) > 0 and timepoint_error is None:
                latest_end = max(b for _, b in time_ranges_min)
                if latest_end < recording_length_min_available - 1e-6:
                    st.warning(
                        f"Your windows end at {latest_end:g} min, but the recording is "
                        f"{recording_length_min_available:.2f} min. (That’s okay if intentional.)"
                    )

            if timepoint_error:
                st.error(timepoint_error)

            if len(time_ranges_min) > 0:
                st.markdown("**Time windows defined:**")
                for lbl, (smin, emin) in zip(time_labels, time_ranges_min):
                    st.write(f"• {lbl} → ({smin:g}, {emin:g}) min")

    # -----------------------------
    # Run button
    # -----------------------------
    run_disabled = (len(selected_groups) == 0) or (len(selected_conditions) == 0) or (use_timepoint_comparison and (timepoint_error is not None))

    if st.button("Run LUPE-AMPS Analysis", disabled=run_disabled):
        with st.spinner("Running LUPE-AMPS analysis..."):
            try:
                project_name = st.session_state["current_project"]
                try:
                    behavior_LUPE_AMPS(
                        project_name=project_name,
                        selected_groups=selected_groups,
                        selected_conditions=selected_conditions,
                        injury_site=injury_site,
                        use_timepoint_comparison=use_timepoint_comparison,
                        time_ranges_min=time_ranges_min,
                        time_labels=time_labels,
                        model_path=str(amps_centroids_path),
                        pca_model_path=str(amps_pca_params_path),
                    )
                except TypeError:
                    # Backward-compatible: older versions of behavior_LUPE_AMPS may not accept these kwargs
                    behavior_LUPE_AMPS(
                        project_name=project_name,
                        selected_groups=selected_groups,
                        selected_conditions=selected_conditions,
                        injury_site=injury_site,
                        use_timepoint_comparison=use_timepoint_comparison,
                        time_ranges_min=time_ranges_min,
                        time_labels=time_labels,
                    )
                st.success("LUPE-AMPS analysis completed! See project directory to see the results.")
            except Exception as e:
                st.error(f"Error during LUPE-AMPS analysis: {e}")

    # Footer
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(
            "<h1 style='text-align: left; color: gray; font-size:16px; font-family:Avenir; font-weight:normal'>"
            "LUPE-AMPS was developed by Sophie Rogers, and this version of the code has been adapted and reproduced by Justin James."
            "</h1>",
            unsafe_allow_html=True,
        )

# Welcome Page
def main():
    st.markdown(
        """
        <div style="background-color:#1f1f1f; color:#eeeeee; border:1px solid #3a3a3a; border-radius:12px; padding:16px 20px; line-height:1.6; font-family:Avenir; text-align:center; font-size:18px; max-width:900px; margin:0 auto;">
          Introducing LUPE, the innovative no code website predicting pain behavior in mice. With our platform, you can input pose and classify mice behavior inside the LUPE Box. LUPE can further summarize a composite pain score for mice behavior. Best of all, LUPE runs without the need for a GPU. With in-depth analysis and interactive visuals, as well as downloadable CSVs for easy integration into your existing workflow. Try LUPE today and unlock a new level of insights into animal behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f" <h1 style='text-align: left; color: #000000; font-size:18px; "
        f"font-family:Avenir; font-weight:normal'>Select an example behavior:</h1> ",
        unsafe_allow_html=True,
    )

    selected_behavior = st.radio('Behaviors',
                                 options=st.session_state['annotated_behaviors'],
                                 index=0,
                                 horizontal=True,
                                 label_visibility='collapsed')

    _, mid_col, _ = st.columns([0.5, 1.5, 0.5])

    try:
        # Convert the path to be Windows-compatible
        behav_viddir = HERE / 'behavior_videos'
        gif_path = behav_viddir / f"{selected_behavior}.gif"

        # Display GIF if it exists - read as bytes for Windows compatibility
        if gif_path.exists():
            with open(gif_path, 'rb') as f:
                gif_bytes = f.read()
            mid_col.image(gif_bytes)
    except:
        pass  # Silently ignore errors

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE is developed by Justin James and Alexander Hsu; Maintained by Justin James.</h1> "
                    , unsafe_allow_html=True)

# Navigation Logic
page_names = ['Home', 'Preprocessing Workflow', 'Analysis Pipeline', 'LUPE-AMPS']
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title=page_names[0], description="Model description"),
    stx.TabBarItemData(id=2, title=page_names[1], description="Run preprocessing"),
    stx.TabBarItemData(id=3, title=page_names[2], description="Run Analysis"),
    stx.TabBarItemData(id=4, title=page_names[3], description="Pain State Analysis"),
], default=1)

if page_names[int(chosen_id) - 1] == 'Home':
    main()
elif page_names[int(chosen_id) - 1] == 'Preprocessing Workflow':
    preprocess_workflow()
elif page_names[int(chosen_id) - 1] == 'Analysis Pipeline':
    analysis_workflow()
elif page_names[int(chosen_id) - 1] == 'LUPE-AMPS':
    pain_state_model_analysis()
