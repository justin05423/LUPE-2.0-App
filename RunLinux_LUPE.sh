#!/bin/bash
# =============================================================================
# LUPE Local Launcher (Mac/Linux)
# =============================================================================
# Double-click this file or run from terminal: ./run_lupe.sh
#
# This script:
# 1. Activates the LUPE2APP conda environment
# 2. Launches the Streamlit app in your default browser
# =============================================================================

# Find conda installation
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda installation."
    echo "Please ensure Anaconda or Miniconda is installed."
    exit 1
fi

# Change to the script's directory (where the app lives)
cd "$(dirname "$0")"

# Activate environment and run
echo "Activating LUPE2APP environment..."
conda activate LUPE2APP

if [ $? -ne 0 ]; then
    echo "Error: Could not activate LUPE2APP environment."
    echo "Please run: conda env create -f LUPE2_App.yaml"
    exit 1
fi

echo "Starting LUPE Analysis App..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run lupe_analysis.py
