#!/bin/bash
# =============================================================================
# LUPE Analysis App - Mac Launcher
# =============================================================================
# Double-click this file to launch LUPE!
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================"
echo "       LUPE Analysis App Launcher"
echo "============================================"
echo ""

# Find conda installation (check common locations)
CONDA_FOUND=false

for CONDA_PATH in \
    "$HOME/anaconda3" \
    "$HOME/miniconda3" \
    "$HOME/opt/anaconda3" \
    "$HOME/opt/miniconda3" \
    "/opt/anaconda3" \
    "/opt/miniconda3" \
    "/usr/local/anaconda3" \
    "/usr/local/miniconda3" \
    "/opt/homebrew/anaconda3" \
    "/opt/homebrew/Caskroom/miniconda/base"
do
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        echo "Found conda at: $CONDA_PATH"
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        CONDA_FOUND=true
        break
    fi
done

if [ "$CONDA_FOUND" = false ]; then
    echo ""
    echo "ERROR: Could not find conda installation!"
    echo ""
    echo "Please ensure Anaconda or Miniconda is installed."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Press any key to close..."
    read -n 1
    exit 1
fi

# Activate the LUPE environment
echo "Activating LUPE2APP environment..."
conda activate LUPE2APP

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Could not activate LUPE2APP environment!"
    echo ""
    echo "Please create it first by running:"
    echo "  conda env create -f LUPE2_App.yaml"
    echo ""
    echo "Press any key to close..."
    read -n 1
    exit 1
fi

echo ""
echo "Starting LUPE Analysis App..."
echo ""
echo "============================================"
echo "  The app will open in your browser at:"
echo "  http://localhost:8501"
echo "============================================"
echo ""
echo "Keep this window open while using LUPE."
echo "Press Ctrl+C or close this window to stop."
echo ""

# Run the Streamlit app
streamlit run lupe_analysis.py --server.headless=false

# If we get here, the app was closed
echo ""
echo "LUPE has been closed."
echo "Press any key to exit..."
read -n 1
