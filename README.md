# LUPE 2.0 App

> 🔬 **Looking for the full analysis pipeline in Jupyter Notebooks?**  
Head over to the [LUPE 2.0 Analysis Package](https://github.com/justin05423/LUPE-2.0-AnalysisPackage). That repository provides the core analysis pipeline, reproducible Jupyter notebooks, and scripts to extend LUPE for advanced users who want direct control over the workflow.

<p align="center">
<img src="public/logo.png" width="400">
</p>

<p align="center">
Introducing LUPE 2.0 APP, the innovative analysis pipeline predicting pain behavior in mice.  
With our platform, you can input pose and classify mice behavior inside the LUPE Box.  
With in-depth analysis and interactive visuals, as well as downloadable CSVs for easy integration into your existing workflow.
</p>

<p align="center">
Try LUPE today and unlock a new level of insights into animal behavior!
</p>

## Table of Contents
- [System Requirements](#system-requirements)
- [Local Installation Guide](#local-installation-guide)
- [Updating App](#updating-lupe-app)
- [App Guide](#app-guide)
- [Physical System Build](#physical-system-build)
- [Contacting](#contacting)

![Annotated Vids](public/annotated_vids_all.gif)

---

# System Requirements

The LUPE-2.0 App requires only a standard computer with enough RAM to support Streamlit-based data analysis and interactive visualizations.

LUPE uses:  
- [DeepLabCut](https://github.com/DeepLabCut)<sup>1,2</sup> for pose estimation  
- [A-SOiD](https://github.com/YttriLab/A-SOID)<sup>3</sup> for behavior classification  

These models are pre-trained and integrated into the app.  
👉 **Be sure to follow [Step 2 in the Local Installation Guide]** to properly obtain and place the LUPE-A-SOiD model before running the app.

> 💡 **Recommended Setup**  
> We recommend installing all dependencies using [Anaconda](https://www.anaconda.com/products/distribution), a package and environment manager that simplifies Python project setup and avoids conflicts.

### OS Requirements

- ✅ **Windows** – fully supported  
- ✅ **macOS** – fully supported  
- ⚠️ **Linux** – supported with manual installation of certain packages

### Python Dependencies

- Please refer to the `requirements.txt` file for all necessary Python libraries.

---

# Local Installation Guide

To run “LUPE-2.0-App” on your own computer, follow these steps:

1. **Clone or Download the Repository**  
   A repository is just the project’s folder on GitHub. You need a copy of that folder on your computer. **Clone** uses Git to make a local copy that stays linked to GitHub, so you can get updates later with `git pull`. **OR** grab a one-time ZIP of the files by clicking **[HERE](https://github.com/justin05423/LUPE-2.0-App/archive/refs/heads/main.zip)**. It’s not linked, so you won’t get updates unless you download again.

   Open your terminal / command prompt and run:
   ```bash
   git clone https://github.com/justin05423/LUPE-2.0-App.git
   ```
   Then set the directory of your terminal/command prompt to this folder:
   ```bash
   cd LUPE-2.0-App
   ```

2. **Download the LUPE 2.0 A-SOiD Model** (If not already present in 'model' folder when repository is downloaded/cloned.) 
   Download from [HERE](https://upenn.box.com/s/9rfslrvcc7m6fji8bmgktnegghyu88b0) and move the contents of the folder into the `Model` folder, found in the now locally downloaded LUPE 2.0 App GitHub folder.  
   > **Note**: For analyzing and retrieving pose estimation for LUPE video data, find the LUPE 2.0 DLC Model [HERE](https://upenn.box.com/s/av3i14c64rj6zls9lz6pda0it5b5q7f3).

3. **Create the Conda Environment**

   For *macOS*:
   ```bash
   conda env create -f LUPE2_App.yaml
   ```

   For *Windows*:
   ```bash
   conda env create -f LUPE2_App_Win.yaml
   ```

4. **Activate the LUPE2APP Environment**  
   In your terminal/command prompt, ensure you are in the `LUPE-2.0-App` directory, then run:
   ```bash
   conda activate LUPE2APP
   ```

5. **Run the App 😎**  
   ```bash
   streamlit run lupe_analysis.py
   ```

---

# Updating LUPE App

To update your local copy of LUPE-2.0-App to the latest version, follow these steps:

1. Open your terminal or command prompt and navigate to the `LUPE-2.0-App` directory:
   ```bash
   cd LUPE-2.0-App
   ```
2. Pull the latest changes from the GitHub repository:
   ```bash
   git pull origin main
   ```
3. If there are updates to the Conda environment files (`LUPE2_App.yaml` or `LUPE2_App_Win.yaml`), update your environment accordingly:
   ```bash
   conda env update -f LUPE2_App.yaml --prune
   ```
   or for Windows:
   ```bash
   conda env update -f LUPE2_App_Win.yaml --prune
   ```
4. Restart your environment and run the app as usual.

---

# App Guide

For a detailed walkthrough on using the LUPE 2.0 App, check out the [App Walkthrough](https://github.com/justin05423/LUPE-2.0-App/wiki/LUPE-2.0-App-Walkthrough--%F0%9F%9A%80).

---

# Physical System Build

For an overview on building the LUPE 2.0 System, check out the [Build](https://github.com/justin05423/LUPE-2.0-App/wiki/LUPE-2.0-Build-%F0%9F%9B%A0%EF%B8%8F-%F0%9F%A7%B0).

---

# Contacting

#### Project Funding  
Collaboration between [Corder Lab](https://corderlab.com/) at University of Pennsylvania and  
[Yttri Lab](https://labs.bio.cmu.edu/yttri/) from Carnegie Mellon.

#### Contributors  
Justin James (Corder Lab) actively develops and maintains this repository/cloud resource.

Other contributors include: Alexander Hsu (Yttri Lab).

---

# License

LUPE is released under a Clear BSD License and is intended for research/academic use only.

# References

1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)  
2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)  
3. [Tillmann JF, Hsu AI, Schwarz MK, Yttri EA. A-SOiD, an active-learning platform for expert-guided, data-efficient discovery of behavior. Nat Methods. 2024 Apr;21(4):703-711. doi: 10.1038/s41592-024-02200-1. Epub 2024 Feb 21. PMID: 38383746.](https://www.nature.com/articles/s41592-024-02200-1)
