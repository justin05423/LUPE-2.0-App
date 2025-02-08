# LUPE 2.0 App Walkthrough 🚀  

Welcome to the **LUPE 2.0 App Walkthrough**! 

This guide will take you through each step of using the LUPE 2.0 App, from loading your data to analyzing mouse behavior. Whether you're a first-time user or need a refresher, follow along with the detailed instructions and screenshots.  

### 🛠 **Try the Demo Data!**  
Want to trial-run the full app pipeline? Or jump straight into the LUPE Analysis pipeline with preprocessed files?  
📥 [Download the Demo Data Here](https://upenn.box.com/s/a05itruolxs6z9efy2f22xznzex1ydw0) <sup>[1]</sup>

🔹 Use the raw **LUPE-DeepLabCut** output files to **run the full preprocessing pipeline**.  
🔹 Alternatively, use the **already preprocessed files** to **directly start LUPE Analysis**.  

  <sup>[1]</sup> Demo data runs **LUPE-collected experiments** comparing control, **1%**, and **5% formalin response** in **C57BL/6 mice**.

## 📌 Table of Contents  
- [1️⃣ Setting Up Your Project](#1️⃣-setting-up-your-project)  
- [2️⃣ Reviewing Uploaded Files & Beginning Preprocess Data Directory](#2️⃣-reviewing-uploaded-files--beginning-preprocess-data-directory)  
- [3️⃣ Preprocess Features Directory](#3️⃣-preprocess-features-directory)  
- [4️⃣ Preprocess Behavior Directory](#4️⃣-preprocess-behavior-directory)  
- [5️⃣ Analysis Pipeline Setup](#5️⃣-analysis-pipeline-setup)  
- [📌 NOTE: Analyses with User Inputs](#📌-note-analyses-with-user-inputs)  
  - [Behavior Binned-Ratio Timeline](#behavior-binned-ratio-timeline)  
  - [Behavior Timepoint Comparison](#behavior-timepoint-comparison)  
  - [Behavior Kinematx](#behavior-kinematx)  
- [📌 NOTE: Reloading Previous Projects Already Processed?](#📌-note-reloading-previous-projects-already-processed)  
- [💡 Pro Tips](#💡-pro-tips)  

---

### 1️⃣ Setting Up Your Project  
Now that you have the app opened, let's set up your project! (If not follow link [here](https://github.com/justin05423/LUPE-2.0-App/tree/main?tab=readme-ov-file#installation-guide) to install/open app.

> 1. Navigate to the **“Preprocessing Workflow”** tab.
> 2. In the **main panel**, enter a name for your project in the **“Enter Project Name”** field.
> 3. Use the **left sidebar** to customize your data organization:  
>    - Set the **number of groups** for your data.  
>    - Assign a **name for each group**.  
>    - Specify the **number of conditions** within each group.  
>    - Name each **condition** accordingly.  
> 4. Upload your **LUPE-DeepLabCut CSV output files** by dragging and dropping them into the **“Upload Files”** section.

<details>
  <summary> Walkthrough Snapshot - Naming Project, Groups, and Conditions</summary>

  ![1 Setting Up Your Project](demo_app/1%20Preprocessing_NameProjectGroupsConditions.png)

</details>

### 2️⃣ Reviewing Uploaded Files & Beginning Preprocess Data Directory
> 1. Review the uploaded files to ensure they are correctly organized according to the specified **group** and **condition** structure.  
> 2. Once verified, **begin Step 1 of preprocessing**.
>    - This step will store all the **LUPE-DLC output data** of your project into a designated directory.
> 3. Time for Step 1 of preprocessing depends on amount of loaded, but usually takes **about less than 5 minutes** for a project with about 30 CSV files.

<details>
  <summary> Walkthrough Snapshot - Reviewing Uploaded Files & Starting Preprocessing</summary>

  ![2 Reviewing Uploaded Files & Beginning Preprocess Data Directory](demo_app/2%20Preprocessing_CheckFileDataDirectory_BeginStep1.png)

</details>

### 3️⃣ Preprocess Features Directory
> 1. Once **Step 1 is complete**, proceed to **Step 2** by clicking the **"Run Preprocessing Step 2"** button in the app.  
> 2. This step extracts **various features** from the LUPE-DLC pose data, which are used to predict and classify behaviors in the LUPE model.  
> 3. **Processing Time:** This step typically takes the longest of the three.  
>    - The duration depends on the amount of data in your project.  
>    - For a **modest project size**, expect it to take **between 5-15 minutes**.  

<details>
  <summary> Walkthrough Snapshot - Running Step 2 - Extract Features</summary>

  ![3 Preprocess Features Directory](demo_app/3%20Preprocessing_Step2ExtractFeatures.png)

</details>

<details>
  <summary> Walkthrough Snapshot - Step 2 Extraction Complete</summary>

  ![Step 3 - Extraction Complete](demo_app/4%20Preprocessing_Step2ExtractFeatures_Complete.png)

</details>

### 4️⃣ Preprocess Behavior Directory
> 1. Once **Feature Extraction (Step 2) is complete**, proceed to **Step 3** by clicking **"Run Preprocessing Step 3"**.  
> 2. This final step **predicts behaviors** based on the extracted LUPE-DLC features.  
> 3. After **Step 3 is complete**, preprocessing is finished, and you can begin the **LUPE analysis pipeline**.  

<details>
  <summary> Walkthrough Snapshot - Running Step 3 - Predict Behaviors</summary>

  ![4 Preprocess Behavior Directory](demo_app/5%20Preprocessing_Step3PredictBehaviors.png)

</details>

### 5️⃣ Analysis Pipeline Setup
> 1. Navigate to the **"Analysis Pipeline"** tab.  
> 2. For your selected project, **choose the groups and conditions** to compare in the analysis.  
> 3. If your desired groups/conditions are **not listed as options**, manually enter them in the **sidebar**.  
> 4. Once setup is complete, select the **analysis type** you wish to perform.  
> 5. Click **"Run Analysis"** to execute the selected analysis.  
> 6. The analysis will complete **immediately**, generating:
>    - **SVG images** with graphical results.  
>    - **CSV files** containing the extracted data.
> 7. [🔍 View Demo Data Output](LUPEAPP_processed_dataset/demo_FormalinResponse/figures/)

<details>
  <summary> Walkthrough Snapshot - Quick Setup for Analysis</summary>

  ![Step 5 - Analysis Quick Setup](demo_app/6%20Analysis_QuickSetUp.png)

</details>

<details>
  <summary> Walkthrough Snapshot - Selecting an Analysis Type</summary>

  ![Step 5 - Selecting Analysis](demo_app/7%20Select%20Analysis.png)

</details>

---

### 📌 NOTE: Analyses with User Inputs  

Some analyses require **user input** to define variables associated with the output. Below are the required inputs for specific analyses:

#### *Behavior Binned-Ratio Timeline* 
- Input the **minute interval** you would like behaviors binned by.  
- The output will show the **percentage of each bin occupied by different behaviors**.  

<details>
  <summary> Walkthrough Snapshot - Behavior Binned-Ratio Timeline Input</summary>

  ![Behavior Binned-Ratio Timeline](demo_app/Input1%20Behavior%20Binned-Ratio%20Timeline.png)

</details>

#### *Behavior Timepoint Comparison* 
- First, **input how many timepoints** you want to compare in the video data (e.g., `2`).  
- Then, enter the **minute intervals** corresponding to the timepoints in increasing order up to the total video length  
  - Example: `0-10, 11-30`  

<details>
  <summary> Walkthrough Snapshot - Behavior Timepoint Comparison Input</summary>

  ![Behavior Timepoint Comparison](demo_app/Input2%20Behavior%20Timepoint%20Comparison.png)

</details>

#### *Behavior Kinematx* 
- **Per group**, select the **conditions** to include for analysis.  
- Toggle and select the **body part of interest** to view the **average displacement (pixels/frame)** for that selection.  

<details>
  <summary> Walkthrough Snapshot - Behavior Kinematx Input</summary>

  ![Behavior Kinematx](demo_app/Input3%20Behavior%20Kinematx.png)

</details>

---

### 📌 NOTE: Reloading Previous Projects Already Processed?  

If you've previously processed a project and want to reload it, follow these steps:

> 1. Navigate to the **"Processing Workflow"** tab in the LUPE 2.0 App.  
> 2. **Re-enter the name of your prior project** in the project name field.  
> 3. The app will recognize your project and display:  
>    _"All preprocessing steps are complete! 🚀"_  
> 4. Proceed to the **"Analysis Pipeline"** tab.  
> 5. In the sidebar, **re-enter the project's groups and conditions** for analysis.

<details>
  <summary> Walkthrough Snapshot - Reloading a Project in Processing Workflow</summary>

  ![Reloading Project - Step 1](demo_app/S1a%20Reloading%20Project.png)

</details>

<details>
  <summary> Walkthrough Snapshot - Reloading a Project in Analysis Pipeline</summary>

  ![Reloading Project - Step 2](demo_app/S1b%20Reloading%20Project.png)

</details>

---

### 💡 Pro Tips  
After preprocessing is complete, you can find your **project's processed data** within your **local GitHub folder**:  
📂 **LUPE-2.0-App/LUPEAPP_processed_dataset/**  
- Inside this directory, look for a **folder named after your project** that contains the `.pkl` data files. This processed data will be used for further analysis in **LUPE's behavior classification pipeline**. 🚀  
- This is also where the output of your LUPE pipeline analyses will be found, organized by the analysis name. ✅







