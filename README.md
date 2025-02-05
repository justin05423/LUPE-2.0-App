# LUPE 2.0 App

<p align="center">
<img src="public/logo.png" width="400">
</p>

<p align="center">
Introducing LUPE 2.0 APP, the innovative analysis pipeline predicting pain behavior in mice. 
With our platform, you can input pose and classify mice behavior inside the LUPE Box. 
With in-depth analysis and interactive visuals, as well as downloadable CSVs for easy integration into your existing workflow.

<p align="center">
Try LUPE today and unlock a new level of insights into animal behavior!
</p>

![Annotated Vids](public/annotated_vids_all.gif)

---

# System Requirements
The LUPE-2.0 App requires only a standard computer with enough RAM to support streamlit data analysis and output. 

DeepLabCut<sup>1,2</sup> and A-SOiD<sup>3</sup> were used to create LUPE-2.0 model for pose estimation and behavior classification, respectively. Refer to GitHub of [DLC](https://github.com/DeepLabCut) or [A-SOiD](https://github.com/YttriLab/A-SOID) for further details. 

#### OS Requirements
- This package is supported for *Windows* and *Mac* but can be run on *Linux* computers given additional installation of require packages.

#### Python Dependencies
- For dependencies please refer to the requirements.txt file.

---

# Installation Guide

1. **Clone or Download the Repository**

   Open your terminal and run:
   ```bash
   git clone https://github.com/your_username/LUPE-2.0-App.git
   cd LUPE-2.0-App
   
2. #### Download the LUPE 2.0 A-SOiD Model [HERE](https://upenn.box.com/s/vkdbktwn5krrinalyujapo7htg2ok5rc) and move the contents of the folder into the 'Model' folder.
    > **Note**: Find the LUPE 2.0 DLC Model [HERE](https://upenn.box.com/s/av3i14c64rj6zls9lz6pda0it5b5q7f3) for analyzing pose estimation for LUPE video data.
    
3.	**Create the Conda Environment**
    ```bash
  	conda env create -f LUPE2_App.yaml
  
4. **Activate the LUPE2APP Environment**
    ```bash
  	conda activate LUPE2_App.yaml
  
5. **Run the App 😎**
    ```bash
  	streamlit run lupe_analysis.py

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
