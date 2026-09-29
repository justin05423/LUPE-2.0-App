# Changelog

All notable changes to the LUPE 2.0 App are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.2.1] - 2026-09-29

### Added
- README: "Pose Estimation: LUPE 2.0 DeepLabCut Model + GUIs" section and Table of Contents entry, linking the
  Box download and the new wiki page. The install-step note now points there too.
- `images/dlc/`: pipeline overview, model-creation figure, 20-body-point legend and video-to-AMPS workflow figure used by the wiki pages.
- Wiki: "LUPE 2.0 DeepLabCut Model and GUIs" page (model breakdown, TensorFlow vs PyTorch, GPU compatibility,
  DeepLabCut setup, LUPE Single-Animal and Multi-Chamber GUI guide, troubleshooting, references).

### Changed
- Wiki Home: "Start Here" row for pose estimation; App Walkthrough: 60 fps wording replaced with the per-project
  Recording Frame Rate introduced in 2.2.0.

## [2.2.0] - 2026-09-28

### Added
- Per-project recording frame rate. Set once on the Preprocessing tab (or edit later from the Analysis tab);
  stored in `project_info_<project>.txt` under a `Timing:` section. The app shows the resulting recording
  length per group/condition live ("112,500 frames each -> 30.00 min at 62.5 fps") so a wrong value is
  obvious before any analysis runs. Optional per group/condition overrides for mixed-rate projects.
- `utils/timing.py`: read/write of the Timing section, `fps_lookup(project)`, and the Streamlit widget.
- Every per-mouse CSV now carries an `fps_used` column.
- The frame-rate widget lists files whose length differs from the rest of their group/condition (e.g. a session that ended early).
- README: "Recording frame rate" section explaining fixed-interval cameras (one frame every 16 ms = 62.5 fps, not 60).

### Changed
- One figure per group for Bout Counts, Bout Durations, Transitions, Location and Total Frames (file names now end in `_<group>`), so any number of groups and conditions stays readable. Bout Counts and Bout Durations stack conditions vertically with a shared x-axis for comparison down the column; Transitions, Total Frames and Location use a balanced grid of up to 4 panels across (5 conditions = 3 + 2, 10 = 4 + 4 + 2). Shared helpers `panel_grid` and `group_figures` in `per_mouse_stats.py`. Previously a fixed figure size squeezed many conditions into one row.
- Bout Counts: behaviors now read top-down in the same order as Bout Durations; titles include n (mice); house style (light spines, x grid).
- Transitions titles include n (mice). Bout Counts, Bout Durations and Transitions label both axes on every panel (ticks and labels are no longer hidden on shared axes).
- All time-based metrics (bouts/min, bout durations, per-second CSVs, minute bins, binned-ratio bins,
  timepoint windows, distance/min, speed, kinematx displacement) use the project frame rate instead of a
  hard-coded 60. Projects without a Timing section still use 60, so existing results are unchanged.
- Minute bins in Binned Mouse Screening and Binned-Ratio Timeline are computed from a float frame rate
  (no truncation at non-integer fps).
- LUPE-AMPS receives the project frame rate as `sampling_rate`; a warning is shown if selected
  groups/conditions have different overrides (AMPS pools at the project default).
- The classifier's own 60 fps (feature extraction) is unchanged and documented as a model constant.

### Fixed
- Behavior Timepoint Comparison computes time from the row index at the current frame rate, so per-second CSVs written at an earlier rate cannot shift the windows.
- `st.image` compatibility across Streamlit versions (`width="stretch"`, `use_container_width`, `use_column_width`).

## [2.1.0] - 2026-09-28

### Added
- `utils/analysis_scripts/per_mouse_stats.py`: shared helpers for per-mouse statistics
  (`bout_metrics`, `describe`, `save_per_mouse_csv`, `save_group_summary`, `save_figure`).
- Every analysis now writes one CSV per mouse/file to `figures/<analysis>/per_mouse/<group>/<condition>/`
  containing, depending on the analysis, the five behavior indicators (percent time, total frames,
  bout count, mean bout duration, bouts/min) and/or descriptive statistics (mean, SD, SEM, median, min, max).
- Every analysis now writes `per_mouse/summary_across_mice_<group>-<condition>.csv`, computed from the
  per-mouse values (n = mice).
- Every figure is saved as PNG (300 dpi) alongside the existing SVG.
- Behavior Location: optional per-animal image (six behavior panels) via a checkbox.
- Behavior Transitions: optional per-animal transition heatmap via a checkbox.
- Behavior Bout Durations: per-mouse median dots overlaid on the boxplots; a log-scale copy in
  `log_scale/`; a companion bar chart of mean bout duration (mean +/- SEM across mice).
- Behavior Binned Mouse Screening: `mouse_id_key.csv` mapping figure labels to full file names;
  in-app Group/Condition viewer; adjustable label length.

### Changed
- Behavior Bout Counts: chart now plots bouts per minute, normalized by each file's own duration,
  so videos of different lengths are comparable (was raw counts labeled "/30 min"). Group CSV reports
  descriptives for both bouts/min and raw counts.
- Behavior Bout Durations: x-axis is shared across panels and scaled to the largest whisker (was a fixed
  0 to 6 s); boxes drawn with matplotlib (no seaborn dependency); house style matched to other figures.
- Behavior Binned Mouse Screening: one heatmap per behavior per group-condition under
  `<group>/<condition>/` (was a single heatmap pooling all mice with full DLC file names). Colors share a
  per-behavior maximum unless a fixed maximum is set.
- Behavior Timepoint Comparison: outputs reorganized to `per_mouse/<group>/<condition>/`; adds Total Frames
  and Bout Count to the per-window indicators; summaries are per group-condition.
- Distance Traveled: distance calculation vectorized; group CSV adds median, min, max and n.
- Group-level SD is now the sample SD (ddof = 1) in all analyses.
- Analysis descriptions in the app rewritten to state outputs (SVG + PNG, per-mouse CSVs, prerequisites).

### Fixed
- Behavior Timepoint Comparison cohort summaries were never written when condition names did not appear
  in file names, and pooled all conditions per group. Summaries are now built in memory per group-condition.
- Screening heatmap titles lowercased the L/R suffix of licking behaviors.
- `st.image` compatibility with older Streamlit versions.

### Removed
- Behavior Timepoint Comparison: flat `analysis_<file>.csv` files and the `cohort_summaries/` folder.
- Compiled `__pycache__` files are no longer tracked; `.gitignore` expanded.
