# Changelog

All notable changes to the LUPE 2.0 App are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
