"""
Shared helpers for per-mouse (per-file) indicator statistics.

Every analysis in utils/analysis_scripts imports from here so that each file
(= one mouse) gets its own CSV with the same five behavior indicators and/or the
same five descriptive statistics, and so that group-level summaries are built
from per-mouse values (n = mice) rather than from pooled frames or bouts.

Behavior indicators (per behavior, per mouse):
    percent_time, total_frames, bout_count, mean_bout_duration_s, bouts_per_min

Descriptives (per metric):
    mean, sd, sem, median, min, max   (+ n)
"""

import os
import warnings
import numpy as np
import pandas as pd

from utils.meta import behavior_names

DESCRIPTIVE_COLS = ["n", "mean", "sd", "sem", "median", "min", "max"]
INDICATOR_COLS = ["percent_time", "total_frames", "bout_count",
                  "mean_bout_duration_s", "bouts_per_min"]


def safe_name(s):
    """Make a string safe for use as a file/folder name on all platforms."""
    s = "unnamed" if s is None else str(s)
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s.rstrip(" .") or "unnamed"


def describe(values):
    """Return the five descriptive statistics (+ n) for a 1D array-like.

    NaNs are ignored. Empty input returns n = 0 and NaN for everything else.
    SD is the sample SD (ddof=1); SEM = SD / sqrt(n).
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return {k: (0 if k == "n" else np.nan) for k in DESCRIPTIVE_COLS}
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "sd": sd,
        "sem": sd / np.sqrt(n) if n > 0 else np.nan,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def parse_bouts(predict):
    """Return (labels, durations_in_frames) for every contiguous bout."""
    predict = np.asarray(predict).astype(int).ravel()
    if predict.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    ends = np.hstack([starts[1:], predict.size])
    return predict[starts], ends - starts


def bout_metrics(predict, fps=60, behaviors=None):
    """Per-behavior indicator stats for a single mouse.

    Returns a DataFrame with one row per behavior and columns:
        behavior, percent_time, total_frames, bout_count, mean_bout_duration_s,
        bouts_per_min, plus bout-duration descriptives
        (bout_duration_sd_s, bout_duration_sem_s, bout_duration_median_s,
         bout_duration_min_s, bout_duration_max_s).
    """
    behaviors = behavior_names if behaviors is None else behaviors
    predict = np.asarray(predict).astype(int).ravel()
    n_frames = int(predict.size)
    minutes = n_frames / fps / 60.0 if n_frames else np.nan
    labels, durs = parse_bouts(predict)

    rows = []
    for b, name in enumerate(behaviors):
        d = durs[labels == b] / fps
        total_frames = int(np.sum(predict == b))
        desc = describe(d)
        rows.append({
            "behavior": name,
            "percent_time": 100.0 * total_frames / n_frames if n_frames else np.nan,
            "total_frames": total_frames,
            "bout_count": int(d.size),
            "mean_bout_duration_s": desc["mean"],
            "bouts_per_min": d.size / minutes if minutes else np.nan,
            "bout_duration_sd_s": desc["sd"],
            "bout_duration_sem_s": desc["sem"],
            "bout_duration_median_s": desc["median"],
            "bout_duration_min_s": desc["min"],
            "bout_duration_max_s": desc["max"],
        })
    return pd.DataFrame(rows)


def per_mouse_dir(analysis_dir, group, condition):
    """<analysis_dir>/per_mouse/<group>/<condition>/ (created if missing)."""
    path = os.path.join(analysis_dir, "per_mouse", safe_name(group), safe_name(condition))
    os.makedirs(path, exist_ok=True)
    return path


def save_per_mouse_csv(df, analysis_dir, group, condition, file_name, suffix=""):
    """Write one CSV for one mouse. Group/condition/file columns are prepended."""
    df = df.copy()
    df.insert(0, "file", file_name)
    df.insert(0, "condition", condition)
    df.insert(0, "group", group)
    out = os.path.join(per_mouse_dir(analysis_dir, group, condition),
                       f"{safe_name(file_name)}{suffix}.csv")
    df.to_csv(out, index=False)
    return out


def summarize_across_mice(per_mouse_df, value_cols, by=("behavior",)):
    """Descriptives across mice for each value column.

    per_mouse_df: long DataFrame with one row per (mouse, `by` key).
    Returns a DataFrame with `by` columns + metric + n/mean/sd/sem/median/min/max.
    """
    by = list(by)
    if per_mouse_df is None or per_mouse_df.empty:
        return pd.DataFrame(columns=by + ["metric"] + DESCRIPTIVE_COLS)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        groups = per_mouse_df.groupby(by, sort=False, observed=True) if by else [((), per_mouse_df)]
        for key, sub in groups:
            key = key if isinstance(key, tuple) else (key,)
            for col in value_cols:
                if col not in sub.columns:
                    continue
                rec = dict(zip(by, key))
                rec["metric"] = col
                rec.update(describe(sub[col].values))
                rows.append(rec)
    return pd.DataFrame(rows)


def save_group_summary(per_mouse_df, value_cols, analysis_dir, group, condition, by=("behavior",)):
    """Write <analysis_dir>/per_mouse/summary_across_mice_<group>-<condition>.csv."""
    summary = summarize_across_mice(per_mouse_df, value_cols, by=by)
    out_dir = os.path.join(analysis_dir, "per_mouse")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"summary_across_mice_{safe_name(group)}-{safe_name(condition)}.csv")
    summary.insert(0, "condition", condition)
    summary.insert(0, "group", group)
    summary.to_csv(out, index=False)
    return summary


def save_figure(fig, svg_path, dpi=300, **kwargs):
    """Save a matplotlib figure as SVG and as a PNG with the same stem.

    svg_path may end in .svg (or have no extension). Returns (svg_path, png_path).
    """
    root, ext = os.path.splitext(str(svg_path))
    svg_path = root + ".svg"
    png_path = root + ".png"
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(svg_path, format="svg", **kwargs)
    fig.savefig(png_path, format="png", dpi=dpi, **kwargs)
    return svg_path, png_path
