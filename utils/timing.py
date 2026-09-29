"""
Recording frame rate per project.

The frame rate is stored in the project's own metadata file,
LUPEAPP_processed_dataset/<project>/project_info_<project>.txt, in a "Timing:" section:

    Timing:
      default: fps = 62.5
      overrides:
        GroupA / Cond2: fps = 60

A project with no Timing section uses 60 fps (the historical LUPE assumption), so existing
projects behave exactly as before until someone sets a value.

The classifier's own sampling rate (60 fps in utils/classification.py and utils/preprocess_step2.py)
is a model constant and is NOT affected by anything here. This module only governs the conversion
of frames to seconds and minutes in the analyses.
"""

import os
import re
import pickle
from datetime import datetime

DEFAULT_FPS = 60.0
_TIMING_HEADER = "Timing:"
_CHECK_HEADER = "Timing check"


def project_dir(project_name):
    return os.path.join(".", "LUPEAPP_processed_dataset", project_name)


def project_info_path(project_name):
    return os.path.join(project_dir(project_name), f"project_info_{project_name}.txt")


# --------------------------------------------------------------------------- read / write

def read_timing(project_name):
    """Return {'default_fps': float, 'overrides': {(group, condition): fps}}."""
    timing = {"default_fps": DEFAULT_FPS, "overrides": {}}
    path = project_info_path(project_name)
    if not os.path.exists(path):
        return timing
    in_timing = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith(_TIMING_HEADER):
                in_timing = True
                continue
            if in_timing:
                # Section ends at the next top-level header (non-indented, ends with ':') or blank line
                if not stripped or (not line.startswith((" ", "\t")) and stripped.endswith(":")):
                    in_timing = False
                    continue
                m = re.match(r"^\s*default:\s*fps\s*=\s*([0-9.]+)", line)
                if m:
                    timing["default_fps"] = float(m.group(1))
                    continue
                m = re.match(r"^\s*(.+?)\s*/\s*(.+?):\s*fps\s*=\s*([0-9.]+)", line)
                if m:
                    timing["overrides"][(m.group(1).strip(), m.group(2).strip())] = float(m.group(3))
    return timing


def _strip_sections(text, headers):
    """Remove the named sections (header line through the next blank line) from the file text."""
    lines = text.split("\n")
    out, skipping = [], False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(h) for h in headers):
            skipping = True
            continue
        if skipping:
            if not stripped:
                skipping = False
            continue
        out.append(line)
    # collapse runs of blank lines at the end
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def write_timing(project_name, default_fps, overrides=None, frame_counts=None):
    """Write (or replace) the Timing section in project_info_<project>.txt.

    overrides: {(group, condition): fps}. Entries equal to default_fps are dropped.
    frame_counts: optional {(group, condition): [n_frames, ...]} to write a reference block.
    Creates a minimal project_info file if none exists.
    """
    overrides = {k: float(v) for k, v in (overrides or {}).items()
                 if v is not None and abs(float(v) - float(default_fps)) > 1e-9}
    path = project_info_path(project_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = f"Project: {project_name}\nCreated/Updated: {datetime.now().isoformat()}"
    text = _strip_sections(text, [_TIMING_HEADER, _CHECK_HEADER])

    block = [f"\n\n{_TIMING_HEADER}", f"  default: fps = {float(default_fps):g}"]
    if overrides:
        block.append("  overrides:")
        for (g, c), v in sorted(overrides.items()):
            block.append(f"    {g} / {c}: fps = {v:g}")

    if frame_counts:
        block.append(f"\n{_CHECK_HEADER} (auto-generated {datetime.now().isoformat(timespec='seconds')}):")
        timing = {"default_fps": float(default_fps), "overrides": overrides}
        for (g, c), counts in sorted(frame_counts.items()):
            if not counts:
                continue
            fps = get_fps(timing, g, c)
            block.append(f"  {g} / {c}: {describe_length(counts, fps)}")
            if isinstance(counts, dict):
                for ln in outlier_lines(counts, fps, max_name=200):
                    block.append(f"      {ln}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n".join(block) + "\n")
    return path


# --------------------------------------------------------------------------- lookup

def get_fps(timing, group=None, condition=None):
    """fps for a group/condition given a timing dict from read_timing()."""
    if timing is None:
        return DEFAULT_FPS
    return float(timing.get("overrides", {}).get((group, condition), timing.get("default_fps", DEFAULT_FPS)))


def fps_lookup(project_name):
    """Return a callable fps(group, condition) bound to the project's saved timing."""
    timing = read_timing(project_name)
    return lambda group=None, condition=None: get_fps(timing, group, condition)


def constant_fps(fps=DEFAULT_FPS):
    """A lookup that always returns one value (used as the default in analysis scripts)."""
    fps = float(fps)
    return lambda group=None, condition=None: fps


# --------------------------------------------------------------------------- frame counts

def frame_counts(project_name):
    """{(group, condition): [n_frames per file]} from behaviors pkl, else raw_data pkl, else {}."""
    base = project_dir(project_name)
    for name in (f"behaviors_{project_name}.pkl", f"raw_data_{project_name}.pkl"):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        counts = {}
        for g, conds in data.items():
            for c, files in conds.items():
                counts[(g, c)] = {fn: len(v) for fn, v in files.items()}
        if counts:
            return counts
    return {}


def describe_length(counts, fps):
    """One-line human summary: '20 files, 112,500 frames each -> 30.00 min at 62.5 fps'."""
    counts = [int(n) for n in (counts.values() if isinstance(counts, dict) else counts) if n]
    if not counts:
        return "no files"
    lo, hi = min(counts), max(counts)
    lo_min, hi_min = lo / fps / 60.0, hi / fps / 60.0
    if hi - lo <= max(2, 0.002 * hi):  # essentially identical lengths
        return f"{len(counts)} files, {hi:,} frames each -> {hi_min:.2f} min at {fps:g} fps"
    return (f"{len(counts)} files, {lo:,} to {hi:,} frames -> "
            f"{lo_min:.2f} to {hi_min:.2f} min at {fps:g} fps")


def length_outliers(counts_by_file, fps, tol=0.005):
    """Files whose frame count is more than `tol` (fraction) away from the most common count.

    Returns (reference_count, n_at_reference, [(file, n_frames, minutes), ...]) sorted by frame count.
    counts_by_file: {file_name: n_frames}.
    """
    if not counts_by_file:
        return None, 0, []
    vals = list(counts_by_file.values())
    # Reference = the frame count with the most files within +/-2 frames of it
    ref = max(set(vals), key=lambda v: sum(abs(x - v) <= 2 for x in vals))
    out = [(f, n, n / fps / 60.0) for f, n in counts_by_file.items() if abs(n - ref) > tol * ref]
    n_ref = len(vals) - len(out)
    return ref, n_ref, sorted(out, key=lambda t: t[1])


def outlier_lines(counts_by_file, fps, max_name=48):
    """Human lines for length_outliers(); empty list when there are none."""
    ref, n_ref, out = length_outliers(counts_by_file, fps)
    if not out:
        return []
    shorter = [o for o in out if o[1] < ref]
    longer = [o for o in out if o[1] > ref]
    head = f"{n_ref} files at about {ref:,} frames"
    parts = []
    if shorter:
        parts.append(f"{len(shorter)} shorter")
    if longer:
        parts.append(f"{len(longer)} longer")
    lines = [f"{head}; {' and '.join(parts)}:"]
    for f, n, m in out:
        name = f if len(f) <= max_name else f[:max_name] + "…"
        lines.append(f"{name}  {n:,} frames ({m:.2f} min)")
    return lines


# --------------------------------------------------------------------------- streamlit widget

def timing_widget(st, project_name, key_prefix="timing", show_save=True):
    """Draw the frame-rate input, live length line, optional overrides, and Save button.

    Returns (default_fps, overrides) as currently entered (saved or not).
    """
    timing = read_timing(project_name)
    counts = frame_counts(project_name)

    default_fps = st.number_input(
        "Recording frame rate (fps)",
        min_value=1.0, max_value=1000.0, value=float(timing["default_fps"]), step=0.5, format="%.2f",
        key=f"{key_prefix}_default_fps",
        help="Frames per second your camera actually recorded. If your camera was set to a fixed frame "
             "interval (for example one frame every 16 ms), the true rate is 1000 / interval (62.5 fps), "
             "not the nominal 60. Every time-based metric (bouts/min, seconds, minute bins) uses this value.",
    )

    # Length lines are filled in after the overrides are read, so they reflect current entries
    lines_slot = st.container()

    overrides = dict(timing["overrides"])
    keys = sorted(counts.keys()) if counts else sorted(overrides.keys())
    # A checkbox rather than an expander, so the widget can itself sit inside an expander
    show_overrides = st.checkbox("Some groups or conditions were recorded at a different frame rate",
                                 value=bool(overrides), key=f"{key_prefix}_show_overrides")
    if show_overrides:
        if not keys:
            st.caption("Group/condition overrides become available after preprocessing step 1.")
        for (g, c) in keys:
            col1, col2 = st.columns([1, 1])
            is_diff = col1.checkbox(f"{g} / {c}", value=(g, c) in overrides,
                                    key=f"{key_prefix}_diff_{g}_{c}")
            if is_diff:
                val = col2.number_input("fps", min_value=1.0, max_value=1000.0,
                                        value=float(overrides.get((g, c), default_fps)),
                                        step=0.5, format="%.2f", key=f"{key_prefix}_ovr_{g}_{c}",
                                        label_visibility="collapsed")
                overrides[(g, c)] = val
            else:
                overrides.pop((g, c), None)
    if not show_overrides:
        overrides = {}
    overrides = {k: v for k, v in overrides.items() if abs(v - default_fps) > 1e-9}

    current = {"default_fps": default_fps, "overrides": overrides}
    with lines_slot:
        if counts:
            lines = []
            for (g, c), n in sorted(counts.items()):
                fps_gc = get_fps(current, g, c)
                lines.append(f"- **{g} / {c}**: {describe_length(n, fps_gc)}")
                for ln in outlier_lines(n, fps_gc):
                    lines.append(f"    - <small>{ln}</small>")
            st.markdown("At this frame rate your recordings are:\n" + "\n".join(lines), unsafe_allow_html=True)
            st.caption("Check these against your protocol time (how long you actually recorded, not the duration a "
                       "video player reports). If every file in a group reads longer or shorter than your protocol, "
                       "the frame rate is wrong; correct it above. Files listed individually as shorter or longer "
                       "were recorded for a different time at the same rate; do not change the frame rate for them.")
        else:
            st.caption("Recording lengths will be shown here once files have been preprocessed.")

    if show_save and st.button("Save frame rate", key=f"{key_prefix}_save"):
        write_timing(project_name, default_fps, overrides, frame_counts=counts)
        st.success(f"Saved: {default_fps:g} fps" + (f" with {len(overrides)} override(s)" if overrides else "")
                   + f" -> {project_info_path(project_name)}")
        st.info("Re-run Behavior CSV Classification and any analyses you need. "
                "Outputs already in figures/ were computed at the previous frame rate.")

    return default_fps, overrides


def timing_summary_line(project_name):
    """Short one-liner for display in the Analysis tab."""
    timing = read_timing(project_name)
    s = f"{timing['default_fps']:g} fps"
    if timing["overrides"]:
        s += " (" + ", ".join(f"{g}/{c}: {v:g}" for (g, c), v in sorted(timing["overrides"].items())) + ")"
    return s
