# ============================================================
# PROFILOMETER PROFILE ALIGNMENT + AVERAGING APP
# ============================================================
#
# Features:
#   - Upload multiple profilometer CSV files
#   - Automatically identify Distance / Height columns
#   - Individual X shift
#   - Individual Y shift
#   - Individual left/right edge trimming
#   - Robust spike removal:
#       * Off
#       * High spikes only
#       * Low spikes only
#       * High + low spikes
#   - Spike points are displayed in red
#   - Common overlap averaging
#   - Full average profile
#   - Final independent trimming of average profile
#   - Ra, Rq, Rt metrics
#   - Export corrected profiles
#   - Export aligned profiles
#   - Export full average
#   - Export final truncated average
#   - Export metrics
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from io import BytesIO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Profilometer Profile Averaging",
    layout="wide"
)

st.title("Profilometer Profile Alignment & Averaging")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_header_row(file_bytes):
    """
    Try to locate the header row containing Distance / Height.
    """

    try:
        preview = pd.read_csv(
            BytesIO(file_bytes),
            header=None,
            nrows=30,
            encoding_errors="ignore"
        )
    except Exception:
        return 0

    for i in range(len(preview)):

        row = preview.iloc[i].astype(str).str.lower()

        has_distance = row.str.contains(
            "distance|dist|x",
            regex=True
        ).any()

        has_height = row.str.contains(
            "height|z|roughness",
            regex=True
        ).any()

        if has_distance and has_height:
            return i

    return 0


def load_profile(uploaded_file):
    """
    Load a profilometer CSV and identify Distance / Height columns.
    """

    file_bytes = uploaded_file.getvalue()

    header_row = find_header_row(file_bytes)

    df = pd.read_csv(
        BytesIO(file_bytes),
        header=header_row,
        encoding_errors="ignore"
    )

    # Clean column names
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Find X column
    # --------------------------------------------------------

    x_candidates = [
        c for c in df.columns
        if any(
            key in str(c).lower()
            for key in [
                "distance",
                "dist",
                "position",
                "length"
            ]
        )
    ]

    # --------------------------------------------------------
    # Find Z column
    # --------------------------------------------------------

    z_candidates = [
        c for c in df.columns
        if any(
            key in str(c).lower()
            for key in [
                "height",
                "roughness",
                "profile",
                "elevation"
            ]
        )
    ]

    # If automatic detection fails, use numeric columns
    if len(x_candidates) == 0 or len(z_candidates) == 0:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) >= 2:

            x_col = numeric_cols[0]
            z_col = numeric_cols[1]

        else:
            raise ValueError(
                f"Could not identify Distance/Height columns "
                f"in {uploaded_file.name}"
            )

    else:

        x_col = x_candidates[0]
        z_col = z_candidates[0]

    # Convert to numeric
    x = pd.to_numeric(
        df[x_col],
        errors="coerce"
    )

    z = pd.to_numeric(
        df[z_col],
        errors="coerce"
    )

    clean = pd.DataFrame({
        "Distance": x,
        "Height": z
    })

    clean = clean.dropna()

    # Remove duplicate X values
    clean = clean.drop_duplicates(
        subset="Distance"
    )

    # Sort by distance
    clean = clean.sort_values(
        "Distance"
    )

    return clean.reset_index(drop=True)


# ============================================================
# SPIKE REMOVAL
# ============================================================

def remove_spikes(
    distance,
    height,
    window,
    threshold,
    mode
):
    """
    Robust local-median spike removal.

    Modes:
        Off
        High spikes only
        Low spikes only
        High + low spikes

    A spike is identified relative to a local median profile.

    MAD-based sigma is used instead of ordinary standard deviation
    so that extreme spikes do not strongly influence the threshold.
    """

    height = np.asarray(
        height,
        dtype=float
    )

    n = len(height)

    # --------------------------------------------------------
    # Nothing to do
    # --------------------------------------------------------

    if mode == "Off":

        return (
            height.copy(),
            np.zeros(n, dtype=bool)
        )

    if n < 5:

        return (
            height.copy(),
            np.zeros(n, dtype=bool)
        )

    # --------------------------------------------------------
    # Make window odd
    # --------------------------------------------------------

    window = int(window)

    if window < 3:
        window = 3

    if window % 2 == 0:
        window += 1

    # Do not allow window to exceed profile length
    if window > n:

        window = n

        if window % 2 == 0:
            window -= 1

    if window < 3:

        return (
            height.copy(),
            np.zeros(n, dtype=bool)
        )

    # --------------------------------------------------------
    # Local median
    #
    # reflect prevents artificial zero-padding at boundaries
    # --------------------------------------------------------

    median_profile = median_filter(
        height,
        size=window,
        mode="reflect"
    )

    # --------------------------------------------------------
    # Residual relative to local median
    # --------------------------------------------------------

    residual = height - median_profile

    # --------------------------------------------------------
    # Robust MAD estimate
    # --------------------------------------------------------

    median_residual = np.median(
        residual
    )

    mad = np.median(
        np.abs(
            residual - median_residual
        )
    )

    if mad > 0:

        sigma = 1.4826 * mad

    else:

        sigma = np.std(
            residual
        )

    # --------------------------------------------------------
    # If profile is essentially flat
    # --------------------------------------------------------

    if sigma == 0 or not np.isfinite(sigma):

        spike_mask = np.zeros(
            n,
            dtype=bool
        )

    else:

        # ----------------------------------------------------
        # HIGH SPIKES ONLY
        # ----------------------------------------------------

        if mode == "High spikes only":

            spike_mask = (
                residual >
                threshold * sigma
            )

        # ----------------------------------------------------
        # LOW SPIKES ONLY
        # ----------------------------------------------------

        elif mode == "Low spikes only":

            spike_mask = (
                residual <
                -threshold * sigma
            )

        # ----------------------------------------------------
        # BOTH DIRECTIONS
        # ----------------------------------------------------

        elif mode == "High + low spikes":

            spike_mask = (
                np.abs(residual) >
                threshold * sigma
            )

        else:

            spike_mask = np.zeros(
                n,
                dtype=bool
            )

    # --------------------------------------------------------
    # Replace spikes with local median
    # --------------------------------------------------------

    cleaned = height.copy()

    cleaned[spike_mask] = (
        median_profile[spike_mask]
    )

    return cleaned, spike_mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(x, z):

    if len(z) == 0:

        return {
            "Ra": np.nan,
            "Rq": np.nan,
            "Rt": np.nan
        }

    # Remove mean line
    centered = z - np.mean(z)

    Ra = np.mean(
        np.abs(centered)
    )

    Rq = np.sqrt(
        np.mean(centered ** 2)
    )

    Rt = (
        np.max(centered)
        -
        np.min(centered)
    )

    return {
        "Ra": Ra,
        "Rq": Rq,
        "Rt": Rt
    }


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_profiles(
    profiles,
    spacing=None
):
    """
    Interpolate all corrected profiles onto a common X grid.
    """

    if len(profiles) == 0:
        return None, None

    # --------------------------------------------------------
    # Determine common overlap
    # --------------------------------------------------------

    xmin = max(
        profile["Distance"].min()
        for profile in profiles
    )

    xmax = min(
        profile["Distance"].max()
        for profile in profiles
    )

    if xmin >= xmax:

        raise ValueError(
            "The profiles do not have a common "
            "overlapping distance range."
        )

    # --------------------------------------------------------
    # Determine spacing
    # --------------------------------------------------------

    if spacing is None:

        spacings = []

        for profile in profiles:

            dx = np.diff(
                profile["Distance"].values
            )

            dx = dx[
                np.isfinite(dx)
                & (dx > 0)
            ]

            if len(dx) > 0:

                spacings.append(
                    np.median(dx)
                )

        if len(spacings) == 0:

            spacing = (
                xmax - xmin
            ) / 1000

        else:

            spacing = np.median(
                spacings
            )

    # --------------------------------------------------------
    # Common grid
    # --------------------------------------------------------

    common_x = np.arange(
        xmin,
        xmax + spacing,
        spacing
    )

    aligned = []

    for profile in profiles:

        x = profile["Distance"].values
        z = profile["Height"].values

        z_interp = np.interp(
            common_x,
            x,
            z
        )

        aligned.append(
            z_interp
        )

    aligned = np.asarray(
        aligned
    )

    return common_x, aligned


# ============================================================
# SESSION STATE
# ============================================================

if "profiles" not in st.session_state:
    st.session_state.profiles = {}

if "results" not in st.session_state:
    st.session_state.results = None


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload Profiles")

uploaded_files = st.file_uploader(
    "Upload one or more profilometer CSV files",
    type=["csv"],
    accept_multiple_files=True
)


if uploaded_files:

    # Load newly uploaded files
    for uploaded_file in uploaded_files:

        if uploaded_file.name not in st.session_state.profiles:

            try:

                df = load_profile(
                    uploaded_file
                )

                st.session_state.profiles[
                    uploaded_file.name
                ] = {
                    "raw": df
                }

            except Exception as e:

                st.error(
                    f"Could not load "
                    f"{uploaded_file.name}: {e}"
                )


# ============================================================
# STOP IF NO DATA
# ============================================================

if len(st.session_state.profiles) == 0:

    st.info(
        "Upload your profilometer CSV files to begin."
    )

    st.stop()


# ============================================================
# PROFILE SETTINGS
# ============================================================

st.header("2. Individual Profile Settings")

st.write(
    "Adjust each profile independently before averaging. "
    "Left/right cuts are applied BEFORE averaging."
)


profile_names = list(
    st.session_state.profiles.keys()
)


settings = {}


for i, name in enumerate(profile_names):

    raw = (
        st.session_state.profiles[name]["raw"]
    )

    st.subheader(
        f"Profile {i + 1}: {name}"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # X SHIFT
    # --------------------------------------------------------

    with col1:

        x_shift = st.number_input(
            "X shift",
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"xshift_{name}"
        )

    # --------------------------------------------------------
    # Y SHIFT
    # --------------------------------------------------------

    with col2:

        y_shift = st.number_input(
            "Y shift",
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"yshift_{name}"
        )

    # --------------------------------------------------------
    # LEFT / RIGHT TRIM
    # --------------------------------------------------------

    with col3:

        st.write("Edge trimming")

        c1, c2 = st.columns(2)

        with c1:

            left_cut = st.number_input(
                "Left cut",
                value=0.0,
                min_value=0.0,
                step=0.01,
                format="%.4f",
                key=f"leftcut_{name}"
            )

        with c2:

            right_cut = st.number_input(
                "Right cut",
                value=0.0,
                min_value=0.0,
                step=0.01,
                format="%.4f",
                key=f"rightcut_{name}"
            )

    # --------------------------------------------------------
    # SPIKE SETTINGS
    # --------------------------------------------------------

    st.markdown("**Spike removal**")

    spike_col1, spike_col2, spike_col3 = st.columns(3)

    with spike_col1:

        spike_mode = st.selectbox(
            "Spike mode",
            [
                "Off",
                "High spikes only",
                "Low spikes only",
                "High + low spikes"
            ],
            index=1,
            key=f"spike_mode_{name}"
        )

    with spike_col2:

        spike_window = st.slider(
            "Spike window",
            min_value=3,
            max_value=51,
            value=7,
            step=2,
            key=f"spike_window_{name}",
            help=(
                "Number of points used to determine "
                "the local median. Larger values "
                "remove broader features."
            )
        )

    with spike_col3:

        spike_threshold = st.slider(
            "Spike threshold",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key=f"spike_threshold_{name}",
            help=(
                "Lower values remove more points. "
                "Higher values are more conservative."
            )
        )

    settings[name] = {
        "x_shift": x_shift,
        "y_shift": y_shift,
        "left_cut": left_cut,
        "right_cut": right_cut,
        "spike_mode": spike_mode,
        "spike_window": spike_window,
        "spike_threshold": spike_threshold
    }


# ============================================================
# PROCESS INDIVIDUAL PROFILES
# ============================================================

st.header("3. Corrected Profiles")

corrected_profiles = []

spike_summary = []


for name in profile_names:

    raw = (
        st.session_state.profiles[name]["raw"]
        .copy()
    )

    s = settings[name]

    x = raw["Distance"].values.copy()
    z = raw["Height"].values.copy()

    # --------------------------------------------------------
    # X SHIFT
    # --------------------------------------------------------

    x = x + s["x_shift"]

    # --------------------------------------------------------
    # Y SHIFT
    # --------------------------------------------------------

    z = z + s["y_shift"]

    # --------------------------------------------------------
    # INDIVIDUAL EDGE TRIMMING
    #
    # This occurs BEFORE spike removal and averaging.
    # --------------------------------------------------------

    xmin = x.min() + s["left_cut"]
    xmax = x.max() - s["right_cut"]

    mask = (
        (x >= xmin)
        &
        (x <= xmax)
    )

    x = x[mask]
    z = z[mask]

    # --------------------------------------------------------
    # SPIKE REMOVAL
    # --------------------------------------------------------

    z_clean, spike_mask = remove_spikes(
        x,
        z,
        s["spike_window"],
        s["spike_threshold"],
        s["spike_mode"]
    )

    num_spikes = int(
        np.sum(spike_mask)
    )

    spike_summary.append({
        "Profile": name,
        "Spike mode": s["spike_mode"],
        "Spikes removed": num_spikes,
        "Window": s["spike_window"],
        "Threshold": s["spike_threshold"]
    })

    # --------------------------------------------------------
    # Store corrected profile
    # --------------------------------------------------------

    corrected = pd.DataFrame({
        "Distance": x,
        "Height": z_clean,
        "OriginalHeight": z,
        "SpikeRemoved": spike_mask
    })

    corrected_profiles.append(
        corrected
    )

    st.write(
        f"**{name}:** "
        f"{len(x):,} points remaining | "
        f"{num_spikes:,} spikes removed"
    )

    # --------------------------------------------------------
    # Plot corrected profile
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 4)
    )

    ax.plot(
        x,
        z,
        linewidth=0.8,
        label="Original / trimmed"
    )

    ax.plot(
        x,
        z_clean,
        linewidth=1.2,
        label="Corrected"
    )

    # Highlight removed spikes
    if num_spikes > 0:

        ax.scatter(
            x[spike_mask],
            z[spike_mask],
            s=18,
            label="Removed spikes",
            zorder=5
        )

    ax.set_xlabel(
        "Distance"
    )

    ax.set_ylabel(
        "Height"
    )

    ax.set_title(
        f"Corrected Profile: {name}"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# SPIKE SUMMARY
# ============================================================

st.subheader("Spike Removal Summary")

spike_df = pd.DataFrame(
    spike_summary
)

st.dataframe(
    spike_df,
    use_container_width=True
)


# ============================================================
# AVERAGING
# ============================================================

st.header("4. Average Profiles")

try:

    common_x, aligned = interpolate_profiles(
        corrected_profiles
    )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    average_height = np.mean(
        aligned,
        axis=0
    )

    average_std = np.std(
        aligned,
        axis=0
    )

    full_average = pd.DataFrame({
        "Distance": common_x,
        "AverageHeight": average_height,
        "StdDev": average_std
    })

    # --------------------------------------------------------
    # Plot aligned profiles
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    for i, name in enumerate(profile_names):

        ax.plot(
            common_x,
            aligned[i],
            linewidth=0.8,
            alpha=0.65,
            label=name
        )

    ax.set_xlabel(
        "Distance"
    )

    ax.set_ylabel(
        "Height"
    )

    ax.set_title(
        "Aligned Corrected Profiles"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    # --------------------------------------------------------
    # Full average plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        common_x,
        average_height,
        linewidth=1.5,
        label="Average"
    )

    ax.fill_between(
        common_x,
        average_height - average_std,
        average_height + average_std,
        alpha=0.25,
        label="±1 SD"
    )

    ax.set_xlabel(
        "Distance"
    )

    ax.set_ylabel(
        "Height"
    )

    ax.set_title(
        "Full Average Profile"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    st.pyplot(
        fig,
        clear_figure=True
    )

except Exception as e:

    st.error(
        f"Could not average profiles: {e}"
    )

    st.stop()


# ============================================================
# FINAL AVERAGE TRIMMING
# ============================================================

st.header("5. Final Average Trimming")

st.write(
    "These cuts are applied ONLY to the final average. "
    "They do not change the individual-profile averaging."
)

col1, col2 = st.columns(2)

with col1:

    final_left = st.number_input(
        "Final average left cut",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f"
    )

with col2:

    final_right = st.number_input(
        "Final average right cut",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f"
    )


# ============================================================
# APPLY FINAL TRIMMING
# ============================================================

final_xmin = (
    common_x.min()
    +
    final_left
)

final_xmax = (
    common_x.max()
    -
    final_right
)

final_mask = (
    (common_x >= final_xmin)
    &
    (common_x <= final_xmax)
)

final_x = common_x[
    final_mask
]

final_y = average_height[
    final_mask
]

final_sd = average_std[
    final_mask
]


# ============================================================
# FINAL METRICS
# ============================================================

metrics = calculate_metrics(
    final_x,
    final_y
)


# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("6. Final Profile Metrics")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "Ra",
        f"{metrics['Ra']:.6g}"
    )

with metric2:

    st.metric(
        "Rq",
        f"{metrics['Rq']:.6g}"
    )

with metric3:

    st.metric(
        "Rt",
        f"{metrics['Rt']:.6g}"
    )


# ============================================================
# FINAL AVERAGE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    final_x,
    final_y,
    linewidth=1.5,
    label="Final average"
)

ax.fill_between(
    final_x,
    final_y - final_sd,
    final_y + final_sd,
    alpha=0.25,
    label="±1 SD"
)

ax.set_xlabel(
    "Distance"
)

ax.set_ylabel(
    "Height"
)

ax.set_title(
    "Final Truncated Average Profile"
)

ax.grid(
    alpha=0.25
)

ax.legend()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# EXPORT DATA
# ============================================================

st.header("7. Export Data")


# ============================================================
# FINAL AVERAGE
# ============================================================

final_average_df = pd.DataFrame({
    "Distance": final_x,
    "AverageHeight": final_y,
    "StdDev": final_sd
})

final_average_csv = (
    final_average_df
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="Download Final Average CSV",
    data=final_average_csv,
    file_name="Final_Average_Profile.csv",
    mime="text/csv"
)


# ============================================================
# FULL AVERAGE
# ============================================================

full_average_csv = (
    full_average
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Full Average CSV",
    data=full_average_csv,
    file_name="Full_Average_Profile.csv",
    mime="text/csv"
)


# ============================================================
# ALIGNED PROFILES
# ============================================================

aligned_df = pd.DataFrame({
    "Distance": common_x
})

for i, name in enumerate(profile_names):

    safe_name = (
        name
        .replace(".csv", "")
        .replace(" ", "_")
        .replace("-", "_")
    )

    aligned_df[
        safe_name
    ] = aligned[i]


aligned_csv = (
    aligned_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Aligned Profiles CSV",
    data=aligned_csv,
    file_name="Aligned_Profiles.csv",
    mime="text/csv"
)


# ============================================================
# CORRECTED INDIVIDUAL PROFILES
# ============================================================

corrected_combined = []

for name, profile in zip(
    profile_names,
    corrected_profiles
):

    temp = profile.copy()

    temp.insert(
        0,
        "Profile",
        name
    )

    corrected_combined.append(
        temp
    )


corrected_df = pd.concat(
    corrected_combined,
    ignore_index=True
)

corrected_csv = (
    corrected_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Corrected Individual Profiles",
    data=corrected_csv,
    file_name="Corrected_Individual_Profiles.csv",
    mime="text/csv"
)


# ============================================================
# METRICS EXPORT
# ============================================================

metrics_df = pd.DataFrame([
    {
        "Metric": "Ra",
        "Value": metrics["Ra"]
    },
    {
        "Metric": "Rq",
        "Value": metrics["Rq"]
    },
    {
        "Metric": "Rt",
        "Value": metrics["Rt"]
    }
])

metrics_csv = (
    metrics_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Metrics CSV",
    data=metrics_csv,
    file_name="Profile_Metrics.csv",
    mime="text/csv"
)


# ============================================================
# SPIKE SUMMARY EXPORT
# ============================================================

spike_summary_csv = (
    spike_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download Spike Removal Summary",
    data=spike_summary_csv,
    file_name="Spike_Removal_Summary.csv",
    mime="text/csv"
)


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("How spike removal works")

st.markdown(
    """
**High spikes only** is the default because it is designed for
isolated upward artifacts that can artificially inflate the
profile average.

For every point, the program:

1. Calculates a local median profile.
2. Calculates how far each point is from that local median.
3. Uses a robust MAD-based estimate of normal variation.
4. Identifies unusually high points.
5. Replaces those points with the local median.
6. Uses the cleaned profile for averaging.

### Spike threshold

- **Lower threshold** → more aggressive spike removal
- **Higher threshold** → more conservative spike removal

A good starting point is **5.0**.

### Spike window

The window controls how many neighboring points are used to
determine the local median.

- **3–7** → good for very narrow, isolated spikes
- **9–15** → broader artifacts
- **20+** → increasingly aggressive and can start treating
  real profile features as spikes

The removed points are shown in **red** on each profile plot.
"""
)