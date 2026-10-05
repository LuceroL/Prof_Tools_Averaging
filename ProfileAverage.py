# ============================================================
# PROFILOMETER PROFILE ALIGNMENT + AVERAGING APP
# ============================================================
#
# Designed for profilometer exports such as:
#
# Meta Data
# ...
# Data
# Lateral(µm),Total Profile(Å),,
# 0,-5528.24,,
# 0.77769,-5547.78,,
# ...
#
# FEATURES
# ------------------------------------------------------------
# - Upload multiple profilometer CSV files
# - Handles metadata before the Data section
# - Handles extra trailing commas
# - Automatically identifies Lateral / Total Profile columns
# - Individual X shift
# - Individual Y shift
# - Individual left/right trimming
# - High spike removal
# - Low spike removal
# - High + low spike removal
# - Removed spikes shown in red
# - Robust MAD-based spike detection
# - Common overlap averaging
# - Standard deviation
# - Final average trimming
# - Ra / Rq / Rt
# - Multiple CSV exports
#
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter

from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Profilometer Profile Averaging",
    layout="wide"
)


st.title(
    "Profilometer Profile Alignment & Averaging"
)


st.caption(
    "Align, trim, remove spikes, and average profilometer profiles."
)


# ============================================================
# FILE LOADER
# ============================================================

def load_profile(uploaded_file):
    """
    Load profilometer CSV files with metadata and inconsistent
    trailing commas.

    Specifically supports files containing:

        Data
        Lateral(µm),Total Profile(Å),,
        x,z,,

    Extra columns are ignored.
    """

    # --------------------------------------------------------
    # Read raw bytes
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    # Try UTF-8 first, then common alternatives
    text = None

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252"
    ]

    for encoding in encodings:

        try:

            text = file_bytes.decode(
                encoding
            )

            break

        except UnicodeDecodeError:

            continue

    if text is None:

        raise ValueError(
            "Could not decode the file."
        )

    # --------------------------------------------------------
    # Split into lines
    # --------------------------------------------------------

    lines = text.splitlines()

    if len(lines) == 0:

        raise ValueError(
            "The CSV file is empty."
        )

    # --------------------------------------------------------
    # Find the actual data header
    #
    # We specifically look for:
    #
    # Lateral
    # Total Profile
    #
    # --------------------------------------------------------

    header_index = None

    for i, line in enumerate(lines):

        lower_line = (
            line.strip()
            .lower()
        )

        if (
            "lateral" in lower_line
            and
            "total profile" in lower_line
        ):

            header_index = i

            break

    # --------------------------------------------------------
    # If the exact profilometer format wasn't found,
    # look for a Data marker and then inspect the next rows.
    # --------------------------------------------------------

    if header_index is None:

        data_marker_index = None

        for i, line in enumerate(lines):

            if line.strip().lower() == "data":

                data_marker_index = i

                break

        if data_marker_index is not None:

            for i in range(
                data_marker_index + 1,
                min(
                    data_marker_index + 20,
                    len(lines)
                )
            ):

                lower_line = (
                    lines[i]
                    .strip()
                    .lower()
                )

                if (
                    "lateral" in lower_line
                    or
                    "distance" in lower_line
                ):

                    header_index = i

                    break

    # --------------------------------------------------------
    # If still not found, attempt generic CSV detection
    # --------------------------------------------------------

    if header_index is None:

        try:

            preview = pd.read_csv(
                BytesIO(file_bytes),
                header=None,
                nrows=50,
                engine="python",
                encoding="latin1"
            )

        except Exception as e:

            raise ValueError(
                f"Could not inspect CSV: {e}"
            )

        for i in range(
            len(preview)
        ):

            row = (
                preview.iloc[i]
                .astype(str)
                .str.lower()
            )

            row_text = " ".join(
                row.tolist()
            )

            if (
                "distance" in row_text
                and
                (
                    "height" in row_text
                    or
                    "profile" in row_text
                )
            ):

                header_index = i

                break

    # --------------------------------------------------------
    # Final failure
    # --------------------------------------------------------

    if header_index is None:

        raise ValueError(
            "Could not find the profilometer data header. "
            "Expected columns such as "
            "'Lateral(µm)' and 'Total Profile(Å)'."
        )

    # ========================================================
    # PARSE THE DATA MANUALLY
    # ========================================================
    #
    # This is the important fix.
    #
    # We do NOT let Pandas decide how many columns the row
    # should have.
    #
    # We only take the first two fields:
    #
    #     Distance
    #     Height
    #
    # This handles:
    #
    #     0,-5528.24,,
    #
    # without throwing an error.
    # ========================================================

    header_parts = (
        lines[header_index]
        .strip()
        .split(",")
    )

    # --------------------------------------------------------
    # Identify distance and height column indices
    # --------------------------------------------------------

    distance_index = None
    height_index = None

    for i, column in enumerate(
        header_parts
    ):

        column_clean = (
            column
            .strip()
            .lower()
        )

        if (
            distance_index is None
            and
            (
                "lateral" in column_clean
                or
                "distance" in column_clean
                or
                "position" in column_clean
            )
        ):

            distance_index = i

        if (
            height_index is None
            and
            (
                "total profile" in column_clean
                or
                "height" in column_clean
                or
                "elevation" in column_clean
                or
                "profile" in column_clean
            )
        ):

            height_index = i

    # --------------------------------------------------------
    # If named columns weren't found, assume first two
    # --------------------------------------------------------

    if (
        distance_index is None
        or
        height_index is None
    ):

        distance_index = 0
        height_index = 1

    # --------------------------------------------------------
    # Read data lines
    # --------------------------------------------------------

    distance_values = []
    height_values = []

    for line in lines[
        header_index + 1:
    ]:

        line = line.strip()

        if not line:

            continue

        # Ignore obvious metadata / section markers
        if line.lower() in [
            "data",
            "metadata"
        ]:

            continue

        parts = line.split(",")

        # Need enough fields
        if (
            len(parts)
            <= max(
                distance_index,
                height_index
            )
        ):

            continue

        distance_string = (
            parts[distance_index]
            .strip()
        )

        height_string = (
            parts[height_index]
            .strip()
        )

        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        try:

            distance = float(
                distance_string
            )

            height = float(
                height_string
            )

        except (ValueError, TypeError):

            continue

        if (
            np.isfinite(distance)
            and
            np.isfinite(height)
        ):

            distance_values.append(
                distance
            )

            height_values.append(
                height
            )

    # --------------------------------------------------------
    # Make dataframe
    # --------------------------------------------------------

    clean = pd.DataFrame({
        "Distance": distance_values,
        "Height": height_values
    })

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if len(clean) < 5:

        raise ValueError(
            f"Only {len(clean)} valid profile points "
            "were found."
        )

    # --------------------------------------------------------
    # Remove duplicate X values
    # --------------------------------------------------------

    clean = clean.drop_duplicates(
        subset="Distance",
        keep="first"
    )

    # --------------------------------------------------------
    # Sort by distance
    # --------------------------------------------------------

    clean = clean.sort_values(
        "Distance"
    )

    clean = clean.reset_index(
        drop=True
    )

    return clean


# ============================================================
# SPIKE REMOVAL
# ============================================================

def remove_spikes(
    height,
    window,
    threshold,
    mode
):
    """
    Remove local spikes using a robust median/MAD approach.

    Modes:
        Off
        High spikes only
        Low spikes only
        High + low spikes

    Returns:
        cleaned profile
        spike mask
    """

    height = np.asarray(
        height,
        dtype=float
    )

    n = len(height)

    # --------------------------------------------------------
    # OFF
    # --------------------------------------------------------

    if mode == "Off":

        return (
            height.copy(),
            np.zeros(
                n,
                dtype=bool
            )
        )

    if n < 5:

        return (
            height.copy(),
            np.zeros(
                n,
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # Make window odd
    # --------------------------------------------------------

    window = int(window)

    if window < 3:

        window = 3

    if window % 2 == 0:

        window += 1

    # Do not exceed data length
    if window > n:

        window = n

        if window % 2 == 0:

            window -= 1

    if window < 3:

        return (
            height.copy(),
            np.zeros(
                n,
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # Local median
    #
    # reflect avoids artificial zero padding
    # --------------------------------------------------------

    local_median = median_filter(
        height,
        size=window,
        mode="reflect"
    )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    residual = (
        height
        -
        local_median
    )

    # --------------------------------------------------------
    # Robust MAD
    # --------------------------------------------------------

    residual_median = np.median(
        residual
    )

    mad = np.median(
        np.abs(
            residual
            -
            residual_median
        )
    )

    if mad > 0:

        sigma = (
            1.4826
            *
            mad
        )

    else:

        sigma = np.std(
            residual
        )

    # --------------------------------------------------------
    # Determine spikes
    # --------------------------------------------------------

    if (
        sigma == 0
        or
        not np.isfinite(sigma)
    ):

        spike_mask = np.zeros(
            n,
            dtype=bool
        )

    elif mode == "High spikes only":

        spike_mask = (
            residual
            >
            threshold * sigma
        )

    elif mode == "Low spikes only":

        spike_mask = (
            residual
            <
            -threshold * sigma
        )

    elif mode == "High + low spikes":

        spike_mask = (
            np.abs(residual)
            >
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

    cleaned[
        spike_mask
    ] = local_median[
        spike_mask
    ]

    return (
        cleaned,
        spike_mask
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    x,
    height
):
    """
    Calculate Ra, Rq and Rt.
    """

    if len(height) == 0:

        return {
            "Ra": np.nan,
            "Rq": np.nan,
            "Rt": np.nan
        }

    # Remove mean line
    centered = (
        height
        -
        np.mean(height)
    )

    Ra = np.mean(
        np.abs(centered)
    )

    Rq = np.sqrt(
        np.mean(
            centered ** 2
        )
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
# INTERPOLATION / ALIGNMENT
# ============================================================

def interpolate_profiles(
    profiles
):
    """
    Put all corrected profiles onto a common distance grid.

    Only the region shared by ALL profiles is averaged.
    """

    if len(profiles) == 0:

        return None, None

    # --------------------------------------------------------
    # Common overlap
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
    # Determine representative spacing
    # --------------------------------------------------------

    spacings = []

    for profile in profiles:

        dx = np.diff(
            profile["Distance"].values
        )

        dx = dx[
            np.isfinite(dx)
            &
            (dx > 0)
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

    if spacing <= 0:

        raise ValueError(
            "Could not determine profile spacing."
        )

    # --------------------------------------------------------
    # Common X grid
    # --------------------------------------------------------

    common_x = np.arange(
        xmin,
        xmax + spacing,
        spacing
    )

    # Make sure last point isn't outside overlap
    common_x = common_x[
        common_x <= xmax
    ]

    # --------------------------------------------------------
    # Interpolate
    # --------------------------------------------------------

    aligned = []

    for profile in profiles:

        x = (
            profile["Distance"]
            .values
        )

        z = (
            profile["Height"]
            .values
        )

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

    return (
        common_x,
        aligned
    )


# ============================================================
# SESSION STATE
# ============================================================

if "profiles" not in st.session_state:

    st.session_state.profiles = {}


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "1. Upload Profiles"
)

uploaded_files = st.file_uploader(
    "Upload one or more profilometer CSV files",
    type=["csv"],
    accept_multiple_files=True
)


# ============================================================
# LOAD FILES
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        name = uploaded_file.name

        try:

            df = load_profile(
                uploaded_file
            )

            st.session_state.profiles[
                name
            ] = {
                "raw": df
            }

        except Exception as e:

            st.error(
                f"Could not load {name}: {e}"
            )


# ============================================================
# STOP IF NOTHING LOADED
# ============================================================

if len(
    st.session_state.profiles
) == 0:

    st.info(
        "Upload one or more CSV files to begin."
    )

    st.stop()


# ============================================================
# LOADED FILE SUMMARY
# ============================================================

st.success(
    f"{len(st.session_state.profiles)} "
    "profile(s) loaded successfully."
)


loaded_summary = []

for name, info in (
    st.session_state.profiles.items()
):

    df = info["raw"]

    loaded_summary.append({
        "File": name,
        "Points": len(df),
        "Distance min": df["Distance"].min(),
        "Distance max": df["Distance"].max(),
        "Height min": df["Height"].min(),
        "Height max": df["Height"].max()
    })


loaded_summary_df = pd.DataFrame(
    loaded_summary
)

st.dataframe(
    loaded_summary_df,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL SETTINGS
# ============================================================

st.header(
    "2. Individual Profile Settings"
)

st.write(
    "These corrections are applied individually before "
    "the profiles are averaged."
)


profile_names = list(
    st.session_state.profiles.keys()
)

settings = {}


for i, name in enumerate(
    profile_names
):

    raw = (
        st.session_state.profiles[
            name
        ]["raw"]
        .copy()
    )

    st.subheader(
        f"Profile {i + 1}: {name}"
    )

    # --------------------------------------------------------
    # SHIFT SETTINGS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        x_shift = st.number_input(
            "X shift",
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"xshift_{name}"
        )

    with col2:

        y_shift = st.number_input(
            "Y shift",
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"yshift_{name}"
        )

    # --------------------------------------------------------
    # EDGE TRIMMING
    # --------------------------------------------------------

    st.markdown(
        "**Individual edge trimming**"
    )

    col1, col2 = st.columns(2)

    with col1:

        left_cut = st.number_input(
            "Left cut",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"leftcut_{name}"
        )

    with col2:

        right_cut = st.number_input(
            "Right cut",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"rightcut_{name}"
        )

    # --------------------------------------------------------
    # SPIKE REMOVAL
    # --------------------------------------------------------

    st.markdown(
        "**Spike removal**"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

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

    with col2:

        spike_window = st.slider(
            "Spike window",
            min_value=3,
            max_value=51,
            value=7,
            step=2,
            key=f"spike_window_{name}",
            help=(
                "Number of neighboring points "
                "used to calculate the local median."
            )
        )

    with col3:

        spike_threshold = st.slider(
            "Spike threshold",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key=f"spike_threshold_{name}",
            help=(
                "Lower = more aggressive. "
                "Higher = more conservative."
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
# PROCESS PROFILES
# ============================================================

st.header(
    "3. Corrected Profiles"
)


corrected_profiles = []

spike_summary = []


for name in profile_names:

    raw = (
        st.session_state.profiles[
            name
        ]["raw"]
        .copy()
    )

    s = settings[name]

    # --------------------------------------------------------
    # Original arrays
    # --------------------------------------------------------

    x = raw[
        "Distance"
    ].values.copy()

    z = raw[
        "Height"
    ].values.copy()

    # --------------------------------------------------------
    # X shift
    # --------------------------------------------------------

    x = (
        x
        +
        s["x_shift"]
    )

    # --------------------------------------------------------
    # Y shift
    # --------------------------------------------------------

    z = (
        z
        +
        s["y_shift"]
    )

    # --------------------------------------------------------
    # INDIVIDUAL EDGE CUTTING
    #
    # IMPORTANT:
    # These points are completely removed before averaging.
    # --------------------------------------------------------

    original_min = x.min()
    original_max = x.max()

    xmin = (
        original_min
        +
        s["left_cut"]
    )

    xmax = (
        original_max
        -
        s["right_cut"]
    )

    if xmin >= xmax:

        st.error(
            f"{name}: left/right trimming removes "
            "the entire profile."
        )

        st.stop()

    trim_mask = (
        (x >= xmin)
        &
        (x <= xmax)
    )

    x = x[
        trim_mask
    ]

    z = z[
        trim_mask
    ]

    # --------------------------------------------------------
    # SPIKE REMOVAL
    # --------------------------------------------------------

    z_clean, spike_mask = remove_spikes(
        z,
        s["spike_window"],
        s["spike_threshold"],
        s["spike_mode"]
    )

    num_spikes = int(
        np.sum(spike_mask)
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    spike_summary.append({
        "Profile": name,
        "Mode": s["spike_mode"],
        "Window": s["spike_window"],
        "Threshold": s["spike_threshold"],
        "Spikes Removed": num_spikes,
        "Points Remaining": len(x)
    })

    # --------------------------------------------------------
    # Store corrected dataframe
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

    # --------------------------------------------------------
    # Profile information
    # --------------------------------------------------------

    st.write(
        f"**{name}** — "
        f"{len(x):,} points remaining; "
        f"**{num_spikes:,} spikes removed**"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    # Original trimmed
    ax.plot(
        x,
        z,
        linewidth=0.7,
        alpha=0.7,
        label="Trimmed original"
    )

    # Corrected
    ax.plot(
        x,
        z_clean,
        linewidth=1.2,
        label="Corrected"
    )

    # --------------------------------------------------------
    # RED SPIKE MARKERS
    # --------------------------------------------------------

    if num_spikes > 0:

        ax.scatter(
            x[spike_mask],
            z[spike_mask],
            s=20,
            label=(
                f"Removed spikes ({num_spikes})"
            ),
            zorder=5
        )

    ax.set_xlabel(
        "Distance (µm)"
    )

    ax.set_ylabel(
        "Height (Å)"
    )

    ax.set_title(
        f"Corrected Profile — {name}"
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
# SPIKE SUMMARY
# ============================================================

st.subheader(
    "Spike Removal Summary"
)

spike_df = pd.DataFrame(
    spike_summary
)

st.dataframe(
    spike_df,
    use_container_width=True
)


# ============================================================
# AVERAGE
# ============================================================

st.header(
    "4. Average Profiles"
)


try:

    common_x, aligned = (
        interpolate_profiles(
            corrected_profiles
        )
    )

except Exception as e:

    st.error(
        f"Could not align profiles: {e}"
    )

    st.stop()


# ============================================================
# AVERAGE
# ============================================================

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


# ============================================================
# ALIGNED PROFILES PLOT
# ============================================================

st.subheader(
    "Aligned Corrected Profiles"
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)


for i, name in enumerate(
    profile_names
):

    ax.plot(
        common_x,
        aligned[i],
        linewidth=0.8,
        alpha=0.65,
        label=name
    )


ax.set_xlabel(
    "Distance (µm)"
)

ax.set_ylabel(
    "Height (Å)"
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


# ============================================================
# FULL AVERAGE PLOT
# ============================================================

st.subheader(
    "Full Average"
)


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
    "Distance (µm)"
)

ax.set_ylabel(
    "Height (Å)"
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


# ============================================================
# FINAL AVERAGE TRIMMING
# ============================================================

st.header(
    "5. Final Average Trimming"
)

st.write(
    "These cuts affect only the final average. "
    "They do NOT change which data were used to calculate "
    "the individual-profile average."
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
# APPLY FINAL TRIM
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


if final_xmin >= final_xmax:

    st.error(
        "Final average trimming removes the entire profile."
    )

    st.stop()


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
# METRICS
# ============================================================

metrics = calculate_metrics(
    final_x,
    final_y
)


# ============================================================
# METRICS DISPLAY
# ============================================================

st.header(
    "6. Final Profile Metrics"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Ra",
        f"{metrics['Ra']:.6g} Å"
    )


with col2:

    st.metric(
        "Rq",
        f"{metrics['Rq']:.6g} Å"
    )


with col3:

    st.metric(
        "Rt",
        f"{metrics['Rt']:.6g} Å"
    )


# ============================================================
# FINAL PROFILE PLOT
# ============================================================

st.subheader(
    "Final Truncated Average"
)


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
    "Distance (µm)"
)

ax.set_ylabel(
    "Height (Å)"
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
# EXPORT SECTION
# ============================================================

st.header(
    "7. Export Data"
)


# ============================================================
# FINAL AVERAGE CSV
# ============================================================

final_average_df = pd.DataFrame({
    "Distance_um": final_x,
    "AverageHeight_A": final_y,
    "StdDev_A": final_sd
})


st.download_button(
    label="Download Final Average CSV",
    data=(
        final_average_df
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Final_Average_Profile.csv",
    mime="text/csv"
)


# ============================================================
# FULL AVERAGE CSV
# ============================================================

st.download_button(
    label="Download Full Average CSV",
    data=(
        full_average
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Full_Average_Profile.csv",
    mime="text/csv"
)


# ============================================================
# ALIGNED PROFILES CSV
# ============================================================

aligned_df = pd.DataFrame({
    "Distance_um": common_x
})


for i, name in enumerate(
    profile_names
):

    safe_name = (
        name
        .replace(".csv", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    aligned_df[
        safe_name
    ] = aligned[i]


st.download_button(
    label="Download Aligned Profiles CSV",
    data=(
        aligned_df
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Aligned_Profiles.csv",
    mime="text/csv"
)


# ============================================================
# CORRECTED INDIVIDUAL PROFILES CSV
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


st.download_button(
    label="Download Corrected Individual Profiles",
    data=(
        corrected_df
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Corrected_Individual_Profiles.csv",
    mime="text/csv"
)


# ============================================================
# METRICS CSV
# ============================================================

metrics_df = pd.DataFrame([
    {
        "Metric": "Ra",
        "Value_A": metrics["Ra"]
    },
    {
        "Metric": "Rq",
        "Value_A": metrics["Rq"]
    },
    {
        "Metric": "Rt",
        "Value_A": metrics["Rt"]
    }
])


st.download_button(
    label="Download Metrics CSV",
    data=(
        metrics_df
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Profile_Metrics.csv",
    mime="text/csv"
)


# ============================================================
# SPIKE SUMMARY CSV
# ============================================================

st.download_button(
    label="Download Spike Removal Summary",
    data=(
        spike_df
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="Spike_Removal_Summary.csv",
    mime="text/csv"
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.divider()

st.header(
    "Spike Removal Guide"
)

st.markdown(
    """
### Recommended starting settings

**Spike mode:** High spikes only  
**Spike window:** 7  
**Spike threshold:** 5.0

### What the settings do

**High spikes only**

Removes points that are unusually high compared with
their local surroundings. This is the recommended setting
if your main problem is isolated upward artifacts.

**Low spikes only**

Removes unusually low points.

**High + low spikes**

Removes unusually high AND unusually low points.

**Off**

No spike correction is performed.

### Threshold

A lower threshold is more aggressive.

| Threshold | Behavior |
|---|---|
| 3–4 | Aggressive |
| 5 | Good starting point |
| 6–7 | Conservative |
| 8–10 | Very conservative |

### Window

The window controls the local region used to determine
what the profile should look like.

| Window | Best for |
|---|---|
| 3–5 | Very narrow spikes |
| 7 | Isolated spikes — recommended starting point |
| 9–15 | Wider artifacts |
| 17+ | Very broad deviations |

Be careful with very large windows because legitimate
profile features can eventually be interpreted as spikes.

### Important

The red points shown on each individual profile are the
points that were identified as spikes.

Those points are replaced with the local median **before
the profile enters the averaging calculation**.

Individual left/right cuts are also applied before averaging.

Final-average trimming occurs afterward and only changes
the portion of the already-calculated average that is
reported/exported.
"""
)