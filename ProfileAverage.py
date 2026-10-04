import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import io
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Profilometer Profile Alignment",
    layout="wide"
)

st.title("Profilometer Profile Alignment & Averaging")

st.markdown(
    """
    Upload your profilometer CSV files and interactively:
    
    - shift profiles left/right
    - shift profiles up/down
    - truncate profile edges
    - remove isolated spikes
    - align profiles
    - calculate an average surface profile
    - export corrected and averaged data
    """
)


# ============================================================
# LOAD PROFILOMETER FILE
# ============================================================

def load_profile(uploaded_file):

    # Read uploaded file
    raw_bytes = uploaded_file.getvalue()

    text = raw_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    start_index = None

    for i, line in enumerate(lines):

        parts = line.strip().split(",")

        if len(parts) < 2:
            continue

        try:

            float(parts[0])
            float(parts[1])

            start_index = i
            break

        except:

            continue

    if start_index is None:

        raise ValueError(
            f"Could not find numeric data in "
            f"{uploaded_file.name}"
        )

    df = pd.read_csv(
        io.StringIO(text),
        skiprows=start_index,
        header=None
    )

    df = df.iloc[:, :2].copy()

    df.columns = [
        "Distance",
        "Height"
    ]

    # Remove nonnumeric rows
    df["Distance"] = pd.to_numeric(
        df["Distance"],
        errors="coerce"
    )

    df["Height"] = pd.to_numeric(
        df["Height"],
        errors="coerce"
    )

    df = df.dropna()

    # Sort by distance
    df = df.sort_values(
        "Distance"
    )

    # Remove duplicate x values
    df = df.drop_duplicates(
        subset="Distance"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# DESPIKE FUNCTION
# ============================================================

def remove_spikes(
    distance,
    height,
    window,
    threshold
):

    height = np.asarray(
        height,
        dtype=float
    )

    # Make sure window is odd
    if window % 2 == 0:
        window += 1

    if window < 3:
        window = 3

    if window >= len(height):
        return height.copy(), np.zeros(
            len(height),
            dtype=bool
        )

    median_profile = medfilt(
        height,
        kernel_size=window
    )

    deviation = np.abs(
        height - median_profile
    )

    # Robust scale estimate
    mad = np.median(
        np.abs(
            deviation -
            np.median(deviation)
        )
    )

    if mad == 0:

        sigma = np.std(
            deviation
        )

    else:

        sigma = 1.4826 * mad

    if sigma == 0:

        spike_mask = np.zeros(
            len(height),
            dtype=bool
        )

    else:

        spike_mask = (
            deviation >
            threshold * sigma
        )

    cleaned = height.copy()

    # Replace detected spikes
    cleaned[spike_mask] = (
        median_profile[spike_mask]
    )

    return cleaned, spike_mask


# ============================================================
# SHIFT PROFILE
# ============================================================

def shift_profile(
    df,
    shift_x,
    shift_y
):

    result = df.copy()

    result["Distance"] = (
        result["Distance"] +
        shift_x
    )

    result["Height"] = (
        result["Height"] +
        shift_y
    )

    return result


# ============================================================
# TRUNCATE PROFILE
# ============================================================

def truncate_profile(
    df,
    left,
    right
):

    x_min = df["Distance"].min()
    x_max = df["Distance"].max()

    lower = x_min + left
    upper = x_max - right

    result = df[
        (df["Distance"] >= lower) &
        (df["Distance"] <= upper)
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True
    )

    return result


# ============================================================
# SESSION STATE
# ============================================================

if "profiles" not in st.session_state:

    st.session_state.profiles = {}


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload profilometer CSV files",
    type=["csv"],
    accept_multiple_files=True
)


if not uploaded_files:

    st.info(
        "Upload two or more profilometer CSV files "
        "to begin."
    )

    st.stop()


# ============================================================
# LOAD FILES
# ============================================================

for uploaded_file in uploaded_files:

    if uploaded_file.name not in (
        st.session_state.profiles
    ):

        try:

            st.session_state.profiles[
                uploaded_file.name
            ] = load_profile(
                uploaded_file
            )

        except Exception as e:

            st.error(
                f"Error loading "
                f"{uploaded_file.name}: {e}"
            )


# Remove files no longer uploaded

current_names = [
    f.name for f in uploaded_files
]

st.session_state.profiles = {
    name: df
    for name, df
    in st.session_state.profiles.items()
    if name in current_names
}


# ============================================================
# GLOBAL SETTINGS
# ============================================================

st.sidebar.header(
    "Global Settings"
)

show_raw = st.sidebar.checkbox(
    "Show raw profiles",
    value=True
)

show_corrected = st.sidebar.checkbox(
    "Show corrected profiles",
    value=True
)

# ============================================================
# RAW PROFILE PLOT
# ============================================================

if show_raw:

    st.subheader(
        "Raw Profiles"
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    for name, df in (
        st.session_state.profiles.items()
    ):

        ax.plot(
            df["Distance"],
            df["Height"],
            linewidth=1,
            label=name
        )

    ax.set_xlabel(
        "Distance × 100 μm"
    )

    ax.set_ylabel(
        "Height (nm)"
    )

    ax.set_title(
        "Raw Profilometer Profiles"
    )

    ax.legend(
        bbox_to_anchor=(1.02, 0.5),
        loc="center left"
    )

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# PROFILE CORRECTION
# ============================================================

st.header(
    "Profile Corrections"
)

st.markdown(
    """
    Adjust each profile independently.  
    **X shift** moves the profile left/right.  
    **Y shift** moves it up/down.  
    **Left/right truncation** removes data from the ends.
    """
)


corrected_profiles = {}

spike_masks = {}


# ============================================================
# ONE EXPANDER PER PROFILE
# ============================================================

for profile_number, (
    name,
    original
) in enumerate(
    st.session_state.profiles.items()
):

    st.subheader(
        f"Profile {profile_number + 1}: {name}"
    )

    x_min = float(
        original["Distance"].min()
    )

    x_max = float(
        original["Distance"].max()
    )

    x_range = x_max - x_min

    height_min = float(
        original["Height"].min()
    )

    height_max = float(
        original["Height"].max()
    )

    height_range = (
        height_max -
        height_min
    )

    if height_range == 0:
        height_range = 1

    col1, col2, col3, col4 = st.columns(4)

    # --------------------------------------------------------
    # X SHIFT
    # --------------------------------------------------------

    with col1:

        shift_x = st.number_input(
            "Shift X",
            value=0.0,
            step=1.0,
            key=f"xshift_{profile_number}",
            help="Positive = move profile right"
        )

    # --------------------------------------------------------
    # Y SHIFT
    # --------------------------------------------------------

    with col2:

        shift_y = st.number_input(
            "Shift Y (nm)",
            value=0.0,
            step=1.0,
            key=f"yshift_{profile_number}",
            help="Positive = move profile up"
        )

    # --------------------------------------------------------
    # LEFT TRUNCATION
    # --------------------------------------------------------

    with col3:

        truncate_left = st.number_input(
            "Remove left edge",
            min_value=0.0,
            max_value=float(x_range),
            value=0.0,
            step=1.0,
            key=f"left_{profile_number}"
        )

    # --------------------------------------------------------
    # RIGHT TRUNCATION
    # --------------------------------------------------------

    with col4:

        truncate_right = st.number_input(
            "Remove right edge",
            min_value=0.0,
            max_value=float(x_range),
            value=0.0,
            step=1.0,
            key=f"right_{profile_number}"
        )

    # ========================================================
    # SPIKE SETTINGS
    # ========================================================

    spike_col1, spike_col2 = st.columns(2)

    with spike_col1:

        remove_spikes_checkbox = st.checkbox(
            "Remove wild spikes",
            value=True,
            key=f"spikes_{profile_number}"
        )

    with spike_col2:

        spike_threshold = st.slider(
            "Spike sensitivity",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key=f"threshold_{profile_number}",
            help=(
                "Lower values remove more points. "
                "Higher values are more conservative."
            )
        )

    spike_window = st.slider(
        "Spike detection window",
        min_value=3,
        max_value=51,
        value=11,
        step=2,
        key=f"window_{profile_number}"
    )

    # ========================================================
    # APPLY CORRECTIONS
    # ========================================================

    corrected = shift_profile(
        original,
        shift_x,
        shift_y
    )

    corrected = truncate_profile(
        corrected,
        truncate_left,
        truncate_right
    )

    if len(corrected) < 5:

        st.warning(
            "Too few points remain after truncation."
        )

        continue

    if remove_spikes_checkbox:

        cleaned_height, spike_mask = (
            remove_spikes(
                corrected["Distance"].values,
                corrected["Height"].values,
                spike_window,
                spike_threshold
            )
        )

        corrected["Height"] = (
            cleaned_height
        )

    else:

        spike_mask = np.zeros(
            len(corrected),
            dtype=bool
        )

    corrected_profiles[name] = (
        corrected
    )

    spike_masks[name] = (
        spike_mask
    )

    # ========================================================
    # INDIVIDUAL PREVIEW
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11, 4)
    )

    ax.plot(
        original["Distance"],
        original["Height"],
        linewidth=1,
        alpha=0.35,
        label="Raw"
    )

    ax.plot(
        corrected["Distance"],
        corrected["Height"],
        linewidth=1.5,
        label="Corrected"
    )

    if np.any(spike_mask):

        ax.scatter(
            corrected.loc[
                spike_mask,
                "Distance"
            ],
            corrected.loc[
                spike_mask,
                "Height"
            ],
            s=20,
            label="Removed spikes"
        )

    ax.set_xlabel(
        "Distance × 100 μm"
    )

    ax.set_ylabel(
        "Height (nm)"
    )

    ax.set_title(
        f"{name} — Correction Preview"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# CORRECTED PROFILES
# ============================================================

st.header(
    "Aligned / Corrected Profiles"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

for name, df in corrected_profiles.items():

    ax.plot(
        df["Distance"],
        df["Height"],
        linewidth=1.5,
        label=name
    )

ax.set_xlabel(
    "Distance × 100 μm"
)

ax.set_ylabel(
    "Height (nm)"
)

ax.set_title(
    "Corrected Profiles"
)

ax.legend(
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# AVERAGING
# ============================================================

st.header(
    "Average Profile"
)

if len(corrected_profiles) < 2:

    st.warning(
        "At least two valid corrected profiles "
        "are required for averaging."
    )

    st.stop()


# ============================================================
# COMMON DISTANCE RANGE
# ============================================================

common_min = max(
    df["Distance"].min()
    for df in corrected_profiles.values()
)

common_max = min(
    df["Distance"].max()
    for df in corrected_profiles.values()
)

if common_min >= common_max:

    st.error(
        "The corrected profiles do not overlap "
        "in the X direction."
    )

    st.stop()


# ============================================================
# NUMBER OF POINTS
# ============================================================

point_counts = [
    len(df)
    for df in corrected_profiles.values()
]

default_points = min(
    min(point_counts),
    10000
)

n_points = st.number_input(
    "Number of points in average curve",
    min_value=100,
    max_value=50000,
    value=int(default_points),
    step=100
)


distance = np.linspace(
    common_min,
    common_max,
    int(n_points)
)


# ============================================================
# INTERPOLATE
# ============================================================

height_matrix = []

profile_names = []

for name, df in corrected_profiles.items():

    interpolated = np.interp(
        distance,
        df["Distance"].values,
        df["Height"].values
    )

    height_matrix.append(
        interpolated
    )

    profile_names.append(
        name
    )


height_matrix = np.array(
    height_matrix
)


# ============================================================
# AVERAGE
# ============================================================

average_height = np.mean(
    height_matrix,
    axis=0
)

std_height = np.std(
    height_matrix,
    axis=0
)


# ============================================================
# CENTER AVERAGE PROFILE
# ============================================================

centered_profile = (
    average_height -
    np.mean(average_height)
)


# ============================================================
# METRICS
# ============================================================

peak_index = np.argmax(
    centered_profile
)

peak_height = (
    centered_profile[
        peak_index
    ]
)

peak_location = (
    distance[
        peak_index
    ]
)


valley_index = np.argmin(
    centered_profile
)

valley_depth = (
    centered_profile[
        valley_index
    ]
)

valley_location = (
    distance[
        valley_index
    ]
)


Rt = (
    peak_height -
    valley_depth
)

Ra = np.mean(
    np.abs(
        centered_profile
    )
)

Rq = np.sqrt(
    np.mean(
        centered_profile ** 2
    )
)


# ============================================================
# DISPLAY METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Peak-to-Valley Rt",
    f"{Rt:.2f} nm"
)

metric2.metric(
    "Ra",
    f"{Ra:.2f} nm"
)

metric3.metric(
    "Rq",
    f"{Rq:.2f} nm"
)

metric4.metric(
    "Profiles Averaged",
    str(len(corrected_profiles))
)


# ============================================================
# AVERAGE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    distance,
    centered_profile,
    linewidth=2,
    label="Average Profile"
)

ax.fill_between(
    distance,
    centered_profile - std_height,
    centered_profile + std_height,
    alpha=0.25,
    label="±1 SD"
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    label="Average Surface"
)

ax.scatter(
    peak_location,
    peak_height,
    marker="^",
    s=100,
    label="Highest Peak"
)

ax.scatter(
    valley_location,
    valley_depth,
    marker="v",
    s=100,
    label="Deepest Valley"
)

ax.set_xlabel(
    "Distance × 100 μm"
)

ax.set_ylabel(
    "Height Relative to Average Surface (nm)"
)

ax.set_title(
    "Average Surface Profile"
)

ax.legend(
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# EXPORT AVERAGE PROFILE
# ============================================================

st.header(
    "Export"
)


# ------------------------------------------------------------
# Average CSV
# ------------------------------------------------------------

average_export = pd.DataFrame({
    "Distance": distance,
    "Average_Height": average_height,
    "Centered_Average_Height": centered_profile,
    "Standard_Deviation": std_height
})


average_csv = (
    average_export
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="Download average profile CSV",
    data=average_csv,
    file_name="Average_Profile.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# Aligned profile matrix
# ------------------------------------------------------------

matrix_dict = {
    "Distance": distance
}

for i, name in enumerate(
    profile_names
):

    matrix_dict[
        name
    ] = height_matrix[i]


matrix_export = pd.DataFrame(
    matrix_dict
)


matrix_csv = (
    matrix_export
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="Download aligned profiles CSV",
    data=matrix_csv,
    file_name="Aligned_Profiles.csv",
    mime="text/csv"
)


# ============================================================
# EXPORT INDIVIDUAL CORRECTED PROFILES
# ============================================================

st.subheader(
    "Corrected Individual Profiles"
)

for name, df in corrected_profiles.items():

    safe_name = os.path.splitext(
        name
    )[0]

    export_df = df.copy()

    export_df.columns = [
        "Distance",
        "Corrected_Height"
    ]

    export_csv = (
        export_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label=f"Download {name}",
        data=export_csv,
        file_name=(
            f"{safe_name}_Corrected.csv"
        ),
        mime="text/csv",
        key=f"download_{safe_name}"
    )


# ============================================================
# EXPORT METRICS
# ============================================================

metrics_export = pd.DataFrame({
    "Metric": [
        "Number of Profiles",
        "Peak Height (nm)",
        "Peak Location",
        "Valley Depth (nm)",
        "Valley Location",
        "Rt (nm)",
        "Ra (nm)",
        "Rq (nm)"
    ],

    "Value": [
        len(corrected_profiles),
        peak_height,
        peak_location,
        valley_depth,
        valley_location,
        Rt,
        Ra,
        Rq
    ]
})


metrics_csv = (
    metrics_export
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="Download surface metrics CSV",
    data=metrics_csv,
    file_name="Surface_Metrics.csv",
    mime="text/csv"
)