import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import io
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Profilometer Profile Averaging Tool",
    layout="wide"
)

st.title("Profilometer Profile Alignment & Averaging Tool")

st.markdown("""
Upload multiple profilometer CSV files, adjust each profile individually,
remove spikes, align the profiles, calculate the average, and apply final
truncation before exporting the final result.
""")


# ============================================================
# LOAD PROFILE
# ============================================================

def load_profile(uploaded_file):

    try:
        raw_bytes = uploaded_file.getvalue()

        # Try several common encodings
        text = None

        for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("Could not decode file.")

        lines = text.splitlines()

        # ----------------------------------------------------
        # Find the appropriate header
        # ----------------------------------------------------

        header_index = None
        distance_col = None
        height_col = None

        for i, line in enumerate(lines):

            lower = line.lower()

            if (
                "lateral" in lower
                and "total profile" in lower
            ):
                header_index = i

                header_parts = line.split(",")

                for j, part in enumerate(header_parts):

                    p = part.strip().lower()

                    if (
                        distance_col is None
                        and (
                            "lateral" in p
                            or "distance" in p
                            or "x" == p
                        )
                    ):
                        distance_col = j

                    if (
                        height_col is None
                        and (
                            "total profile" in p
                            or "height" in p
                            or "profile" in p
                        )
                    ):
                        height_col = j

                break

        # ----------------------------------------------------
        # If no standard header found, look after Data marker
        # ----------------------------------------------------

        if header_index is None:

            data_index = None

            for i, line in enumerate(lines):
                if line.strip().lower() == "data":
                    data_index = i
                    break

            if data_index is not None:

                for i in range(data_index + 1, len(lines)):

                    lower = lines[i].lower()

                    if (
                        "lateral" in lower
                        or "distance" in lower
                    ):

                        header_index = i
                        header_parts = lines[i].split(",")

                        for j, part in enumerate(header_parts):

                            p = part.strip().lower()

                            if (
                                distance_col is None
                                and (
                                    "lateral" in p
                                    or "distance" in p
                                    or "x" == p
                                )
                            ):
                                distance_col = j

                            if (
                                height_col is None
                                and (
                                    "total profile" in p
                                    or "height" in p
                                    or "profile" in p
                                )
                            ):
                                height_col = j

                        break

        # ----------------------------------------------------
        # Generic fallback
        # ----------------------------------------------------

        if header_index is None:

            for i, line in enumerate(lines):

                parts = line.split(",")

                if len(parts) >= 2:

                    try:
                        float(parts[0].strip())
                        float(parts[1].strip())

                        header_index = i - 1
                        distance_col = 0
                        height_col = 1
                        break

                    except ValueError:
                        continue

        if header_index is None:
            raise ValueError(
                "Could not find profile data in this file."
            )

        if distance_col is None:
            distance_col = 0

        if height_col is None:
            height_col = 1

        # ----------------------------------------------------
        # Manually read data
        # This avoids errors from extra trailing commas
        # ----------------------------------------------------

        x_values = []
        y_values = []

        for line in lines[header_index + 1:]:

            if not line.strip():
                continue

            parts = line.split(",")

            if len(parts) <= max(distance_col, height_col):
                continue

            try:

                x = float(parts[distance_col].strip())
                y = float(parts[height_col].strip())

                if np.isfinite(x) and np.isfinite(y):

                    x_values.append(x)
                    y_values.append(y)

            except (ValueError, TypeError):
                continue

        if len(x_values) < 2:
            raise ValueError(
                "Could not find enough numeric profile data."
            )

        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)

        # Sort by X
        order = np.argsort(x)

        x = x[order]
        y = y[order]

        # Remove duplicate X values
        unique_x, unique_indices = np.unique(
            x,
            return_index=True
        )

        x = unique_x
        y = y[unique_indices]

        return x, y

    except Exception as e:

        raise ValueError(str(e))


# ============================================================
# SPIKE REMOVAL
# ============================================================

def remove_spikes(
    height,
    window=11,
    threshold=5.0,
    mode="High spikes only"
):

    y = np.asarray(height, dtype=float).copy()

    n = len(y)

    if n < 5:
        return y, np.zeros(n, dtype=bool)

    # Ensure odd window
    if window % 2 == 0:
        window += 1

    window = max(3, window)

    if window >= n:
        window = n if n % 2 == 1 else n - 1

    if window < 3:
        return y, np.zeros(n, dtype=bool)

    # Local median
    local_median = median_filter(
        y,
        size=window,
        mode="reflect"
    )

    residual = y - local_median

    # Robust MAD
    median_residual = np.median(residual)

    mad = np.median(
        np.abs(residual - median_residual)
    )

    # Convert MAD to sigma-like quantity
    sigma = 1.4826 * mad

    if sigma <= 0 or not np.isfinite(sigma):

        return y, np.zeros(n, dtype=bool)

    if mode == "High spikes only":

        spike_mask = residual > threshold * sigma

    elif mode == "Low spikes only":

        spike_mask = residual < -threshold * sigma

    elif mode == "High + low spikes":

        spike_mask = (
            np.abs(residual)
            > threshold * sigma
        )

    else:

        spike_mask = np.zeros(n, dtype=bool)

    corrected = y.copy()

    corrected[spike_mask] = local_median[spike_mask]

    return corrected, spike_mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(x, y):

    if len(y) == 0:

        return {
            "Ra": np.nan,
            "Rq": np.nan,
            "Rt": np.nan
        }

    # Center profile around mean
    centered = y - np.mean(y)

    Ra = np.mean(np.abs(centered))
    Rq = np.sqrt(np.mean(centered ** 2))
    Rt = np.max(y) - np.min(y)

    return {
        "Ra": Ra,
        "Rq": Rq,
        "Rt": Rt
    }


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_profiles(profiles):

    if len(profiles) == 0:
        return None

    # Find common overlap
    x_mins = [np.min(p["x"]) for p in profiles]
    x_maxs = [np.max(p["x"]) for p in profiles]

    common_min = max(x_mins)
    common_max = min(x_maxs)

    if common_min >= common_max:

        raise ValueError(
            "The profiles do not have a common X-overlap."
        )

    # Determine approximate spacing
    spacings = []

    for p in profiles:

        dx = np.diff(p["x"])

        dx = dx[np.isfinite(dx) & (dx > 0)]

        if len(dx) > 0:
            spacings.append(np.median(dx))

    if len(spacings) == 0:

        raise ValueError(
            "Could not determine X spacing."
        )

    common_dx = np.median(spacings)

    n_points = int(
        np.floor(
            (common_max - common_min)
            / common_dx
        )
    ) + 1

    common_x = (
        common_min
        + np.arange(n_points) * common_dx
    )

    interpolated = []

    for p in profiles:

        y_interp = np.interp(
            common_x,
            p["x"],
            p["y"]
        )

        interpolated.append(y_interp)

    return common_x, np.asarray(interpolated)


# ============================================================
# SESSION STATE
# ============================================================

if "profiles" not in st.session_state:
    st.session_state.profiles = {}


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload Profiles")

uploaded_files = st.file_uploader(
    "Upload profilometer CSV files",
    type=["csv", "txt"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        if filename not in st.session_state.profiles:

            try:

                x, y = load_profile(uploaded_file)

                st.session_state.profiles[filename] = {
                    "original_x": x,
                    "original_y": y
                }

            except Exception as e:

                st.error(
                    f"Could not load {filename}: {e}"
                )


profiles = st.session_state.profiles


if not profiles:

    st.info(
        "Upload one or more profilometer CSV files to begin."
    )

    st.stop()


# ============================================================
# LOADED PROFILE SUMMARY
# ============================================================

st.subheader("Loaded Profiles")

summary_rows = []

for name, p in profiles.items():

    summary_rows.append({
        "File": name,
        "Points": len(p["original_x"]),
        "X Min": np.min(p["original_x"]),
        "X Max": np.max(p["original_x"]),
        "Height Min": np.min(p["original_y"]),
        "Height Max": np.max(p["original_y"])
    })

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL PROFILE ADJUSTMENTS
# ============================================================

st.header("2. Individual Profile Adjustments")

st.markdown("""
These adjustments are applied to each profile **before averaging**.
The left/right trimming therefore removes those points from the
average calculation entirely.
""")


processed_profiles = []

spike_summary = []

for profile_number, (name, p) in enumerate(
    profiles.items()
):

    with st.expander(
        f"{name}",
        expanded=True
    ):

        x_original = p["original_x"]
        y_original = p["original_y"]

        # ----------------------------------------------------
        # X SHIFT
        # ----------------------------------------------------

        x_shift = st.number_input(
            "X shift (µm)",
            value=0.0,
            step=0.1,
            format="%.4f",
            key=f"xshift_{profile_number}_{name}"
        )

        # ----------------------------------------------------
        # Y SHIFT
        # ----------------------------------------------------

        y_shift = st.number_input(
            "Y shift (Å)",
            value=0.0,
            step=1.0,
            format="%.4f",
            key=f"yshift_{profile_number}_{name}"
        )

        # ----------------------------------------------------
        # EDGE TRIMMING
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            left_trim = st.number_input(
                "Remove from LEFT edge (µm)",
                value=0.0,
                min_value=0.0,
                step=1.0,
                format="%.3f",
                key=f"lefttrim_{profile_number}_{name}"
            )

        with col2:

            right_trim = st.number_input(
                "Remove from RIGHT edge (µm)",
                value=0.0,
                min_value=0.0,
                step=1.0,
                format="%.3f",
                key=f"righttrim_{profile_number}_{name}"
            )

        # ----------------------------------------------------
        # SPIKE SETTINGS
        # ----------------------------------------------------

        st.markdown("#### Spike Removal")

        spike_mode = st.selectbox(
            "Spike removal mode",
            [
                "Off",
                "High spikes only",
                "Low spikes only",
                "High + low spikes"
            ],
            index=1,
            key=f"spikemode_{profile_number}_{name}"
        )

        spike_window = st.number_input(
            "Local median window",
            min_value=3,
            max_value=101,
            value=11,
            step=2,
            key=f"spikewindow_{profile_number}_{name}"
        )

        spike_threshold = st.number_input(
            "Spike threshold",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            key=f"spikethreshold_{profile_number}_{name}"
        )

        # ----------------------------------------------------
        # APPLY X/Y SHIFT
        # ----------------------------------------------------

        x_adjusted = x_original + x_shift
        y_adjusted = y_original + y_shift

        # ----------------------------------------------------
        # REMOVE SPIKES
        # ----------------------------------------------------

        if spike_mode == "Off":

            y_corrected = y_adjusted.copy()
            spike_mask = np.zeros(
                len(y_adjusted),
                dtype=bool
            )

        else:

            y_corrected, spike_mask = remove_spikes(
                y_adjusted,
                window=spike_window,
                threshold=spike_threshold,
                mode=spike_mode
            )

        # ----------------------------------------------------
        # EDGE TRIMMING
        # ----------------------------------------------------

        keep_mask = np.ones(
            len(x_adjusted),
            dtype=bool
        )

        x_min = np.min(x_adjusted)
        x_max = np.max(x_adjusted)

        keep_mask &= (
            x_adjusted
            >= x_min + left_trim
        )

        keep_mask &= (
            x_adjusted
            <= x_max - right_trim
        )

        x_final = x_adjusted[keep_mask]
        y_final = y_corrected[keep_mask]
        spikes_final = spike_mask[keep_mask]

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        processed_profiles.append({
            "name": name,
            "x": x_final,
            "y": y_final,
            "spike_mask": spikes_final
        })

        spike_summary.append({
            "File": name,
            "Points Before": len(x_original),
            "Points After Edge Trim": len(x_final),
            "Spikes Removed": int(
                np.sum(spikes_final)
            ),
            "Left Trim (µm)": left_trim,
            "Right Trim (µm)": right_trim,
            "X Shift (µm)": x_shift,
            "Y Shift (Å)": y_shift,
            "Spike Mode": spike_mode,
            "Spike Window": spike_window,
            "Spike Threshold": spike_threshold
        })

        # ----------------------------------------------------
        # INDIVIDUAL CORRECTED PROFILE GRAPH
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            x_final,
            y_final,
            linewidth=1.0,
            label="Corrected profile"
        )

        if np.any(spikes_final):

            ax.scatter(
                x_final[spikes_final],
                y_final[spikes_final],
                s=12,
                label="Removed spikes"
            )

        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("Height (Å)")
        ax.set_title(
            f"{name} — Corrected Profile"
        )

        ax.grid(True, alpha=0.25)
        ax.legend()

        st.pyplot(
            fig,
            clear_figure=True
        )

        st.write(
            f"**Points retained:** {len(x_final):,}  |  "
            f"**Spikes removed:** {np.sum(spikes_final):,}"
        )


# ============================================================
# SPIKE SUMMARY
# ============================================================

st.header("3. Spike Removal Summary")

spike_summary_df = pd.DataFrame(
    spike_summary
)

st.dataframe(
    spike_summary_df,
    use_container_width=True
)


# ============================================================
# ALIGN PROFILES
# ============================================================

st.header("4. Alignment & Averaging")

try:

    common_x, interpolated = interpolate_profiles(
        processed_profiles
    )

except Exception as e:

    st.error(str(e))
    st.stop()


# ------------------------------------------------------------
# ALIGNED PROFILE GRAPH
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 5)
)

for i, p in enumerate(processed_profiles):

    y_interp = interpolated[i]

    ax.plot(
        common_x,
        y_interp,
        linewidth=0.8,
        alpha=0.6,
        label=p["name"]
    )

ax.set_xlabel("Distance (µm)")
ax.set_ylabel("Height (Å)")
ax.set_title("Aligned Profiles")

ax.grid(True, alpha=0.25)

if len(processed_profiles) <= 10:
    ax.legend()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# FULL AVERAGE
# ============================================================

average_y = np.mean(
    interpolated,
    axis=0
)

std_y = np.std(
    interpolated,
    axis=0
)


# ============================================================
# FULL AVERAGE GRAPH
# ============================================================

st.subheader("Full Average Profile")

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    common_x,
    average_y,
    linewidth=1.5,
    label="Average"
)

ax.fill_between(
    common_x,
    average_y - std_y,
    average_y + std_y,
    alpha=0.25,
    label="±1 SD"
)

ax.set_xlabel("Distance (µm)")
ax.set_ylabel("Height (Å)")
ax.set_title("Full Average Profile")

ax.grid(True, alpha=0.25)
ax.legend()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# FINAL AVERAGE TRIMMING
# ============================================================

st.header("5. Final Average Adjustments")

st.markdown("""
These adjustments are applied **after averaging**. They only affect
the final exported/used average profile and do not change the
individual-profile averaging itself.
""")


# ------------------------------------------------------------
# OUTPUT LABEL
# ------------------------------------------------------------

output_label = st.text_input(
    "Final output label",
    value="Y1",
    help=(
        "Enter only the label you want used for the final outputs, "
        "for example Y1, Y2, Sample1, etc."
    )
).strip()


# Make label safe for filenames
safe_output_label = re.sub(
    r"[^\w\-]+",
    "_",
    output_label
).strip("_")

if not safe_output_label:
    safe_output_label = "Profile"


st.caption(
    f"Example output: {safe_output_label}_Final_Average.csv"
)


# ------------------------------------------------------------
# FINAL TRIMMING
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    final_left_trim = st.number_input(
        "Final LEFT truncation (µm)",
        value=0.0,
        min_value=0.0,
        step=1.0,
        format="%.3f",
        key="final_left_trim"
    )

with col2:

    final_right_trim = st.number_input(
        "Final RIGHT truncation (µm)",
        value=0.0,
        min_value=0.0,
        step=1.0,
        format="%.3f",
        key="final_right_trim"
    )


# ============================================================
# APPLY FINAL TRUNCATION
# ============================================================

final_keep = np.ones(
    len(common_x),
    dtype=bool
)

full_x_min = np.min(common_x)
full_x_max = np.max(common_x)

final_keep &= (
    common_x
    >= full_x_min + final_left_trim
)

final_keep &= (
    common_x
    <= full_x_max - final_right_trim
)

final_x = common_x[final_keep]
final_y = average_y[final_keep]
final_std = std_y[final_keep]


if len(final_x) < 2:

    st.error(
        "Final truncation removes too much of the profile. "
        "Please reduce the left/right truncation."
    )

    st.stop()


# ============================================================
# FINAL GRAPH — IMMEDIATELY AFTER ADJUSTMENTS
# ============================================================

st.subheader(
    f"{safe_output_label} - Final Average Profile"
)

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    final_x,
    final_y,
    linewidth=1.8,
    label="Final average"
)

ax.fill_between(
    final_x,
    final_y - final_std,
    final_y + final_std,
    alpha=0.25,
    label="±1 SD"
)

ax.set_xlabel("Distance (µm)")
ax.set_ylabel("Height (Å)")

ax.set_title(
    f"{safe_output_label} - Final Average Profile"
)

ax.grid(True, alpha=0.25)
ax.legend()

st.pyplot(
    fig,
    clear_figure=True
)

st.caption(
    f"Final profile range: "
    f"{final_x.min():.3f} to {final_x.max():.3f} µm  |  "
    f"{len(final_x):,} points"
)


# ============================================================
# FINAL METRICS
# ============================================================

st.header("6. Final Profile Metrics")

metrics = calculate_metrics(
    final_x,
    final_y
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:

    st.metric(
        "Ra",
        f"{metrics['Ra']:.4f} Å"
    )

with metric_col2:

    st.metric(
        "Rq",
        f"{metrics['Rq']:.4f} Å"
    )

with metric_col3:

    st.metric(
        "Rt",
        f"{metrics['Rt']:.4f} Å"
    )


# ============================================================
# EXPORT DATA
# ============================================================

st.header("7. Export Results")


# ------------------------------------------------------------
# FINAL AVERAGE
# ------------------------------------------------------------

final_average_df = pd.DataFrame({
    "Distance_um": final_x,
    "AverageHeight_A": final_y,
    "StdDev_A": final_std
})

final_average_csv = final_average_df.to_csv(
    index=False
).encode("utf-8")


# ------------------------------------------------------------
# FULL AVERAGE
# ------------------------------------------------------------

full_average_df = pd.DataFrame({
    "Distance_um": common_x,
    "AverageHeight_A": average_y,
    "StdDev_A": std_y
})

full_average_csv = full_average_df.to_csv(
    index=False
).encode("utf-8")


# ------------------------------------------------------------
# ALIGNED PROFILES
# ------------------------------------------------------------

aligned_data = {
    "Distance_um": common_x
}

for i, p in enumerate(processed_profiles):

    aligned_data[p["name"]] = interpolated[i]

aligned_df = pd.DataFrame(
    aligned_data
)

aligned_csv = aligned_df.to_csv(
    index=False
).encode("utf-8")


# ------------------------------------------------------------
# CORRECTED INDIVIDUAL PROFILES
# ------------------------------------------------------------

corrected_data = []

for p in processed_profiles:

    temp_df = pd.DataFrame({
        "File": p["name"],
        "Distance_um": p["x"],
        "CorrectedHeight_A": p["y"],
        "SpikeRemoved": p["spike_mask"]
    })

    corrected_data.append(temp_df)

corrected_df = pd.concat(
    corrected_data,
    ignore_index=True
)

corrected_csv = corrected_df.to_csv(
    index=False
).encode("utf-8")


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "Label": safe_output_label,
        "Ra_A": metrics["Ra"],
        "Rq_A": metrics["Rq"],
        "Rt_A": metrics["Rt"],
        "Points": len(final_x),
        "X_Start_um": final_x.min(),
        "X_End_um": final_x.max(),
        "Final_Left_Truncation_um": final_left_trim,
        "Final_Right_Truncation_um": final_right_trim
    }
])

metrics_csv = metrics_df.to_csv(
    index=False
).encode("utf-8")


# ------------------------------------------------------------
# SPIKE SUMMARY
# ------------------------------------------------------------

spike_summary_csv = spike_summary_df.to_csv(
    index=False
).encode("utf-8")


# ============================================================
# DOWNLOAD BUTTONS
# ============================================================

st.subheader("Downloads")

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="Download Final Average",
        data=final_average_csv,
        file_name=f"{safe_output_label}_Final_Average.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="Download Full Average",
        data=full_average_csv,
        file_name=f"{safe_output_label}_Full_Average.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="Download Aligned Profiles",
        data=aligned_csv,
        file_name=f"{safe_output_label}_Aligned_Profiles.csv",
        mime="text/csv",
        use_container_width=True
    )


with download_col2:

    st.download_button(
        label="Download Corrected Individual Profiles",
        data=corrected_csv,
        file_name=(
            f"{safe_output_label}_"
            "Corrected_Individual_Profiles.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="Download Profile Metrics",
        data=metrics_csv,
        file_name=f"{safe_output_label}_Profile_Metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="Download Spike Removal Summary",
        data=spike_summary_csv,
        file_name=(
            f"{safe_output_label}_"
            "Spike_Removal_Summary.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# EXPORT PREVIEW
# ============================================================

st.subheader("Export Naming Preview")

preview_names = pd.DataFrame({
    "Output": [
        "Final Average",
        "Full Average",
        "Aligned Profiles",
        "Corrected Individual Profiles",
        "Profile Metrics",
        "Spike Removal Summary"
    ],
    "Filename": [
        f"{safe_output_label}_Final_Average.csv",
        f"{safe_output_label}_Full_Average.csv",
        f"{safe_output_label}_Aligned_Profiles.csv",
        f"{safe_output_label}_Corrected_Individual_Profiles.csv",
        f"{safe_output_label}_Profile_Metrics.csv",
        f"{safe_output_label}_Spike_Removal_Summary.csv"
    ]
})

st.dataframe(
    preview_names,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# SPIKE REMOVAL GUIDE
# ============================================================

with st.expander("Spike Removal Guide"):

    st.markdown("""
### Spike removal

**Off**
- No spike removal is performed.

**High spikes only**
- Removes isolated upward spikes.
- This is the recommended default if your main problem is
  occasional high profilometer spikes.

**Low spikes only**
- Removes isolated downward spikes.

**High + low spikes**
- Removes both upward and downward isolated spikes.

### Local median window

This determines how many neighboring points are used to estimate
the local expected profile.

Typical starting values:

- 7–11: aggressive/local correction
- 11–21: good general starting range
- 21–51: smoother profiles with broader features

### Spike threshold

Higher values are more conservative.

Typical starting values:

- 3–4: aggressive
- 5: good default
- 6–8: conservative
- 10+: only very extreme spikes

The removed points are shown in red on the individual profile plots.
""")


# ============================================================
# END
# ============================================================

st.success(
    f"Ready — final output label is **{safe_output_label}**."
)