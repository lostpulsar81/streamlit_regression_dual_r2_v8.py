import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("Interactive Regression Analysis")
st.caption(
    "Fit linear or polynomial models to group means or raw replicates, while plotting mean ± error and an overlaid regression curve."
)


def poly(x, *coeffs):
    x = np.asarray(x, dtype=float)
    return sum(c * x**i for i, c in enumerate(coeffs))


def format_equation(coeffs):
    parts = []
    for i, c in enumerate(coeffs):
        if i == 0:
            parts.append(f"{c:.6g}")
        elif i == 1:
            parts.append(f"{c:+.6g}·x")
        else:
            parts.append(f"{c:+.6g}·x^{i}")
    return "y = " + " ".join(parts)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def overall_model_test(y_true, y_pred, degree):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    n = len(y_true)
    predictors = degree
    denom_df = n - (predictors + 1)

    if predictors <= 0 or denom_df <= 0:
        return np.nan, np.nan

    if ss_res <= 0:
        return np.inf, 0.0

    f_stat = ((ss_tot - ss_res) / predictors) / (ss_res / denom_df)
    p_value = 1 - stats.f.cdf(f_stat, predictors, denom_df)
    return f_stat, p_value


def fit_model(x, y, degree, sigma=None, absolute_sigma=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = np.ones(degree + 1, dtype=float)

    params, cov = curve_fit(
        lambda xx, *a: poly(xx, *a),
        x,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=20000,
    )

    param_errors = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(params), np.nan)
    y_pred = poly(x, *params)
    return params, param_errors, y_pred


def build_raw_vectors(concentrations, data_df):
    x_blocks = []
    y_blocks = []

    for idx, conc in enumerate(concentrations):
        column_values = pd.to_numeric(data_df.iloc[:, idx], errors="coerce").dropna().to_numpy(dtype=float)
        if column_values.size == 0:
            continue
        x_blocks.append(np.full(column_values.shape, float(conc), dtype=float))
        y_blocks.append(column_values)

    if not x_blocks:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(x_blocks), np.concatenate(y_blocks)


def choose_vector(option_name, std_values, sem_values):
    if option_name == "SD":
        return std_values
    if option_name == "SEM":
        return sem_values
    return None


def get_numeric_bar_width(concentrations):
    unique_sorted = np.unique(np.asarray(concentrations, dtype=float))
    if unique_sorted.size > 1:
        diffs = np.diff(unique_sorted)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            return 0.6 * float(np.min(positive_diffs))

    x_range = float(np.max(unique_sorted) - np.min(unique_sorted)) if unique_sorted.size > 0 else 1.0
    return max(1.0, 0.08 * x_range if x_range > 0 else 1.0)


def add_confined_peak_annotation(ax, x, y, text, font_size, color="orange"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_rel = 0.5 if x1 == x0 else (x - x0) / (x1 - x0)
    y_rel = 0.5 if y1 == y0 else (y - y0) / (y1 - y0)

    dx = -14 if x_rel > 0.78 else 14
    dy = -14 if y_rel > 0.86 else 14
    ha = "right" if dx < 0 else "left"
    va = "top" if dy < 0 else "bottom"

    ann = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=max(font_size, 8),
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8),
        arrowprops=dict(arrowstyle="-", color=color, lw=1),
        annotation_clip=True,
        zorder=6,
    )
    ann.set_clip_on(True)
    return ann


def needs_tick_rotation(labels, positions):
    if len(labels) <= 1:
        return False
    max_label_len = max(len(str(lbl)) for lbl in labels)
    pos = np.asarray(positions, dtype=float)
    diffs = np.diff(np.unique(pos))
    min_diff = np.min(diffs) if diffs.size > 0 else 1.0
    total_span = float(np.max(pos) - np.min(pos)) if pos.size > 1 else 1.0
    crowd_ratio = min_diff / total_span if total_span > 0 else 1.0
    return max_label_len >= 4 and crowd_ratio < 0.08


def build_display_mapping(concentrations, mode):
    concentrations = np.asarray(concentrations, dtype=float)

    if mode == "Categorical (equal spacing, visual only)":
        positions = np.arange(len(concentrations), dtype=float)

        def mapper(values):
            values = np.asarray(values, dtype=float)
            return np.interp(values, concentrations, positions)

        return positions, mapper, 0.6

    positions = concentrations.astype(float)

    def mapper(values):
        return np.asarray(values, dtype=float)

    return positions, mapper, get_numeric_bar_width(concentrations)


excel_file = None
sheet_name = None

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Data")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            available_sheets = excel_file.sheet_names
            if not available_sheets:
                st.error("The uploaded Excel file does not contain any visible sheets.")
                st.stop()
            sheet_name = st.selectbox("Sheet name", available_sheets, index=0)
        except Exception as exc:
            st.error(f"Unable to read the Excel file or its sheet names: {exc}")
            st.stop()
    else:
        st.text_input("Sheet name", value="", disabled=True, placeholder="Upload a file to load the first sheet automatically")

    st.subheader("Model")
    line_color = st.color_picker("Regression curve color", value="#1f77b4")
    chart_type = st.selectbox("Chart type", ["Line", "Bar"])
    x_display_mode = st.selectbox(
        "X-axis display",
        ["Numeric (true spacing)", "Categorical (equal spacing, visual only)"],
        index=0,
        help="Numeric keeps the real concentration spacing. Categorical spreads labels evenly only for display, while the fit still uses the true concentration values.",
    )
    model_type = st.selectbox("Regression type", ["Linear", "Polynomial"], index=1)
    degree = 1 if model_type == "Linear" else st.slider("Polynomial degree", 2, 10, 2)
    fit_basis = st.selectbox(
        "Fit based on",
        ["Group means", "Raw replicates"],
        help="Group means fits one mean value per concentration. Raw replicates fits all individual replicate values.",
    )

    if "show_raw_points" not in st.session_state:
        st.session_state.show_raw_points = False

    if fit_basis == "Raw replicates":
        st.session_state.show_raw_points = True

    display_error_type = st.selectbox(
        "Error bars shown on mean points",
        ["SD", "SEM"],
        index=0,
        help="SD shows the spread of replicate values. SEM shows the uncertainty of the mean.",
    )
    weight_basis = st.selectbox(
        "Weights for mean fit",
        ["None", "SEM", "SD"],
        index=1,
        disabled=(fit_basis == "Raw replicates"),
        help="Used only when fitting group means. Recommended: show SD on the graph, but weight the mean fit with SEM.",
    )
    show_raw_points = st.checkbox(
        "Show raw replicate points",
        key="show_raw_points",
        disabled=(fit_basis == "Raw replicates"),
        help="Automatically enabled when the fit is based on raw replicates.",
    )

    st.subheader("Text")
    title = st.text_input("Plot title", f"{sheet_name} regression" if sheet_name else "Regression analysis")
    x_label_override = st.text_input("X-axis label", "")
    y_label = st.text_input("Y-axis label", sheet_name if sheet_name else "Response variable")
    legend_label = "Regression curve"
    data_label = "Mean ± error"

    st.subheader("Fonts")
    font_size_legend = st.slider("Legend font size", 6, 24, 10)
    font_size_tick = st.slider("Axis tick font size", 6, 24, 10)
    font_size_axis_titles = st.slider("Axis title font size", 6, 24, 12)
    font_size_title = st.slider("Plot title font size", 6, 30, 14)
    font_size_stats_box = st.slider("Stats box font size", 6, 24, 9)

    st.subheader("Saving")
    file_name = st.text_input("File name", value="regression_plot_dual_r2_v5")
    save_format = st.selectbox("Format", ["png", "jpg"], index=0)

with right_col:
    with st.expander("Description and recommended use", expanded=False):
        st.write(
            "- **Line** draws mean points with error bars and a continuous regression curve.\n"
            "- **Bar** shows mean bars with error bars and overlays the regression curve on the same plot.\n"
            "- **Numeric (true spacing)** keeps the real concentration spacing on the X axis.\n"
            "- **Categorical (equal spacing, visual only)** spreads concentrations evenly for readability, but the fit itself is still calculated with the real numeric concentrations.\n"
            "- **Group means** fits the curve to one mean value per concentration.\n"
            "- **Raw replicates** fits the curve to all individual experimental values.\n"
            "- When **Raw replicates** is selected, **Weights for mean fit** is disabled and **Show raw replicate points** is automatically enabled.\n"
            "- **Error bars shown on mean points** controls only the plotted bars/points.\n"
            "- **Weights for mean fit** controls only the weighting of the regression when fitting means.\n"
            "- Recommended setup for many biological datasets: show **SD** on the graph to represent real variability, and use **SEM** as weights if you fit the group means."
        )
        st.write(
            "**R² meanings**\n"
            "- **R² (fit basis)**: fit quality on the same data used to estimate the model.\n"
            "- **R² (means)**: how well the curve follows the concentration means.\n"
            "- **R² (raw)**: how well the same curve explains all individual replicates."
        )

    if uploaded_file and sheet_name:
        try:
            raw_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            if raw_df.shape[0] < 2 or raw_df.shape[1] < 2:
                st.error(
                    "The selected sheet must contain at least 2 rows and 2 columns: concentrations in row 1 starting from A1, and replicate data below."
                )
                st.stop()

            concentration_cells = raw_df.iloc[0, :]
            try:
                concentrations = pd.to_numeric(concentration_cells, errors="raise").to_numpy(dtype=float)
            except Exception:
                st.error("All cells in the first row, starting from A1, must be numeric concentration values.")
                st.stop()

            data_df = raw_df.iloc[1:, :].apply(pd.to_numeric, errors="coerce")
            data_df = data_df.dropna(how="all")

            if data_df.empty:
                st.error("No replicate data found below the concentration row.")
                st.stop()

            mean_heights = data_df.mean().to_numpy(dtype=float)
            std_heights = data_df.std(ddof=1).to_numpy(dtype=float)
            counts = data_df.count().to_numpy(dtype=float)
            sem_heights = np.divide(
                std_heights,
                np.sqrt(counts),
                out=np.full_like(std_heights, np.nan, dtype=float),
                where=counts > 0,
            )

            if len(mean_heights) != len(concentrations):
                st.error(
                    f"Incompatible number of points: the sheet provides {len(mean_heights)} data columns, but {len(concentrations)} concentration values were found."
                )
                st.stop()

            if len(concentrations) < 2:
                st.error("At least two concentration columns are required.")
                st.stop()

            x_raw, y_raw = build_raw_vectors(concentrations, data_df)
            if x_raw.size == 0:
                st.error("No valid raw replicate values were found.")
                st.stop()

            displayed_error_values = choose_vector(display_error_type, std_heights, sem_heights)
            x_label = x_label_override.strip() if x_label_override.strip() else "Concentration"

            fit_on_means = fit_basis == "Group means"
            effective_weight_basis = weight_basis if fit_on_means else "Ignored (raw replicate fit)"
            sigma = None
            absolute_sigma = False
            weighting_description = "unweighted"

            if fit_on_means:
                x_fit_data = concentrations
                y_fit_data = mean_heights

                if weight_basis != "None":
                    sigma = choose_vector(weight_basis, std_heights, sem_heights)
                    if sigma is None or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
                        st.error(
                            f"For weighted regression on means, all {weight_basis} values must be finite and greater than zero."
                        )
                        st.stop()
                    absolute_sigma = True
                    weighting_description = f"weighted by {weight_basis}"
            else:
                x_fit_data = x_raw
                y_fit_data = y_raw

            params, param_errors, y_pred_fit_basis = fit_model(
                x_fit_data,
                y_fit_data,
                degree,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
            )

            x_fit = np.linspace(float(np.min(concentrations)), float(np.max(concentrations)), 600)
            y_fit = poly(x_fit, *params)
            y_pred_means = poly(concentrations, *params)
            y_pred_raw = poly(x_raw, *params)

            r2_fit_basis = compute_r2(y_fit_data, y_pred_fit_basis)
            r2_means = compute_r2(mean_heights, y_pred_means)
            r2_raw = compute_r2(y_raw, y_pred_raw)
            f_stat, p_value = overall_model_test(y_fit_data, y_pred_fit_basis, degree)

            max_index = int(np.argmax(y_fit))
            max_concentration = float(x_fit[max_index])
            max_height = float(y_fit[max_index])
            equation = format_equation(params)

            max_data_index = int(np.argmax(mean_heights))
            max_data_concentration = float(concentrations[max_data_index])
            max_data_height = float(mean_heights[max_data_index])

            display_positions, display_mapper, bar_width = build_display_mapping(concentrations, x_display_mode)
            x_plot_means = display_mapper(concentrations)
            x_plot_raw = display_mapper(x_raw)
            x_plot_fit = display_mapper(x_fit)
            x_plot_max = float(display_mapper(np.array([max_concentration], dtype=float))[0])

            figure_width = 11 if x_display_mode == "Categorical (equal spacing, visual only)" else 12
            fig, ax = plt.subplots(figsize=(figure_width, 6.5))

            rng = np.random.default_rng(0)

            if chart_type == "Bar":
                ax.bar(
                    x_plot_means,
                    mean_heights,
                    width=bar_width,
                    yerr=displayed_error_values,
                    capsize=5,
                    color="#d9d9d9",
                    edgecolor="black",
                    linewidth=1,
                    label=f"{data_label} ({display_error_type})",
                    zorder=2,
                )
                if show_raw_points:
                    jitter = 0.18 * bar_width
                    ax.scatter(
                        x_plot_raw + rng.uniform(-jitter, jitter, size=x_raw.size),
                        y_raw,
                        alpha=0.35,
                        s=24,
                        label="Raw replicates",
                        zorder=3,
                    )
            else:
                if show_raw_points:
                    jitter = 0.045 if x_display_mode == "Categorical (equal spacing, visual only)" else 0.0
                    raw_x_for_plot = x_plot_raw + rng.uniform(-jitter, jitter, size=x_raw.size) if jitter > 0 else x_plot_raw
                    ax.scatter(
                        raw_x_for_plot,
                        y_raw,
                        alpha=0.25,
                        s=28,
                        label="Raw replicates",
                        zorder=2,
                    )

                ax.errorbar(
                    x_plot_means,
                    mean_heights,
                    yerr=displayed_error_values,
                    fmt="o",
                    capsize=5,
                    color="black",
                    markersize=6,
                    label=f"{data_label} ({display_error_type})",
                    zorder=4,
                )

            ax.plot(x_plot_fit, y_fit, color=line_color, label=legend_label, linewidth=2, zorder=5)

            ax.axvline(
                x=x_plot_max,
                linestyle="--",
                color="orange",
                label=f"Max reg: {max_concentration:.1f} {x_label}",
                zorder=4,
            )
            ax.scatter(x_plot_max, max_height, color="orange", zorder=6)

            if x_display_mode == "Categorical (equal spacing, visual only)":
                x_margin = 0.5
            else:
                x_range = float(np.max(concentrations) - np.min(concentrations))
                x_margin = max(0.05 * x_range, 0.75 * bar_width if chart_type == "Bar" else 0.0, 1.0)
            ax.set_xlim(float(np.min(x_plot_means) - x_margin), float(np.max(x_plot_means) + x_margin))

            upper_candidates = [mean_heights + displayed_error_values, y_fit]
            lower_candidates = [mean_heights - displayed_error_values, y_fit]
            if show_raw_points:
                upper_candidates.append(y_raw)
                lower_candidates.append(y_raw)
            y_upper = float(np.nanmax(np.concatenate(upper_candidates)))
            y_lower = float(np.nanmin(np.concatenate(lower_candidates)))
            y_range = y_upper - y_lower
            y_margin_top = 0.16 * y_range if y_range > 0 else max(abs(y_upper), 1.0) * 0.16 + 1.0
            y_margin_bottom = 0.08 * y_range if y_range > 0 else max(abs(y_lower), 1.0) * 0.08 + 0.5
            ax.set_ylim(y_lower - y_margin_bottom, y_upper + y_margin_top)

            add_confined_peak_annotation(
                ax,
                x_plot_max,
                max_height,
                f"Max reg: {max_concentration:.1f} {x_label}",
                font_size_tick,
                color="orange",
            )

            fit_basis_label = "means" if fit_on_means else "raw replicates"
            stats_text = (
                f"Fit basis: {fit_basis_label} ({weighting_description})\n"
                f"R² (fit basis): {r2_fit_basis:.4f}\n"
                f"R² (means): {r2_means:.4f}\n"
                f"R² (raw): {r2_raw:.4f}\n"
                f"p-value: {p_value:.4g}\n"
                f"x_opt: {max_concentration:.2f} {x_label}\n"
                f"{equation}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=font_size_stats_box,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
            )

            ax.set_xlabel(x_label, fontsize=font_size_axis_titles)
            ax.set_ylabel(y_label, fontsize=font_size_axis_titles)
            ax.set_title(title, fontsize=font_size_title)
            ax.tick_params(axis="both", labelsize=font_size_tick)
            ax.set_xticks(x_plot_means)
            xtick_labels = [f"{c:g}" for c in concentrations]
            rotation = 45 if needs_tick_rotation(xtick_labels, x_plot_means) else 0
            ha = "right" if rotation else "center"
            ax.set_xticklabels(xtick_labels, rotation=rotation, ha=ha)
            ax.legend(loc="upper right", fontsize=font_size_legend, frameon=True)
            ax.grid(True, axis="both", alpha=0.35, zorder=1)
            fig.tight_layout()

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format=save_format, dpi=300, bbox_inches="tight")
            img_buffer.seek(0)

            st.pyplot(fig, clear_figure=False)

            download_spacer, download_col = st.columns([5, 1])
            with download_col:
                st.download_button(
                    label=f"Download plot ({save_format.upper()})",
                    data=img_buffer,
                    file_name=f"{file_name}.{save_format}",
                    mime=f"image/{'jpeg' if save_format == 'jpg' else 'png'}",
                )

            st.subheader("Results")
            res_col1, res_col2, res_col3 = st.columns([1, 1.2, 1.1])

            with res_col1:
                with st.container(border=True):
                    st.markdown("**Results**")
                    st.write(f"**Chart type:** {chart_type}")
                    st.write(f"**X-axis display:** {x_display_mode}")
                    st.write(f"**Fit basis:** {fit_basis_label} ({weighting_description})")
                    st.write(f"**Displayed error bars:** {display_error_type}")
                    st.write(f"**Weights for mean fit:** {effective_weight_basis}")
                    st.write(f"**Equation:** {equation}")
                    st.write(f"**R² (fit basis):** {r2_fit_basis:.6f}")
                    st.write(f"**R² (means):** {r2_means:.6f}")
                    st.write(f"**R² (raw):** {r2_raw:.6f}")
                    st.write(f"**p-value:** {p_value:.6g}")
                    st.write(f"**F statistic:** {f_stat:.6g}" if np.isfinite(f_stat) else "**F statistic:** not available")
                    st.write(f"**Max reg:** {max_concentration:.3f} {x_label}")
                    st.write(f"**Max data:** {max_data_concentration:.3f} {x_label}")

            with res_col2:
                with st.container(border=True):
                    st.markdown("**Definitions**")
                    st.write("- **Max reg** = maximum estimated from the fitted regression curve within the tested X range.")
                    st.write("- **Max data** = highest observed group mean among the tested concentrations.")
                    st.write("- **R² (fit basis)** = goodness of fit computed on the same data used to estimate the model parameters.")
                    st.write("- **R² (means)** = how well the curve describes the trend of the concentration means.")
                    st.write("- **R² (raw)** = how well the same curve explains the dispersion of all individual replicates.")
                    st.write("- **Displayed error bars** describe what is shown visually on the graph only.")
                    st.write("- **Weights for mean fit** affect only the regression on group means, not the appearance of the bars or points.")
                    st.write("- A high **R² (means)** with a lower **R² (raw)** usually means the curve follows the average trend well, but the biological/experimental variability around that trend is still high.")
                    st.write("- Recommended setup when fitting means: show **SD** on the graph and weight the regression with **SEM**.")
                    st.write("- When fitting **raw replicates**, the app forces display of raw points and disables **Weights for mean fit**, because those weights are only meaningful for fits on group means.")
                    st.write("- **Categorical (equal spacing, visual only)** improves readability when concentrations are very unevenly spaced, but the regression is still fitted using the real numeric concentrations.")

            with res_col3:
                with st.container(border=True):
                    st.markdown("**Interpretation help**")
                    st.write(
                        "Use ANOVA/post hoc to decide which tested concentration performs best statistically. "
                        "Use the regression curve and x_opt as an interpolated estimate of the optimum between tested concentrations."
                    )
                    st.write(
                        "- A higher **R² (means)** than **R² (raw)** usually means the curve follows the average trend well, "
                        "but individual replicate variability is still high."
                    )
                    st.write(
                        "- If you fit **Group means**, using **SEM** as weights is usually more appropriate than **SD** when you want weights to reflect the uncertainty of each mean."
                    )
                    st.write(
                        "- If you fit **Raw replicates**, the fit reflects all individual points directly, so weights for mean fit are not used."
                    )

        except Exception as e:
            st.error(f"Error while loading or analyzing the file: {e}")
    else:
        st.info("Upload an Excel file to load the first sheet automatically and display the plot.")
