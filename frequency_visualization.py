import argparse
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid matplotlib tkinter crashes in multi-threaded/batch environments on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import colorsys

try:
    import seaborn as sns
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: seaborn. Please install it in your environment first (pip install seaborn)."
    ) from exc


REQUIRED_SHEETS = {
    "run_summary",
    "object_frequency",
    "field_frequency",
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, out_file: Path) -> None:
    # Use bbox_inches instead of tight_layout to avoid layout warnings on dense labels.
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _legend_bottom(
    ax: plt.Axes,
    title: str,
    max_cols: int = 4,
    fontsize: int = 8,
    wrap_width: int = 16,
    top_y: float = -0.1,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not labels:
        return

    # Deduplicate while preserving order.
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)

    wrapped_labels = ["\n".join(textwrap.wrap(str(x), width=wrap_width, break_long_words=False, break_on_hyphens=False)) for x in uniq_labels]
    ncol = max(1, min(max_cols, len(wrapped_labels)))
    ax.legend(
        uniq_handles,
        wrapped_labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, top_y),
        ncol=ncol,
        fontsize=fontsize,
        title_fontsize=max(8, fontsize),
        frameon=True,
        borderaxespad=0.0,
        labelspacing=0.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _wrap_label(label: str, width: int = 18) -> str:
    text = str(label)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _blue_red_cmap() -> mcolors.Colormap:
    # High values -> red, low values -> blue.
    # Use warm red and softened blue with a gentle neutral transition.
    color_list = [
        "#5B84B1",  # low: soft blue
        "#D9E6F2",  # light blue
        "#F4E6DE",  # warm neutral
        "#E47C6B",  # warm red
    ]
    return mcolors.LinearSegmentedColormap.from_list("warm_blue_to_red", color_list, N=256)


def _blue_red_palette(n: int) -> List:
    cmap = _blue_red_cmap()
    if n <= 1:
        return [cmap(0.6)]
    return [cmap(v) for v in np.linspace(0.05, 0.95, n)]


def _red_blue_palette(n: int) -> List:
    # For bar charts: top bars red -> bottom bars blue (soft, not too dark).
    base = _blue_red_cmap()
    # Descending on this cmap gives red->blue for top-to-bottom bars.
    trim = base(np.linspace(0.95, 0.05, max(n, 2)))
    if n <= 1:
        return [trim[0]]
    return [trim[i] for i in range(n)]


def _adjust_lightness(color, factor: float):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.15, min(0.88, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def _object_family_field_palette(field_labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
    labels = [str(x) for x in field_labels]
    if not labels:
        return {}

    obj_to_fields: Dict[str, List[str]] = {}
    for label in labels:
        obj = label.split(".", 1)[0].strip()
        if obj not in obj_to_fields:
            obj_to_fields[obj] = []
        obj_to_fields[obj].append(label)

    objects = sorted(obj_to_fields.keys())
    base_colors = sns.color_palette("tab10", n_colors=max(len(objects), 1))

    palette: Dict[str, Tuple[float, float, float]] = {}
    for i, obj in enumerate(objects):
        fields = sorted(set(obj_to_fields[obj]))
        base = base_colors[i % len(base_colors)]

        if len(fields) == 1:
            palette[fields[0]] = tuple(base)
            continue

        factors = np.linspace(0.75, 1.25, len(fields))
        for field_name, factor in zip(fields, factors):
            palette[field_name] = tuple(_adjust_lightness(base, float(factor)))

    return palette


def _normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    if "run_status" not in df.columns:
        return df.copy()
    out = df.copy()
    out["run_status"] = out["run_status"].astype(str).str.lower().str.strip()
    return out


def _filter_success(df: pd.DataFrame) -> pd.DataFrame:
    if "run_status" not in df.columns:
        return df.copy()
    mask = df["run_status"].isna() | (df["run_status"] == "success")
    return df.loc[mask].copy()


def _normalize_frequency_by_iterations(freq_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize each workflow-level frequency by its executed iteration count.
    # This makes frequencies comparable when workflows stop at different rounds.
    out = freq_df.copy()
    if "frequency" not in out.columns or "run_key" not in out.columns:
        return out

    run_meta = run_df[["run_key", "iterations_executed"]].drop_duplicates().copy()
    run_meta["iterations_executed"] = pd.to_numeric(run_meta["iterations_executed"], errors="coerce")

    out = out.merge(run_meta, on="run_key", how="left")
    out["raw_frequency"] = pd.to_numeric(out["frequency"], errors="coerce")

    valid_den = out["iterations_executed"] > 0
    out["frequency_ratio"] = np.where(valid_den, out["raw_frequency"] / out["iterations_executed"], np.nan)
    out["frequency_pct"] = out["frequency_ratio"] * 100.0

    # Keep downstream code unchanged by replacing frequency with normalized percent.
    out["frequency"] = out["frequency_pct"]
    return out


def _extract_run_order(run_tag: str) -> float:
    if pd.isna(run_tag):
        return np.nan
    text = str(run_tag)
    match = re.search(r"(\d+)", text)
    if not match:
        return np.nan
    return float(match.group(1))


def discover_city_excels(root_dir: Path) -> List[Path]:
    files = sorted(root_dir.glob("*/*_统计汇总.xlsx"))
    files = [p for p in files if not p.name.startswith("~$")]
    if files:
        return files
    # Fallback: recursive search for any xlsx under city folders.
    return sorted([p for p in root_dir.rglob("*.xlsx") if not p.name.startswith("~$")])


def load_city_workbook(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    sheet_names = set(xls.sheet_names)
    missing = REQUIRED_SHEETS - sheet_names
    if missing:
        raise ValueError(f"Workbook missing required sheets: {sorted(missing)} | file={path}")

    run_summary = pd.read_excel(path, sheet_name="run_summary")
    object_freq = pd.read_excel(path, sheet_name="object_frequency")
    field_freq = pd.read_excel(path, sheet_name="field_frequency")

    for df in (run_summary, object_freq, field_freq):
        df["source_file"] = str(path)

    return run_summary, object_freq, field_freq


def load_all_data(root_dir: Path) -> Dict[str, pd.DataFrame]:
    excel_files = discover_city_excels(root_dir)
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found under: {root_dir}")

    run_list: List[pd.DataFrame] = []
    obj_list: List[pd.DataFrame] = []
    fld_list: List[pd.DataFrame] = []

    for file_path in excel_files:
        try:
            run_df, obj_df, fld_df = load_city_workbook(file_path)
            run_list.append(run_df)
            obj_list.append(obj_df)
            fld_list.append(fld_df)
            print(f"[OK] Loaded: {file_path}")
        except Exception as exc:
            print(f"[WARN] Skip file due to error: {file_path} | {exc}")

    if not run_list:
        raise RuntimeError("No valid workbook loaded. Please check Excel structure.")

    run_all = pd.concat(run_list, ignore_index=True)
    obj_all = pd.concat(obj_list, ignore_index=True)
    fld_all = pd.concat(fld_list, ignore_index=True)

    run_all = _normalize_status(run_all)
    obj_all = _normalize_status(obj_all)
    fld_all = _normalize_status(fld_all)

    run_all = _filter_success(run_all)
    obj_all = _filter_success(obj_all)
    fld_all = _filter_success(fld_all)

    # Numeric conversion.
    if "best_saving_pct_total_site" in run_all.columns:
        run_all["best_saving_pct_total_site"] = pd.to_numeric(
            run_all["best_saving_pct_total_site"], errors="coerce"
        )
    if "iterations_executed" in run_all.columns:
        run_all["iterations_executed"] = pd.to_numeric(run_all["iterations_executed"], errors="coerce")
    obj_all["frequency"] = pd.to_numeric(obj_all.get("frequency"), errors="coerce")
    fld_all["frequency"] = pd.to_numeric(fld_all.get("frequency"), errors="coerce")

    # Create run key for robust merging.
    for df in (run_all, obj_all, fld_all):
        for col in ("city", "run_tag", "workflow_id"):
            if col not in df.columns:
                df[col] = np.nan
        df["city"] = df["city"].astype(str)
        df["run_tag"] = df["run_tag"].astype(str)
        df["workflow_id"] = df["workflow_id"].astype(str)
        df["run_key"] = df["city"] + "|" + df["run_tag"] + "|" + df["workflow_id"]

    run_all["run_order"] = run_all["run_tag"].apply(_extract_run_order)
    obj_all["run_order"] = obj_all["run_tag"].apply(_extract_run_order)
    fld_all["run_order"] = fld_all["run_tag"].apply(_extract_run_order)

    obj_all = _normalize_frequency_by_iterations(obj_all, run_all)
    fld_all = _normalize_frequency_by_iterations(fld_all, run_all)

    return {
        "run_summary": run_all,
        "object_frequency": obj_all,
        "field_frequency": fld_all,
    }


def _set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    # Chinese font fallback chain.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_top_bar(freq_df: pd.DataFrame, item_col: str, city: str, out_file: Path, top_n: int) -> None:
    city_df = freq_df[freq_df["city"] == city].copy()
    if city_df.empty:
        return

    agg = (
        city_df.groupby(item_col, as_index=False)["frequency"]
        .sum()
        .sort_values("frequency", ascending=False)
        .head(top_n)
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = _red_blue_palette(len(agg))
    sns.barplot(
        data=agg,
        y=item_col,
        x="frequency",
        hue=item_col,
        legend=False,
        ax=ax,
        palette=bar_colors,
    )
    ax.set_title(f"{city} | Top {top_n} {item_col} by Frequency")
    ax.set_xlabel("Total Normalized Frequency (%)")
    ax.set_ylabel(item_col)
    _save_figure(fig, out_file)


def plot_city_item_heatmap(freq_df: pd.DataFrame, item_col: str, out_file: Path, top_n: int) -> None:
    total = freq_df.groupby(item_col)["frequency"].sum().sort_values(ascending=False).head(top_n)
    if total.empty:
        return

    top_items = total.index.tolist()
    sub = freq_df[freq_df[item_col].isin(top_items)].copy()

    pivot = (
        sub.pivot_table(index=item_col, columns="city", values="frequency", aggfunc="mean", fill_value=0.0)
        .sort_index(axis=0)
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(16, len(top_items) * 1.0), max(7, len(pivot.index) * 1.1)))
    heat_cmap = _blue_red_cmap()
    sns.heatmap(pivot, cmap=heat_cmap, ax=ax, cbar_kws={"label": "Normalized Frequency (%)"})
    ax.set_title(f"City vs {item_col} Normalized Frequency Heatmap (%) (Top {top_n})")
    ax.set_xlabel("City", fontsize=16)
    ax.set_ylabel(item_col, fontsize=16)
    # Requested axis label orientation: y labels horizontal, x labels 45 degrees.
    wrapped_y = [_wrap_label(t.get_text(), width=18) for t in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    ax.tick_params(axis="x", pad=2)
    _save_figure(fig, out_file)


def build_item_metrics(freq_df: pd.DataFrame, run_df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    run_cols = ["run_key", "city", "run_tag", "workflow_id", "best_saving_pct_total_site"]
    run_sub = run_df[run_cols].drop_duplicates().copy()

    merged = freq_df.merge(run_sub, on=["run_key", "city", "run_tag", "workflow_id"], how="left")
    merged = merged.dropna(subset=["frequency"])
    merged["best_saving_pct_total_site"] = pd.to_numeric(merged["best_saving_pct_total_site"], errors="coerce")
    merged["weighted_value"] = merged["frequency"] * merged["best_saving_pct_total_site"]

    # Hierarchical aggregation for comparability:
    # 1) per city: summarize run-level normalized frequency and frequency-weighted efficiency
    # 2) across cities: take median of city summaries (equal city weight, robust to outliers)
    city_item = merged.groupby(["city", item_col], as_index=False).agg(
        city_median_frequency=("frequency", "median"),
        city_mean_frequency=("frequency", "mean"),
        city_run_coverage=("run_key", "nunique"),
        city_weighted_value=("weighted_value", "sum"),
        city_total_frequency=("frequency", "sum"),
        city_mean_best_saving=("best_saving_pct_total_site", "mean"),
    )
    city_item["city_efficiency_per_frequency"] = (
        city_item["city_weighted_value"] / city_item["city_total_frequency"].replace(0, np.nan)
    )

    grp = city_item.groupby(item_col)
    metrics = grp.agg(
        total_frequency=("city_median_frequency", "median"),
        total_frequency_q25=("city_median_frequency", lambda s: s.quantile(0.25)),
        total_frequency_q75=("city_median_frequency", lambda s: s.quantile(0.75)),
        mean_frequency=("city_mean_frequency", "median"),
        run_coverage=("city_run_coverage", "sum"),
        city_coverage=("city", "nunique"),
        weighted_contribution=("city_weighted_value", "median"),
        mean_best_saving_when_present=("city_mean_best_saving", "median"),
        std_best_saving_when_present=("city_mean_best_saving", "std"),
        efficiency_per_frequency=("city_efficiency_per_frequency", "median"),
        efficiency_per_frequency_q25=("city_efficiency_per_frequency", lambda s: s.quantile(0.25)),
        efficiency_per_frequency_q75=("city_efficiency_per_frequency", lambda s: s.quantile(0.75)),
    ).reset_index()

    metrics["total_frequency_iqr"] = (
        metrics["total_frequency_q75"] - metrics["total_frequency_q25"]
    )
    metrics["efficiency_per_frequency_iqr"] = (
        metrics["efficiency_per_frequency_q75"] - metrics["efficiency_per_frequency_q25"]
    )

    metrics = metrics.sort_values("weighted_contribution", ascending=False)

    return metrics, merged


def _build_palette_map(df: pd.DataFrame, hue_col: str, palette) -> Dict[str, Tuple[float, float, float]]:
    labels = [x for x in pd.unique(df[hue_col].dropna())]
    if not labels:
        return {}
    if isinstance(palette, dict):
        return {str(k): v for k, v in palette.items()}
    colors = sns.color_palette(palette, n_colors=len(labels))
    return {str(k): c for k, c in zip(labels, colors)}


def _add_iqr_errorbars(
    ax: plt.Axes,
    df: pd.DataFrame,
    hue_col: str,
    palette_map: Dict[str, Tuple[float, float, float]],
) -> None:
    needed = [
        "total_frequency",
        "efficiency_per_frequency",
        "total_frequency_q25",
        "total_frequency_q75",
        "efficiency_per_frequency_q25",
        "efficiency_per_frequency_q75",
    ]
    if any(c not in df.columns for c in needed):
        return

    x = pd.to_numeric(df["total_frequency"], errors="coerce")
    y = pd.to_numeric(df["efficiency_per_frequency"], errors="coerce")
    x_q25 = pd.to_numeric(df["total_frequency_q25"], errors="coerce")
    x_q75 = pd.to_numeric(df["total_frequency_q75"], errors="coerce")
    y_q25 = pd.to_numeric(df["efficiency_per_frequency_q25"], errors="coerce")
    y_q75 = pd.to_numeric(df["efficiency_per_frequency_q75"], errors="coerce")

    x_low = (x - x_q25).clip(lower=0).fillna(0).to_numpy()
    x_high = (x_q75 - x).clip(lower=0).fillna(0).to_numpy()
    y_low = (y - y_q25).clip(lower=0).fillna(0).to_numpy()
    y_high = (y_q75 - y).clip(lower=0).fillna(0).to_numpy()

    for idx, row in df.reset_index(drop=True).iterrows():
        key = str(row.get(hue_col, ""))
        line_color = palette_map.get(key, "#666666")
        ax.errorbar(
            float(x.iloc[idx]),
            float(y.iloc[idx]),
            xerr=np.array([[x_low[idx]], [x_high[idx]]]),
            yerr=np.array([[y_low[idx]], [y_high[idx]]]),
            fmt="none",
            ecolor=line_color,
            alpha=0.75,
            elinewidth=1.0,
            capsize=2.0,
            zorder=1,
        )


def plot_freq_saving_scatter(metrics: pd.DataFrame, item_col: str, out_file: Path) -> None:
    df = metrics.dropna(subset=["total_frequency", "efficiency_per_frequency"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    palette_map = _build_palette_map(df, item_col, "tab20")
    _add_iqr_errorbars(ax, df, item_col, palette_map)
    sns.scatterplot(
        data=df,
        x="total_frequency",
        y="efficiency_per_frequency",
        hue=item_col,
        palette=palette_map,
        s=60,
        alpha=0.9,
        ax=ax,
        legend="brief",
    )
    ax.set_title(f"{item_col} Frequency vs Efficiency (City-Balanced, IQR shown)")
    ax.set_xlabel("City-Balanced Median Frequency (%)")
    ax.set_ylabel("City-Balanced Efficiency per Frequency")

    # Keep legend outside to reduce overlap in dense scatter plots.
    if item_col == "object_field":
        _legend_bottom(ax, title=item_col, max_cols=2, fontsize=6, wrap_width=12, top_y=-0.1)
    else:
        _legend_bottom(ax, title=item_col, max_cols=4, fontsize=8, wrap_width=16, top_y=-0.1)

    _save_figure(fig, out_file)


def plot_quadrant(metrics: pd.DataFrame, item_col: str, out_file: Path) -> None:
    df = metrics.dropna(subset=["total_frequency", "efficiency_per_frequency"]).copy()
    if df.empty:
        return

    x_mid = df["total_frequency"].median()
    y_mid = df["efficiency_per_frequency"].median()

    def quadrant(row: pd.Series) -> str:
        high_x = row["total_frequency"] >= x_mid
        high_y = row["efficiency_per_frequency"] >= y_mid
        if high_x and high_y:
            return "高频-高效"
        if high_x and (not high_y):
            return "高频-低效"
        if (not high_x) and high_y:
            return "低频-高效"
        return "低频-低效"

    df["quadrant"] = df.apply(quadrant, axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    plot_df = df.copy()
    marker_size = 120
    marker_alpha = 0.95

    # Field quadrant is typically much denser than object quadrant.
    # Use light jitter + stronger transparency + white edges to reduce visual overlap.
    if item_col == "object_field":
        n_points = len(plot_df)
        marker_size = max(40, min(95, int(7000 / max(n_points, 1))))
        marker_alpha = 0.58

        x_std = float(plot_df["total_frequency"].std(skipna=True) or 0.0)
        y_std = float(plot_df["efficiency_per_frequency"].std(skipna=True) or 0.0)
        x_jitter = 0.01 * x_std
        y_jitter = 0.01 * y_std

        if x_jitter > 0:
            plot_df["total_frequency_plot"] = plot_df["total_frequency"] + np.random.normal(0, x_jitter, size=n_points)
        else:
            plot_df["total_frequency_plot"] = plot_df["total_frequency"]

        if y_jitter > 0:
            plot_df["efficiency_per_frequency_plot"] = plot_df["efficiency_per_frequency"] + np.random.normal(0, y_jitter, size=n_points)
        else:
            plot_df["efficiency_per_frequency_plot"] = plot_df["efficiency_per_frequency"]
    else:
        plot_df["total_frequency_plot"] = plot_df["total_frequency"]
        plot_df["efficiency_per_frequency_plot"] = plot_df["efficiency_per_frequency"]

    palette = "tab20"
    if item_col == "object_field":
        uniq_items = [str(x) for x in pd.unique(plot_df[item_col].dropna())]
        palette = _object_family_field_palette(uniq_items)

    palette_map = _build_palette_map(plot_df, item_col, palette)
    _add_iqr_errorbars(ax, df, item_col, palette_map)

    sns.scatterplot(
        data=plot_df,
        x="total_frequency_plot",
        y="efficiency_per_frequency_plot",
        hue=item_col,
        palette=palette_map,
        s=marker_size,
        linewidth=0.35,
        edgecolor="white",
        ax=ax,
        alpha=marker_alpha,
    )
    ax.axvline(x_mid, color="red", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{item_col} Quadrant: City-Balanced Frequency vs Efficiency (IQR shown)")
    ax.set_xlabel("City-Balanced Median Frequency (%)")
    ax.set_ylabel("City-Balanced Efficiency per Frequency")

    if item_col == "object_field":
        _legend_bottom(ax, title=item_col, max_cols=2, fontsize=6, wrap_width=12, top_y=-0.1)
    else:
        _legend_bottom(ax, title=item_col, max_cols=4, fontsize=8, wrap_width=16, top_y=-0.1)

    _save_figure(fig, out_file)


def plot_cumulative_contribution(metrics: pd.DataFrame, item_col: str, out_file: Path) -> None:
    df = metrics.copy()
    if df.empty:
        return

    df = df.sort_values("weighted_contribution", ascending=False)
    total = df["weighted_contribution"].sum()
    if pd.isna(total) or total == 0:
        return

    df["cum_ratio"] = df["weighted_contribution"].cumsum() / total
    df["rank_ratio"] = np.arange(1, len(df) + 1) / len(df)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["rank_ratio"], df["cum_ratio"], marker="o", linewidth=1.4)
    ax.set_title(f"{item_col} Cumulative Weighted Contribution")
    ax.set_xlabel("Top-K Item Ratio")
    ax.set_ylabel("Cumulative Contribution Ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.45)
    _save_figure(fig, out_file)


def plot_consistency_cv(freq_df: pd.DataFrame, item_col: str, out_file: Path, top_n: int) -> None:
    city_item = freq_df.groupby(["city", item_col], as_index=False)["frequency"].sum()
    pivot = city_item.pivot_table(index=item_col, columns="city", values="frequency", fill_value=0.0)
    if pivot.empty:
        return

    stats = pd.DataFrame({
        item_col: pivot.index,
        "mean_freq": pivot.mean(axis=1).values,
        "std_freq": pivot.std(axis=1).values,
    })
    stats["cv"] = stats["std_freq"] / stats["mean_freq"].replace(0, np.nan)
    stats = stats.dropna(subset=["cv"])
    if stats.empty:
        return

    stats = stats.sort_values("cv", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = _red_blue_palette(len(stats))
    sns.barplot(
        data=stats,
        y=item_col,
        x="cv",
        hue=item_col,
        legend=False,
        ax=ax,
        palette=bar_colors,
    )
    ax.set_title(f"{item_col} Cross-City Variability (CV, higher=more city-specific)")
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel(item_col)
    _save_figure(fig, out_file)


def plot_city_trend(freq_df: pd.DataFrame, item_col: str, city: str, out_file: Path, top_k: int) -> None:
    city_df = freq_df[freq_df["city"] == city].copy()
    if city_df.empty:
        return

    city_df = city_df.dropna(subset=["run_order"])
    if city_df.empty:
        return

    top_items = (
        city_df.groupby(item_col)["frequency"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )
    sub = city_df[city_df[item_col].isin(top_items)].copy()
    if sub.empty:
        return

    trend = (
        sub.groupby(["run_order", item_col], as_index=False)["frequency"]
        .sum()
        .pivot_table(index="run_order", columns=item_col, values="frequency", fill_value=0.0)
        .sort_index()
    )
    if trend.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in trend.columns:
        ax.plot(trend.index, trend[col], marker="o", label=str(col))

    ax.set_title(f"{city} | {item_col} Frequency Trend by Run")
    ax.set_xlabel("Run Order")
    ax.set_ylabel("Normalized Frequency (%)")
    if item_col == "object_field":
        _legend_bottom(ax, title=item_col, max_cols=2, fontsize=6, wrap_width=12, top_y=-0.1)
    else:
        _legend_bottom(ax, title=item_col, max_cols=4, fontsize=8, wrap_width=16, top_y=-0.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, out_file)


def generate_all_plots(data: Dict[str, pd.DataFrame], output_dir: Path, top_n: int) -> None:
    run_df = data["run_summary"].copy()
    obj_df = data["object_frequency"].copy()
    fld_df = data["field_frequency"].copy()

    _safe_mkdir(output_dir)
    overall_dir = output_dir / "overall"
    per_city_dir = output_dir / "per_city"
    _safe_mkdir(overall_dir)
    _safe_mkdir(per_city_dir)

    # Build metrics and merged run-level frames.
    obj_metrics, obj_merged = build_item_metrics(obj_df, run_df, "object_type")
    fld_metrics, fld_merged = build_item_metrics(fld_df, run_df, "object_field")

    # Export metrics tables for further analysis.
    obj_metrics.to_csv(overall_dir / "object_metrics.csv", index=False, encoding="utf-8-sig")
    fld_metrics.to_csv(overall_dir / "field_metrics.csv", index=False, encoding="utf-8-sig")
    obj_merged.to_csv(overall_dir / "object_run_level_merged.csv", index=False, encoding="utf-8-sig")
    fld_merged.to_csv(overall_dir / "field_run_level_merged.csv", index=False, encoding="utf-8-sig")

    # Overall plots.
    plot_city_item_heatmap(obj_df, "object_type", overall_dir / "object_city_heatmap.png", top_n=top_n)
    plot_city_item_heatmap(fld_df, "object_field", overall_dir / "field_city_heatmap.png", top_n=top_n)

    plot_freq_saving_scatter(obj_metrics, "object_type", overall_dir / "object_freq_vs_efficiency_scatter.png")
    plot_freq_saving_scatter(fld_metrics, "object_field", overall_dir / "field_freq_vs_efficiency_scatter.png")

    plot_quadrant(obj_metrics, "object_type", overall_dir / "object_quadrant.png")
    plot_quadrant(fld_metrics, "object_field", overall_dir / "field_quadrant.png")

    plot_cumulative_contribution(obj_metrics, "object_type", overall_dir / "object_cumulative_contribution.png")
    plot_cumulative_contribution(fld_metrics, "object_field", overall_dir / "field_cumulative_contribution.png")

    plot_consistency_cv(obj_df, "object_type", overall_dir / "object_city_variability_cv.png", top_n=top_n)
    plot_consistency_cv(fld_df, "object_field", overall_dir / "field_city_variability_cv.png", top_n=top_n)

    # Per-city plots.
    cities = sorted(set(run_df["city"].dropna().astype(str).tolist()))
    for city in cities:
        city_dir = per_city_dir / city
        _safe_mkdir(city_dir)

        plot_top_bar(obj_df, "object_type", city, city_dir / "object_top_bar.png", top_n=min(15, top_n))
        plot_top_bar(fld_df, "object_field", city, city_dir / "field_top_bar.png", top_n=min(20, top_n))

        plot_city_trend(obj_df, "object_type", city, city_dir / "object_trend_top5.png", top_k=5)
        plot_city_trend(fld_df, "object_field", city, city_dir / "field_trend_top8.png", top_k=8)

    print(f"[OK] All plots and metrics exported to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequency visualization for city optimization Excel reports.")
    parser.add_argument(
        "--root",
        type=str,
        default="各城市迭代结果",
        help="Root folder containing city subfolders with *_统计汇总.xlsx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="画图",
        help="Output directory for figures and CSV summaries",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top N items for heatmaps and key ranking figures",
    )
    return parser.parse_args()


def main() -> None:
    _set_plot_style()
    args = parse_args()

    root_dir = Path(args.root)
    output_dir = Path(args.output)

    if not root_dir.exists():
        raise FileNotFoundError(f"Input root does not exist: {root_dir}")

    data = load_all_data(root_dir)
    generate_all_plots(data, output_dir=output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
