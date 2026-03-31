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

# Climate classification mapping
CLIMATE_ZONES = {
    "哈尔滨": "严寒地区",
    "Harbin": "严寒地区",
    "北京": "寒冷地区",
    "Beijing": "寒冷地区",
    "武汉": "夏热冬冷",
    "Wuhan": "夏热冬冷",
    "上海": "夏热冬冷",
    "Shanghai": "夏热冬冷",
    "成都": "夏热冬冷",
    "Chengdu": "夏热冬冷",
    "广州": "夏热冬暖",
    "Guangzhou": "夏热冬暖",
    "昆明": "温和地区",
    "Kunming": "温和地区",
}

CLIMATE_ZONE_ORDER = ["严寒地区", "寒冷地区", "夏热冬冷", "夏热冬暖", "温和地区"]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, out_file: Path) -> None:
    # Use bbox_inches instead of tight_layout to avoid layout warnings on dense labels.
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


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


def _red_blue_palette(n: int) -> List:
    # For bar charts: top bars red -> bottom bars blue (soft, not too dark).
    base = _blue_red_cmap()
    # Descending on this cmap gives red->blue for top-to-bottom bars.
    trim = base(np.linspace(0.95, 0.05, max(n, 2)))
    if n <= 1:
        return [trim[0]]
    return [trim[i] for i in range(n)]


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


def get_climate_zone(city: str) -> str:
    """Get climate zone for a city."""
    for city_name, zone in CLIMATE_ZONES.items():
        if city.strip() == city_name.strip():
            return zone
    return "未分类"


def _display_item_label(item_col: str) -> str:
    if item_col == "object_type":
        return "Object"
    if item_col == "object_field":
        return "Object Field"
    return str(item_col)


def plot_iterations_distribution(run_df: pd.DataFrame, out_file: Path) -> None:
    """Plot city-level average best_iteration across runs."""
    if "best_iteration" not in run_df.columns:
        return

    plot_df = run_df[["city", "run_tag", "best_iteration"]].drop_duplicates().copy()
    plot_df["best_iteration"] = pd.to_numeric(plot_df["best_iteration"], errors="coerce")
    plot_df = plot_df.dropna(subset=["best_iteration"])
    if plot_df.empty:
        return

    city_mean = (
        plot_df.groupby("city", as_index=False)["best_iteration"]
        .mean()
        .sort_values("city")
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cities = sorted(city_mean["city"].unique())
    bar_colors = _red_blue_palette(len(cities))
    
    sns.barplot(
        data=city_mean,
        x="city",
        y="best_iteration",
        hue="city",
        legend=False,
        ax=ax,
        palette=bar_colors,
    )
    ax.set_title("Early Stopping: Mean Best Iteration by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Mean Best Iteration (across runs)")
    ax.set_ylim(0, 8)
    plt.xticks(rotation=0, ha="center", fontsize=11)
    
    # Add value labels on bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=10)
    
    _save_figure(fig, out_file)


def plot_climate_zone_heatmap(fld_df: pd.DataFrame, out_file: Path, top_n: int) -> None:
    """Plot climate zone vs field frequency heatmap."""
    # Add climate zone to frequency data
    fld_copy = fld_df[["city", "object_field", "frequency"]].copy()
    fld_copy["climate_zone"] = fld_copy["city"].apply(get_climate_zone)
    
    # Get top fields overall
    top_fields = fld_copy.groupby("object_field")["frequency"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    fld_copy = fld_copy[fld_copy["object_field"].isin(top_fields)]
    
    if fld_copy.empty:
        return
    
    # Pivot: rows = fields, columns = climate zones
    pivot = fld_copy.pivot_table(
        index="object_field",
        columns="climate_zone",
        values="frequency",
        aggfunc="mean",
        fill_value=0.0
    )
    
    # Reorder columns by climate zone order
    pivot = pivot[[col for col in CLIMATE_ZONE_ORDER if col in pivot.columns]]
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5), max(8, len(pivot.index) * 0.8)))
    heat_cmap = _blue_red_cmap()
    sns.heatmap(pivot, cmap=heat_cmap, ax=ax, cbar_kws={"label": "Normalized Frequency (%)"})
    
    ax.set_title(f"Climate Zone vs Field Frequency Heatmap (%) (Top {top_n} Fields)")
    ax.set_xlabel("Climate Zone", fontsize=16)
    ax.set_ylabel("Object Field", fontsize=16)
    
    wrapped_y = [_wrap_label(t.get_text(), width=18) for t in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    
    _save_figure(fig, out_file)


def plot_climate_zone_object_heatmap(obj_df: pd.DataFrame, out_file: Path, top_n: int) -> None:
    """Plot climate zone vs object frequency heatmap."""
    # Add climate zone to frequency data
    obj_copy = obj_df[["city", "object_type", "frequency"]].copy()
    obj_copy["climate_zone"] = obj_copy["city"].apply(get_climate_zone)
    
    # Get top objects overall
    top_objects = obj_copy.groupby("object_type")["frequency"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    obj_copy = obj_copy[obj_copy["object_type"].isin(top_objects)]
    
    if obj_copy.empty:
        return
    
    # Pivot: rows = objects, columns = climate zones
    pivot = obj_copy.pivot_table(
        index="object_type",
        columns="climate_zone",
        values="frequency",
        aggfunc="mean",
        fill_value=0.0
    )
    
    # Reorder columns by climate zone order
    pivot = pivot[[col for col in CLIMATE_ZONE_ORDER if col in pivot.columns]]
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5), max(8, len(pivot.index) * 0.8)))
    heat_cmap = _blue_red_cmap()
    sns.heatmap(pivot, cmap=heat_cmap, ax=ax, cbar_kws={"label": "Normalized Frequency (%)"})
    
    ax.set_title(f"Climate Zone vs Object Frequency Heatmap (%) (Top {top_n} Objects)")
    ax.set_xlabel("Climate Zone", fontsize=16)
    ax.set_ylabel("Object", fontsize=16)
    
    wrapped_y = [_wrap_label(t.get_text(), width=18) for t in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    
    _save_figure(fig, out_file)


def plot_city_field_heatmap(fld_df: pd.DataFrame, out_file: Path, top_n: int) -> None:
    """Plot city vs field frequency heatmap."""
    plot_df = fld_df[["city", "object_field", "frequency"]].copy()
    top_fields = (
        plot_df.groupby("object_field")["frequency"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    plot_df = plot_df[plot_df["object_field"].isin(top_fields)]
    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index="object_field",
        columns="city",
        values="frequency",
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.3), max(8, len(pivot.index) * 0.8)))
    sns.heatmap(pivot, cmap=_blue_red_cmap(), ax=ax, cbar_kws={"label": "Normalized Frequency (%)"})
    ax.set_title(f"City vs Field Frequency Heatmap (%) (Top {top_n} Fields)")
    ax.set_xlabel("City", fontsize=16)
    ax.set_ylabel("Object Field", fontsize=16)
    wrapped_y = [_wrap_label(t.get_text(), width=18) for t in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    _save_figure(fig, out_file)


def plot_city_object_heatmap(obj_df: pd.DataFrame, out_file: Path, top_n: int) -> None:
    """Plot city vs object frequency heatmap."""
    plot_df = obj_df[["city", "object_type", "frequency"]].copy()
    top_objects = (
        plot_df.groupby("object_type")["frequency"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    plot_df = plot_df[plot_df["object_type"].isin(top_objects)]
    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index="object_type",
        columns="city",
        values="frequency",
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.3), max(8, len(pivot.index) * 0.8)))
    sns.heatmap(pivot, cmap=_blue_red_cmap(), ax=ax, cbar_kws={"label": "Normalized Frequency (%)"})
    ax.set_title(f"City vs Object Frequency Heatmap (%) (Top {top_n} Objects)")
    ax.set_xlabel("City", fontsize=16)
    ax.set_ylabel("Object", fontsize=16)
    wrapped_y = [_wrap_label(t.get_text(), width=18) for t in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
    _save_figure(fig, out_file)


def plot_city_top_bar(freq_df: pd.DataFrame, item_col: str, city: str, out_file: Path, top_n: int) -> None:
    """Plot top items by mean normalized frequency for one city."""
    city_df = freq_df[freq_df["city"] == city].copy()
    if city_df.empty or item_col not in city_df.columns:
        return

    agg = (
        city_df.groupby(item_col, as_index=False)["frequency"]
        .mean()
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
    display_label = _display_item_label(item_col)
    ax.set_title(f"{city} | Top {top_n} {display_label} by Mean Frequency")
    ax.set_xlabel("Mean Normalized Frequency (%)")
    ax.set_ylabel(display_label)
    _save_figure(fig, out_file)


def export_convergence_metrics(run_df: pd.DataFrame, out_dir: Path) -> None:
    """Export convergence metrics to CSV."""
    metrics = run_df[["city", "iterations_executed", "best_saving_pct_total_site"]].drop_duplicates().copy()
    metrics = metrics.dropna(subset=["iterations_executed", "best_saving_pct_total_site"])
    
    if not metrics.empty:
        metrics.to_csv(out_dir / "convergence_metrics.csv", index=False, encoding="utf-8-sig")
        
        # City-level summary
        city_summary = metrics.groupby("city").agg({
            "iterations_executed": ["min", "max", "mean"],
            "best_saving_pct_total_site": ["min", "max", "mean"],
        }).reset_index()
        city_summary.columns = ["city", "iters_min", "iters_max", "iters_mean", "saving_min", "saving_max", "saving_mean"]
        city_summary.to_csv(out_dir / "city_convergence_summary.csv", index=False, encoding="utf-8-sig")


def generate_supplementary_plots(data: Dict[str, pd.DataFrame], output_dir: Path, top_n: int) -> None:
    """Generate all supplementary analysis plots."""
    run_df = data["run_summary"].copy()
    obj_df = data["object_frequency"].copy()
    fld_df = data["field_frequency"].copy()

    _safe_mkdir(output_dir)
    convergence_dir = output_dir / "convergence"
    climate_dir = output_dir / "climate_analysis"
    per_city_dir = output_dir / "city_profiles"
    _safe_mkdir(convergence_dir)
    _safe_mkdir(climate_dir)
    _safe_mkdir(per_city_dir)

    # Convergence analysis plots
    print("[PLOT] Generating iterations distribution...")
    plot_iterations_distribution(run_df, convergence_dir / "iterations_distribution.png")
    
    # Climate zone analysis plots
    print("[PLOT] Generating climate zone vs field heatmap...")
    plot_climate_zone_heatmap(fld_df, climate_dir / "climate_zone_field_heatmap.png", top_n=top_n)
    print("[PLOT] Generating city vs field heatmap...")
    plot_city_field_heatmap(fld_df, climate_dir / "city_field_heatmap.png", top_n=top_n)
    
    print("[PLOT] Generating climate zone vs object heatmap...")
    plot_climate_zone_object_heatmap(obj_df, climate_dir / "climate_zone_object_heatmap.png", top_n=min(top_n, 15))
    print("[PLOT] Generating city vs object heatmap...")
    plot_city_object_heatmap(obj_df, climate_dir / "city_object_heatmap.png", top_n=min(top_n, 15))
    
    # Export metrics
    print("[CSV] Exporting convergence metrics...")
    export_convergence_metrics(run_df, convergence_dir)

    # Per-city supplementary plots
    cities = sorted(set(run_df["city"].dropna().astype(str).tolist()))
    for city in cities:
        city_dir = per_city_dir / city
        _safe_mkdir(city_dir)

        print(f"[PLOT] Generating per-city plots for: {city}")
        # Keep only per-city top object/field average-frequency bars.
        plot_city_top_bar(obj_df, "object_type", city, city_dir / "object_top_bar.png", top_n=min(15, top_n))
        plot_city_top_bar(fld_df, "object_field", city, city_dir / "field_top_bar.png", top_n=min(20, top_n))
    
    print(f"[OK] All supplementary plots and metrics exported to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convergence and climate analysis for city optimization Excel reports.")
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
        help="Top N items for heatmaps",
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
    generate_supplementary_plots(data, output_dir=output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
