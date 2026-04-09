from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CARBON_FACTOR = 0.5306
DEFAULT_ROOT = Path("各城市迭代结果")
DEFAULT_TREND_PNG = Path("city_carbon_iteration_trend.png")
DEFAULT_TREND_SVG = Path("city_carbon_iteration_trend.svg")
DEFAULT_FINAL_PNG = Path("city_carbon_best_vs_baseline.png")
DEFAULT_FINAL_SVG = Path("city_carbon_best_vs_baseline.svg")
DEFAULT_AVG_PNG = Path("city_carbon_avg_vs_baseline.png")
DEFAULT_AVG_SVG = Path("city_carbon_avg_vs_baseline.svg")
DEFAULT_ITER_CSV = Path("climate_carbon_iteration_summary.csv")
DEFAULT_FINAL_CSV = Path("climate_carbon_final_summary.csv")

CITY_ORDER = ["Harbin", "Beijing", "Shanghai", "Wuhan", "Chengdu", "Guangzhou", "Kunming"]

CLIMATE_ZONES = {
    "哈尔滨": "Severe Cold",
    "Harbin": "Severe Cold",
    "北京": "Cold",
    "Beijing": "Cold",
    "武汉": "Hot Summer / Cold Winter",
    "Wuhan": "Hot Summer / Cold Winter",
    "上海": "Hot Summer / Cold Winter",
    "Shanghai": "Hot Summer / Cold Winter",
    "成都": "Hot Summer / Cold Winter",
    "Chengdu": "Hot Summer / Cold Winter",
    "广州": "Hot Summer / Warm Winter",
    "Guangzhou": "Hot Summer / Warm Winter",
    "昆明": "Mild",
    "Kunming": "Mild",
}
CLIMATE_ZONE_ORDER = ["Severe Cold", "Cold", "Hot Summer / Cold Winter", "Hot Summer / Warm Winter", "Mild"]
CLIMATE_ZONE_COLORS = {
    "Severe Cold": "#4C78A8",
    "Cold": "#6FA8DC",
    "Hot Summer / Cold Winter": "#E97F89",
    "Hot Summer / Warm Winter": "#D96C75",
    "Mild": "#8FBF9F",
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, out_file: Path) -> None:
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _to_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _get_city_from_summary(run_summary: pd.DataFrame, workbook_path: Path) -> str:
    if "city" in run_summary.columns:
        cities = run_summary["city"].dropna().astype(str).str.strip()
        cities = cities[cities != ""]
        if not cities.empty:
            return cities.iloc[0]
    return workbook_path.parent.name


def _get_climate_zone(city: str) -> str:
    text = str(city or "").strip()
    return CLIMATE_ZONES.get(text, "Other")


def _normalize_city_name(city: str) -> str:
    text = str(city or "").strip()
    mapping = {
        "哈尔滨": "Harbin",
        "北京": "Beijing",
        "上海": "Shanghai",
        "武汉": "Wuhan",
        "成都": "Chengdu",
        "广州": "Guangzhou",
        "昆明": "Kunming",
    }
    return mapping.get(text, text)


def _ordered_cities(df: pd.DataFrame) -> List[str]:
    existing = sorted(df["city"].dropna().astype(str).unique().tolist())
    ordered = [c for c in CITY_ORDER if c in existing]
    tail = [c for c in existing if c not in ordered]
    return ordered + tail


def _city_colors(cities: List[str]) -> dict[str, tuple]:
    n = max(len(cities), 2)
    start_rgb = np.array(mcolors.to_rgb("#4C78A8"))
    end_rgb = np.array(mcolors.to_rgb("#E97F89"))
    vals = np.linspace(0.0, 1.0, n)
    colors = {}
    for idx, city in enumerate(cities):
        rgb = start_rgb * (1.0 - vals[idx]) + end_rgb * vals[idx]
        colors[city] = (*rgb.tolist(), 1.0)
    return colors


def discover_city_workbooks(root_dir: Path) -> List[Path]:
    files = sorted(root_dir.rglob("*_统计汇总.xlsx"))
    return [p for p in files if not p.name.startswith("~$")]


def load_city_round_metrics(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    required = {"run_summary", "run_round_metrics"}
    missing = required - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Workbook missing required sheets: {sorted(missing)} | file={path}")

    run_summary = pd.read_excel(path, sheet_name="run_summary")
    round_metrics = pd.read_excel(path, sheet_name="run_round_metrics")

    city = _get_city_from_summary(run_summary, path)
    city = _normalize_city_name(city)
    round_metrics = round_metrics.copy()
    round_metrics["city"] = city
    round_metrics["source_file"] = str(path)
    round_metrics["climate_zone"] = _get_climate_zone(city)

    if "iteration" in round_metrics.columns:
        round_metrics["iteration"] = pd.to_numeric(round_metrics["iteration"], errors="coerce")

    if "eui_kgco2_per_m2" in round_metrics.columns:
        round_metrics["carbon_kgco2_per_m2"] = pd.to_numeric(
            round_metrics["eui_kgco2_per_m2"], errors="coerce"
        )
    else:
        eui = pd.to_numeric(round_metrics.get("eui_kwh_per_m2"), errors="coerce")
        round_metrics["carbon_kgco2_per_m2"] = eui * CARBON_FACTOR

    round_metrics["carbon_kgco2_per_m2"] = pd.to_numeric(round_metrics["carbon_kgco2_per_m2"], errors="coerce")

    for col in ("run_tag", "workflow_id", "run_status", "error_message"):
        if col not in round_metrics.columns:
            round_metrics[col] = ""

    round_metrics["run_tag"] = round_metrics["run_tag"].astype(str)
    round_metrics["workflow_id"] = round_metrics["workflow_id"].astype(str)
    round_metrics["run_key"] = round_metrics["city"].astype(str) + "|" + round_metrics["run_tag"] + "|" + round_metrics["workflow_id"]

    round_metrics = round_metrics.dropna(subset=["iteration", "carbon_kgco2_per_m2"])
    round_metrics["iteration"] = round_metrics["iteration"].astype(int)
    return round_metrics


def load_all_city_metrics(root_dir: Path) -> pd.DataFrame:
    workbooks = discover_city_workbooks(root_dir)
    if not workbooks:
        raise FileNotFoundError(f"No city summary workbooks found under: {root_dir}")

    frames: List[pd.DataFrame] = []
    for path in workbooks:
        try:
            df = load_city_round_metrics(path)
            frames.append(df)
            print(f"[OK] Loaded: {path}")
        except Exception as exc:
            print(f"[WARN] Skip file due to error: {path} | {exc}")

    if not frames:
        raise RuntimeError("No valid city workbooks loaded.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["carbon_kgco2_per_m2"] = pd.to_numeric(all_df["carbon_kgco2_per_m2"], errors="coerce")
    all_df = all_df.dropna(subset=["carbon_kgco2_per_m2", "iteration"])
    return all_df


def _set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def build_city_iteration_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["city", "iteration"], as_index=False)["carbon_kgco2_per_m2"]
        .agg(mean="mean", std="std", count="count")
        .sort_values(["city", "iteration"])
        .reset_index(drop=True)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def build_city_final_summary(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values(["run_key", "iteration"])
        .groupby("run_key", as_index=False)
        .tail(1)
        .copy()
    )
    return final_rows


def build_city_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = df[df["iteration"] == 0].copy()
    summary = (
        baseline_df.groupby("city", as_index=False)["carbon_kgco2_per_m2"]
        .agg(mean="mean", std="std", count="count")
        .sort_values("city")
        .reset_index(drop=True)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def build_city_best_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    return (
        final_df.groupby("city", as_index=False)["carbon_kgco2_per_m2"]
        .min()
        .rename(columns={"carbon_kgco2_per_m2": "target_carbon_kgco2_per_m2"})
        .sort_values("city")
        .reset_index(drop=True)
    )


def build_city_avg_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    return (
        final_df.groupby("city", as_index=False)["carbon_kgco2_per_m2"]
        .mean()
        .rename(columns={"carbon_kgco2_per_m2": "target_carbon_kgco2_per_m2"})
        .sort_values("city")
        .reset_index(drop=True)
    )


def plot_city_iteration_trend(iter_summary: pd.DataFrame, output_png: Path, output_svg: Path, show_plot: bool) -> None:
    cities = _ordered_cities(iter_summary)
    if not cities:
        raise ValueError("No cities found in iteration summary.")

    color_map = _city_colors(cities)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    max_y = float((iter_summary["mean"] + iter_summary["std"]).max()) if not iter_summary.empty else 1.0
    max_y *= 1.12

    for city in cities:
        city_df = iter_summary[iter_summary["city"] == city].sort_values("iteration")
        if city_df.empty:
            continue
        x = city_df["iteration"].to_numpy(dtype=float)
        mean = city_df["mean"].to_numpy(dtype=float)
        std = city_df["std"].to_numpy(dtype=float)
        color = color_map[city]
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=city,
            zorder=3,
        )
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0, zorder=2)

    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Carbon intensity (kgCO2/m²)", fontsize=16)
    ax.tick_params(axis="x", labelsize=13, colors="black")
    ax.tick_params(axis="y", labelsize=13, colors="black")
    ax.set_ylim(0, max(max_y, 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", visible=False)
    legend = ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=11,
    )
    legend.set_zorder(30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _safe_mkdir(output_png.parent)
    _safe_mkdir(output_svg.parent)
    fig.savefig(output_png, dpi=320)
    fig.savefig(output_svg)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_city_baseline_vs_target(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    show_plot: bool,
    target_label: str,
) -> None:
    cities = _ordered_cities(target_df if not target_df.empty else baseline_df)
    if not cities:
        raise ValueError("No cities found for baseline comparison plot.")

    color_map = _city_colors(cities)
    baseline_map = baseline_df.set_index("city")["mean"].to_dict() if not baseline_df.empty else {}
    target_map = target_df.set_index("city")["target_carbon_kgco2_per_m2"].to_dict() if not target_df.empty else {}

    baseline_values = [float(baseline_map.get(city, np.nan)) for city in cities]
    target_values = [float(target_map.get(city, np.nan)) for city in cities]
    saving_ratio_values = []
    for base_val, target_val in zip(baseline_values, target_values):
        if np.isnan(base_val) or np.isnan(target_val) or base_val == 0:
            saving_ratio_values.append(np.nan)
        else:
            saving_ratio_values.append((base_val - target_val) / base_val)

    x = np.arange(len(cities))
    bar_width = 0.34
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax_right = ax.twinx()
    # Keep grid below all data artists and disable right-axis grid to avoid overlay artifacts.
    ax.set_axisbelow(True)
    ax_right.grid(False)

    for idx, city in enumerate(cities):
        color = color_map[city]
        base_val = baseline_values[idx]
        target_val = target_values[idx]
        if not np.isnan(base_val):
            ax.bar(
                x[idx] - bar_width / 2,
                base_val,
                width=bar_width,
                color=color,
                edgecolor="#333333",
                linewidth=1.0,
                alpha=0.55,
                hatch="///",
                zorder=3,
            )
        if not np.isnan(target_val):
            ax.bar(
                x[idx] + bar_width / 2,
                target_val,
                width=bar_width,
                color=color,
                edgecolor="#333333",
                linewidth=1.0,
                alpha=0.92,
                zorder=4,
            )

    line_x = x + bar_width / 2
    ax_right.plot(
        line_x,
        saving_ratio_values,
        color="#E3606D",
        marker="o",
        linewidth=2.2,
        markersize=6,
        zorder=6,
    )

    ax.set_xlabel("City", fontsize=16)
    ax.set_ylabel("Carbon intensity (kgCO2/m²)", fontsize=16)
    ax_right.set_ylabel("Saving ratio", fontsize=16, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=0, ha="center", fontsize=13, color="black")
    ax.tick_params(axis="x", labelsize=14, colors="black")
    ax.tick_params(axis="y", labelsize=14, colors="black")
    ax_right.tick_params(axis="y", labelsize=14, colors="black")
    ax.set_ylim(0, 200)
    ax.set_yticks(np.arange(0, 201, 20))
    ax_right.set_ylim(0, 1)
    ax_right.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#BBBBBB", edgecolor="#333333", hatch="///", label="Baseline simulation"),
        Patch(facecolor="#BBBBBB", edgecolor="#333333", label=target_label),
    ]
    line_handle = plt.Line2D([0], [0], color="#E3606D", marker="o", linewidth=2.2, label="Saving ratio")
    legend = fig.legend(
        handles=legend_handles + [line_handle],
        loc="upper right",
        bbox_to_anchor=(0.94, 0.985),
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=11,
    )
    legend.set_zorder(30)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    _safe_mkdir(output_png.parent)
    _safe_mkdir(output_svg.parent)
    fig.savefig(output_png, dpi=320)
    fig.savefig(output_svg)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze carbon-emission optimization differences across cities.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Root folder containing city summary workbooks.")
    parser.add_argument("--trend-png", type=str, default=str(DEFAULT_TREND_PNG), help="Output PNG path for city iteration trend plot.")
    parser.add_argument("--trend-svg", type=str, default=str(DEFAULT_TREND_SVG), help="Output SVG path for city iteration trend plot.")
    parser.add_argument("--final-png", type=str, default=str(DEFAULT_FINAL_PNG), help="Output PNG path for final distribution plot.")
    parser.add_argument("--final-svg", type=str, default=str(DEFAULT_FINAL_SVG), help="Output SVG path for final distribution plot.")
    parser.add_argument("--avg-png", type=str, default=str(DEFAULT_AVG_PNG), help="Output PNG path for average comparison plot.")
    parser.add_argument("--avg-svg", type=str, default=str(DEFAULT_AVG_SVG), help="Output SVG path for average comparison plot.")
    parser.add_argument("--iter-summary-csv", type=str, default=str(DEFAULT_ITER_CSV), help="CSV output for iteration summary.")
    parser.add_argument("--final-summary-csv", type=str, default=str(DEFAULT_FINAL_CSV), help="CSV output for final-iteration summary.")
    parser.add_argument("--show", action="store_true", help="Show the figure window.")
    return parser.parse_args()


def main() -> None:
    _set_plot_style()
    args = parse_args()

    root_dir = Path(args.root)
    trend_png = Path(args.trend_png)
    trend_svg = Path(args.trend_svg)
    final_png = Path(args.final_png)
    final_svg = Path(args.final_svg)
    avg_png = Path(args.avg_png)
    avg_svg = Path(args.avg_svg)
    iter_csv = Path(args.iter_summary_csv)
    final_csv = Path(args.final_summary_csv)

    all_df = load_all_city_metrics(root_dir)
    iter_summary = build_city_iteration_summary(all_df)
    final_df = build_city_final_summary(all_df)
    baseline_df = build_city_baseline_summary(all_df)
    best_df = build_city_best_summary(final_df)
    avg_df = build_city_avg_summary(final_df)

    iter_summary.to_csv(iter_csv, index=False, encoding="utf-8-sig")
    final_df[["city", "climate_zone", "run_tag", "workflow_id", "iteration", "carbon_kgco2_per_m2"]].to_csv(
        final_csv, index=False, encoding="utf-8-sig"
    )

    print("Iteration summary (first rows):")
    print(iter_summary.head(10).to_string(index=False))
    print("Final-iteration summary (first rows):")
    print(final_df[["city", "climate_zone", "run_tag", "workflow_id", "iteration", "carbon_kgco2_per_m2"]].head(10).to_string(index=False))

    plot_city_iteration_trend(iter_summary, trend_png, trend_svg, args.show)
    plot_city_baseline_vs_target(baseline_df, best_df, final_png, final_svg, args.show, "Best final run")
    plot_city_baseline_vs_target(baseline_df, avg_df, avg_png, avg_svg, args.show, "Average final run")

    print(f"Trend PNG saved to: {trend_png.resolve()}")
    print(f"Trend SVG saved to: {trend_svg.resolve()}")
    print(f"Final PNG saved to: {final_png.resolve()}")
    print(f"Final SVG saved to: {final_svg.resolve()}")
    print(f"Average PNG saved to: {avg_png.resolve()}")
    print(f"Average SVG saved to: {avg_svg.resolve()}")
    print(f"Iteration summary CSV saved to: {iter_csv.resolve()}")
    print(f"Final summary CSV saved to: {final_csv.resolve()}")


if __name__ == "__main__":
    main()
