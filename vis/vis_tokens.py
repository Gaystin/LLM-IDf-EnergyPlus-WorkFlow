from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


DEFAULT_EXCEL = Path("各城市迭代结果_50%节能率") / "各城市迭代结果.xlsx"
DEFAULT_OUTPUT_PNG = Path("city_total_tokens_relationship.png")
DEFAULT_OUTPUT_SVG = Path("city_total_tokens_relationship.svg")

CITY_ORDER = ["Harbin", "Beijing", "Shanghai", "Wuhan", "Chengdu", "Guangzhou", "Kunming"]


def _city_colors(cities: list[str]) -> dict[str, tuple]:
    n = max(len(cities), 2)
    start_rgb = np.array(mcolors.to_rgb("#4C78A8"))
    end_rgb = np.array(mcolors.to_rgb("#E97F89"))
    vals = np.linspace(0.0, 1.0, n)
    colors: dict[str, tuple] = {}
    for idx, city in enumerate(cities):
        rgb = start_rgb * (1.0 - vals[idx]) + end_rgb * vals[idx]
        colors[city] = (*rgb.tolist(), 1.0)
    return colors


def _city_color_map_for_tokens(cities: list[str]) -> dict[str, tuple]:
    existing = sorted(set(cities))
    ordered = [c for c in CITY_ORDER if c in existing]
    tail = [c for c in existing if c not in ordered]
    return _city_colors(ordered + tail)


def _to_numeric_tokens(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(",", "", regex=False)
    numeric = pd.to_numeric(text, errors="coerce").dropna()
    return numeric


def build_city_summary(excel_path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(excel_path)
    rows: list[dict[str, float | int | str]] = []

    for city in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=city)
        if "total_tokens" not in df.columns:
            continue

        tokens = _to_numeric_tokens(df["total_tokens"])
        if tokens.empty:
            continue

        rows.append(
            {
                "city": city,
                "run_count": int(tokens.shape[0]),
                "sum_total_tokens": float(tokens.sum()),
                "avg_total_tokens": float(tokens.mean()),
                "std_total_tokens": float(tokens.std(ddof=1)) if tokens.shape[0] > 1 else 0.0,
            }
        )

    if not rows:
        raise ValueError("No valid city sheets with 'total_tokens' were found.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("avg_total_tokens", ascending=False).reset_index(drop=True)
    return summary


def plot_city_relationship(summary: pd.DataFrame, output_png: Path, output_svg: Path, show_plot: bool) -> None:
    cities = summary["city"].tolist()
    avg_tokens = summary["avg_total_tokens"].to_numpy()
    city_color_map = _city_color_map_for_tokens(cities)
    bar_colors = [city_color_map.get(city, "#4C78A8") for city in cities]

    x = np.arange(len(cities))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_left = plt.subplots(figsize=(12.5, 7))

    line_color = "#E97F89"

    ax_left.bar(
        x,
        avg_tokens,
        width=0.6,
        color=bar_colors,
        alpha=0.4,
        zorder=1,
    )

    ax_left.plot(
        x,
        avg_tokens,
        color=line_color,
        marker="o",
        linewidth=2.2,
        markersize=6,
        zorder=3,
    )

    ax_left.set_xlabel("City", fontsize=16)
    ax_left.set_ylabel("Average total_tokens per run", fontsize=16, color="black")
    ax_left.set_ylim(20000, 120000)
    ax_left.set_yticks(np.arange(20000, 120001, 20000))

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(cities, rotation=0, ha="center", fontsize=13, color="black")
    ax_left.tick_params(axis="x", labelsize=13, colors="black")
    ax_left.tick_params(axis="y", labelsize=13, colors="black")

    ax_left.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_left.grid(axis="x", visible=False)

    # Label size adapts to bar width in screen space for consistent aesthetics.
    bar_data_width = 0.6
    x0_px = ax_left.transData.transform((0, 0))[0]
    x1_px = ax_left.transData.transform((bar_data_width, 0))[0]
    bar_width_px = abs(x1_px - x0_px)
    label_fontsize = float(np.clip(bar_width_px * 0.22, 9, 13))

    for xi, value in zip(x, avg_tokens):
        ax_left.text(
            xi,
            value + 1800,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="black",
            zorder=4,
        )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    fig.savefig(output_svg)

    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the relationship between cities and total_tokens from multi-run sheets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_EXCEL, help="Input Excel file path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PNG, help="Output PNG path.")
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG, help="Output SVG path.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("city_total_tokens_summary.csv"),
        help="Optional CSV output for aggregated city statistics.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure window.")
    args = parser.parse_args()

    summary = build_city_summary(args.input)
    summary.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")

    print("City summary (sorted by average total_tokens):")
    print(summary[["city", "run_count", "sum_total_tokens", "avg_total_tokens", "std_total_tokens"]].to_string(index=False))

    plot_city_relationship(
        summary=summary,
        output_png=args.output,
        output_svg=args.output_svg,
        show_plot=args.show,
    )

    print(f"PNG saved to: {args.output.resolve()}")
    print(f"SVG saved to: {args.output_svg.resolve()}")
    print(f"Summary CSV saved to: {args.summary_csv.resolve()}")


if __name__ == "__main__":
    main()
