from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXCEL = "最佳节能幅度.xlsx"
DEFAULT_SHEET = "Sheet1"


def parse_numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    numeric = pd.to_numeric(text, errors="coerce")
    if numeric.isna().any():
        bad_values = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"Found non-numeric values in series: {bad_values}")
    return numeric


def build_plot(
    input_excel: Path,
    sheet_name: str,
    output_png: Path,
    output_svg: Path,
    show_plot: bool,
) -> None:
    df = pd.read_excel(input_excel, sheet_name=sheet_name)

    required_cols = ["总能耗", "迭代次数"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    n_cases = len(df)
    if n_cases == 0:
        raise ValueError("The Excel sheet is empty.")

    if "序号" in df.columns:
        case_ids = parse_numeric_series(df["序号"]).astype(int)
    else:
        case_ids = pd.Series(range(1, n_cases + 1), name="Case Index")

    total_saving = parse_numeric_series(df["总能耗"])
    iterations = parse_numeric_series(df["迭代次数"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_right = ax_left.twinx()
    # Keep left axis artists (line, legend) above right axis bars in twin-axis plots.
    ax_left.set_zorder(2)
    ax_right.set_zorder(1)
    ax_left.patch.set_visible(False)

    bars = ax_right.bar(
        case_ids,
        iterations,
        width=0.6,
        color="#4C78A8",
        alpha=0.4,
        label="Iterations",
        zorder=1,
    )

    line = ax_left.plot(
        case_ids,
        total_saving,
        color="#E3606D",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Total Energy Saving",
        zorder=3,
    )[0]

    ax_left.set_xlabel("Case Index", fontsize=16)
    ax_left.set_ylabel("Total Energy Saving", fontsize=16, color="black")
    ax_right.set_ylabel("Iterations", fontsize=16, color="black")
    ax_left.tick_params(axis="x", labelsize=13, colors="black")
    ax_left.tick_params(axis="y", labelsize=13, colors="black")
    ax_right.tick_params(axis="y", labelsize=13, colors="black")
    ax_left.set_ylim(0.3, 0.8)
    ax_left.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax_right.set_ylim(1, 11)
    ax_right.set_yticks([1, 3, 5, 7, 9, 11])

    ax_left.set_xticks(case_ids)
    ax_left.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    handles = [line, bars]
    labels = [h.get_label() for h in handles]
    legend = ax_left.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
    )
    legend.set_zorder(30)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot total energy saving (line) and iterations (bar) from manual optimization Excel data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_EXCEL),
        help=f"Path to input Excel file (default: {DEFAULT_EXCEL})",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=DEFAULT_SHEET,
        help=f"Excel sheet name (default: {DEFAULT_SHEET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("best_saving_cases_plot.png"),
        help="Path to output PNG image.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("best_saving_cases_plot.svg"),
        help="Path to output SVG image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    build_plot(
        input_excel=args.input,
        sheet_name=args.sheet,
        output_png=args.output,
        output_svg=args.output_svg,
        show_plot=args.show,
    )
    print(f"PNG saved to: {args.output.resolve()}")
    print(f"SVG saved to: {args.output_svg.resolve()}")


if __name__ == "__main__":
    main()
