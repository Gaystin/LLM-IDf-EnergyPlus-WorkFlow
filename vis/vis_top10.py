from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


DEFAULT_TOP_K = 10
CITY_ORDER = ["Harbin", "Beijing", "Shanghai", "Wuhan", "Chengdu", "Guangzhou", "Kunming"]
REQUIRED_SHEETS = {"run_summary", "object_frequency", "field_frequency"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_input_root() -> Path:
    return _repo_root() / "各城市迭代结果_50%节能率"


def _default_output_root() -> Path:
    return _repo_root() / "figures" / "top10"


def _city_colors(labels: list[str]) -> dict[str, tuple]:
    n = max(len(labels), 2)
    start_rgb = np.array(mcolors.to_rgb("#4C78A8"))
    end_rgb = np.array(mcolors.to_rgb("#E97F89"))
    values = np.linspace(0.0, 1.0, n)
    colors: dict[str, tuple] = {}
    for idx, label in enumerate(labels):
        rgb = start_rgb * (1.0 - values[idx]) + end_rgb * values[idx]
        colors[label] = (*rgb.tolist(), 1.0)
    return colors


def _gradient_colors(count: int) -> list[tuple]:
    if count <= 0:
        return []
    labels = [str(index) for index in range(count)]
    return list(_city_colors(labels).values())


def _ordered_cities(cities: list[str]) -> list[str]:
    existing = sorted(set(cities))
    preferred = [city for city in CITY_ORDER if city in existing]
    tail = [city for city in existing if city not in preferred]
    return preferred + tail


def _city_name_from_workbook(path: Path) -> str:
    suffix = "_统计汇总"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _format_field_label(name: str) -> str:
    if "." not in name:
        return name
    object_part, field_part = name.split(".", 1)
    return f"{object_part}\n{field_part}"


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


def _ensure_run_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("city", "run_tag", "workflow_id"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].astype(str)
    out["run_key"] = out["city"] + "|" + out["run_tag"] + "|" + out["workflow_id"]
    return out


def _normalize_frequency_by_iterations(freq_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
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


def _normalize_city_frequency_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"entity_type", "name", "frequency"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[:, ["entity_type", "name", "frequency"]].copy()
    out["entity_type"] = out["entity_type"].astype(str).str.lower().str.strip()
    out["name"] = out["name"].astype(str).str.strip()
    out["frequency"] = pd.to_numeric(out["frequency"], errors="coerce")
    out = out.dropna(subset=["frequency"])
    out = out.groupby(["entity_type", "name"], as_index=False)["frequency"].mean()
    return out


def load_city_workbooks(input_root: Path) -> dict[str, pd.DataFrame]:
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"找不到输入目录: {input_root}")

    city_tables: dict[str, pd.DataFrame] = {}

    for city_dir in sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        workbook_candidates = sorted(city_dir.glob("*_统计汇总.xlsx"))
        if not workbook_candidates:
            continue

        workbook_path = workbook_candidates[0]
        xls = pd.ExcelFile(workbook_path)
        missing_sheets = REQUIRED_SHEETS.difference(set(xls.sheet_names))
        if missing_sheets:
            continue

        run_df = pd.read_excel(workbook_path, sheet_name="run_summary")
        obj_df = pd.read_excel(workbook_path, sheet_name="object_frequency")
        fld_df = pd.read_excel(workbook_path, sheet_name="field_frequency")

        run_df = _filter_success(_normalize_status(run_df))
        obj_df = _filter_success(_normalize_status(obj_df))
        fld_df = _filter_success(_normalize_status(fld_df))

        run_df = _ensure_run_key(run_df)
        obj_df = _ensure_run_key(obj_df)
        fld_df = _ensure_run_key(fld_df)

        if "iterations_executed" in run_df.columns:
            run_df["iterations_executed"] = pd.to_numeric(run_df["iterations_executed"], errors="coerce")

        obj_df = _normalize_frequency_by_iterations(obj_df, run_df)
        fld_df = _normalize_frequency_by_iterations(fld_df, run_df)

        obj_city = (
            obj_df.groupby("object_type", as_index=False)["frequency"]
            .mean()
            .rename(columns={"object_type": "name"})
        )
        obj_city.insert(0, "entity_type", "object")

        fld_city = (
            fld_df.groupby("object_field", as_index=False)["frequency"]
            .mean()
            .rename(columns={"object_field": "name"})
        )
        fld_city.insert(0, "entity_type", "field")

        city_name = _city_name_from_workbook(workbook_path)
        city_tables[city_name] = _normalize_city_frequency_frame(pd.concat([obj_city, fld_city], ignore_index=True))

    if not city_tables:
        raise ValueError(f"在目录 {input_root} 下没有找到任何有效的城市统计汇总文件。")

    return city_tables


def build_overall_summary(city_tables: dict[str, pd.DataFrame], entity_type: str, top_k: int) -> pd.DataFrame:
    cities = _ordered_cities(list(city_tables.keys()))
    all_names = sorted(
        {
            name
            for city in cities
            for name in city_tables[city].loc[city_tables[city]["entity_type"] == entity_type, "name"].tolist()
        }
    )

    rows: list[dict[str, float | int | str]] = []
    for name in all_names:
        city_values: list[float] = []
        for city in cities:
            city_df = city_tables[city]
            subset = city_df[(city_df["entity_type"] == entity_type) & (city_df["name"] == name)]
            value = float(subset["frequency"].iloc[0]) if not subset.empty else 0.0
            city_values.append(value)

        rows.append(
            {
                "name": name,
                "mean_city_frequency": float(np.mean(city_values)) if city_values else 0.0,
                "sum_city_frequency": float(np.sum(city_values)) if city_values else 0.0,
                "present_city_count": int(np.sum(np.array(city_values) > 0.0)) if city_values else 0,
                "city_count": len(cities),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["mean_city_frequency", "sum_city_frequency", "name"], ascending=[False, False, True])
    summary = summary.reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary.head(min(top_k, len(summary)))


def _annotate_horizontal_bars(ax: plt.Axes, bars, value_formatter) -> None:
    max_value = max((float(bar.get_width()) for bar in bars), default=0.0)
    offset = max(max_value * 0.01, 0.01)
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            value_formatter(value),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )


def plot_topk_bar(
    frame: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: Path,
    output_svg: Path,
    label_col: str = "name",
    value_col: str = "frequency",
    label_formatter=None,
    value_formatter=None,
    show_plot: bool = False,
) -> None:
    plot_df = frame.copy().reset_index(drop=True)
    if plot_df.empty:
        raise ValueError(f"No data available for plotting: {title}")

    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).to_numpy()
    bar_colors = _gradient_colors(len(plot_df))

    if label_formatter is not None:
        labels = [label_formatter(label) for label in labels]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 8.0))

    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=0.62, color=bar_colors, alpha=0.55, zorder=2)

    # Reserve right-side room so value labels won't touch the right spine.
    max_value = float(np.max(values)) if len(values) else 0.0
    right_pad = max(max_value * 0.10, 1.0)
    ax.set_xlim(0.0, max_value + right_pad)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color="black")
    ax.tick_params(axis="x", labelsize=13, colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    if value_formatter is None:
        value_formatter = lambda value: f"{value:.2f}%"

    _annotate_horizontal_bars(ax, bars, value_formatter)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320)
    fig.savefig(output_svg)

    if show_plot:
        plt.show()
    plt.close(fig)


def save_tables(city_tables: dict[str, pd.DataFrame], output_root: Path, top_k: int, show_plot: bool) -> None:
    city_summary_frames: list[pd.DataFrame] = []
    overall_summary_frames: list[pd.DataFrame] = []

    for city in _ordered_cities(list(city_tables.keys())):
        city_df = city_tables[city]
        for entity_type, label_formatter in (("object", None), ("field", _format_field_label)):
            subset = city_df[city_df["entity_type"] == entity_type].copy()
            subset = subset.sort_values(["frequency", "name"], ascending=[False, True]).head(top_k)
            subset = subset.reset_index(drop=True)
            subset.insert(0, "city", city)
            subset.insert(1, "rank", np.arange(1, len(subset) + 1))
            city_summary_frames.append(subset.assign(plot_scope="city", entity_kind=entity_type))

            output_dir = output_root / "cities" / city
            title = f"{city} {entity_type.capitalize()} Top {len(subset)}"
            plot_topk_bar(
                subset,
                title=title,
                xlabel="Mean Normalized Frequency (%)",
                ylabel=entity_type.capitalize(),
                output_png=output_dir / f"{entity_type}_top{top_k}.png",
                output_svg=output_dir / f"{entity_type}_top{top_k}.svg",
                label_formatter=label_formatter,
                value_formatter=lambda value: f"{value:.2f}%",
                show_plot=show_plot,
            )

    for entity_type, label_formatter in (("object", None), ("field", _format_field_label)):
        overall = build_overall_summary(city_tables, entity_type, top_k)
        overall_summary_frames.append(overall.assign(plot_scope="overall", entity_kind=entity_type))

        output_dir = output_root / "overall"
        title = f"Overall {entity_type.capitalize()} Top {len(overall)}"
        plot_topk_bar(
            overall,
            title=title,
            xlabel="Mean Normalized Frequency Across Cities (%)",
            ylabel=entity_type.capitalize(),
            output_png=output_dir / f"overall_{entity_type}_top{top_k}.png",
            output_svg=output_dir / f"overall_{entity_type}_top{top_k}.svg",
            label_formatter=label_formatter,
            value_col="mean_city_frequency",
            value_formatter=lambda value: f"{value:.2f}%",
            show_plot=show_plot,
        )

    if city_summary_frames:
        pd.concat(city_summary_frames, ignore_index=True).to_csv(
            output_root / "city_top10_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if overall_summary_frames:
        pd.concat(overall_summary_frames, ignore_index=True).to_csv(
            output_root / "overall_top10_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top object and field frequencies per city and overall using vis v2 normalization logic."
    )
    parser.add_argument("--input-root", type=Path, default=_default_input_root(), help="Root folder with city result workbooks.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root(), help="Folder to save figures and summaries.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K categories to keep in each chart.")
    parser.add_argument("--show", action="store_true", help="Show figures after saving them.")
    args = parser.parse_args()

    city_tables = load_city_workbooks(args.input_root)
    save_tables(city_tables, args.output_root, max(1, args.top_k), args.show)

    print(f"Processed cities: {', '.join(_ordered_cities(list(city_tables.keys())))}")
    print(f"Figures saved to: {args.output_root.resolve()}")
    print("Frequency metric: per-run frequency / iterations_executed, then city mean; overall is mean of city means.")


if __name__ == "__main__":
    main()
