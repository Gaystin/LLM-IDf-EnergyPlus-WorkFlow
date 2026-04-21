from __future__ import annotations

import argparse
import logging
import math
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
from matplotlib import colors as mcolors

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


GROUP_ORDER = ["Task-Low", "Task-Medium", "Task-High", "Task-Top"]
GROUP_LABELS = {
    "Task-Low": "Low",
    "Task-Medium": "Medium",
    "Task-High": "High",
    "Task-Top": "Top",
}
GROUP_SCORE = {"Task-Low": 1, "Task-Medium": 2, "Task-High": 3, "Task-Top": 4}
GROUP_RANK = {group: idx for idx, group in enumerate(GROUP_ORDER)}
TOP_K = 12
INPUT_ROOT = Path(__file__).parent.parent / "Task-result"
FIGURE_ROOT = Path(__file__).parent.parent / "figures" / "task_analysis"
LOG_ROOT = Path(__file__).parent.parent / "log" / "task_analysis"

SHEET_RUN_SUMMARY = "run_summary"
SHEET_ROUND_DETAILS = "round_details"
SHEET_EARLY_STOP = "early_stop"
SHEET_OBJECT_FREQ = "object_frequency"
SHEET_FIELD_FREQ = "field_frequency"
SHEET_ITER_OBJECT_FREQ = "iteration_object_frequency"
SHEET_ITER_FIELD_FREQ = "iteration_field_frequency"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    base = out_path.with_suffix("") if out_path.suffix else out_path
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.2)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _blue_to_red_cmap() -> mcolors.Colormap:
    color_list = ["#5B84B1", "#D9E6F2", "#F4E6DE", "#E47C6B"]
    return mcolors.LinearSegmentedColormap.from_list("warm_blue_to_red", color_list, N=256)


def _group_colors(groups: Sequence[str]) -> Dict[str, tuple]:
    ordered = [group for group in GROUP_ORDER if group in set(groups)]
    if not ordered:
        ordered = list(GROUP_ORDER)
    n = max(len(ordered), 2)
    start_rgb = np.array(mcolors.to_rgb("#4C78A8"))
    end_rgb = np.array(mcolors.to_rgb("#E97F89"))
    vals = np.linspace(0.0, 1.0, n)
    cmap: Dict[str, tuple] = {}
    for idx, group in enumerate(ordered):
        rgb = start_rgb * (1.0 - vals[idx]) + end_rgb * vals[idx]
        cmap[group] = (*rgb.tolist(), 1.0)
    return cmap


def _gradient_hex(start_hex: str, end_hex: str, n: int) -> List[tuple]:
    n = max(int(n), 1)
    start_rgb = np.array(mcolors.to_rgb(start_hex))
    end_rgb = np.array(mcolors.to_rgb(end_hex))
    vals = np.linspace(0.0, 1.0, n)
    return [tuple((start_rgb * (1.0 - v) + end_rgb * v).tolist()) for v in vals]


def _wrap_label(label: str, width: int = 18) -> str:
    text = str(label)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_status" in out.columns:
        out["run_status"] = out["run_status"].astype(str).str.lower().str.strip()
    return out


def _filter_success(df: pd.DataFrame) -> pd.DataFrame:
    if "run_status" not in df.columns:
        return df.copy()
    mask = df["run_status"].isna() | (df["run_status"] == "success")
    return df.loc[mask].copy()


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _discover_workbooks(input_root: Path) -> List[Path]:
    workbooks: List[Path] = []
    for group in GROUP_ORDER:
        group_dir = input_root / group
        if not group_dir.exists():
            continue
        workbooks.extend(sorted(group_dir.glob("*/*统计汇总.xlsx")))
    return [path for path in workbooks if not path.name.startswith("~$")]


def _parse_group(path: Path) -> str:
    for parent in path.parents:
        if parent.name in GROUP_ORDER:
            return parent.name
    raise ValueError(f"Cannot infer complexity group from path: {path}")


def _parse_task_id(path: Path) -> str:
    return path.parent.name


def _parse_city(path: Path) -> str:
    stem = path.stem
    suffix = "_统计汇总"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _first_valid_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[0]


def _compute_round_metrics(round_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {
        "baseline_total_site_energy_kwh": np.nan,
        "final_total_site_energy_kwh": np.nan,
        "best_total_site_energy_kwh": np.nan,
        "best_energy_iteration": np.nan,
        "energy_drop_kwh": np.nan,
        "energy_drop_pct": np.nan,
        "final_saving_pct": np.nan,
        "best_round_saving_pct": np.nan,
        "saving_improvement_span": np.nan,
        "energy_slope_per_iteration": np.nan,
        "round_count": float(len(round_df)),
    }

    if round_df.empty or "iteration" not in round_df.columns:
        return out

    ordered = round_df.copy()
    ordered["iteration"] = _to_numeric(ordered["iteration"])
    ordered = ordered.sort_values("iteration").reset_index(drop=True)

    if "total_site_energy_kwh" in ordered.columns:
        energy = _to_numeric(ordered["total_site_energy_kwh"])
        valid = ordered[energy.notna()].copy()
        if not valid.empty:
            ordered["total_site_energy_kwh"] = energy
            out["baseline_total_site_energy_kwh"] = float(valid.iloc[0]["total_site_energy_kwh"])
            out["final_total_site_energy_kwh"] = float(valid.iloc[-1]["total_site_energy_kwh"])
            best_idx = valid["total_site_energy_kwh"].idxmin()
            best_row = valid.loc[best_idx]
            out["best_total_site_energy_kwh"] = float(best_row["total_site_energy_kwh"])
            out["best_energy_iteration"] = float(best_row["iteration"])
            baseline = out["baseline_total_site_energy_kwh"]
            best = out["best_total_site_energy_kwh"]
            if baseline and not math.isnan(baseline) and not math.isnan(best) and baseline != 0:
                out["energy_drop_kwh"] = float(baseline - best)
                out["energy_drop_pct"] = float((baseline - best) / baseline * 100.0)

            if len(valid) >= 2:
                x = valid["iteration"].to_numpy(dtype=float)
                y = valid["total_site_energy_kwh"].to_numpy(dtype=float)
                slope, _ = np.polyfit(x, y, 1)
                out["energy_slope_per_iteration"] = float(slope)

    if "saving_pct_total_site_vs_baseline" in ordered.columns:
        saving = _to_numeric(ordered["saving_pct_total_site_vs_baseline"])
        valid_saving = saving.dropna()
        if not valid_saving.empty:
            out["final_saving_pct"] = float(valid_saving.iloc[-1])
            out["best_round_saving_pct"] = float(valid_saving.max())
            out["saving_improvement_span"] = float(valid_saving.max() - valid_saving.min())

    return out


def _normalize_frequency_sheet(df: pd.DataFrame, *, name_col: str, iterations_executed: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out[name_col] = out[name_col].astype(str).str.strip()
    out["frequency"] = _to_numeric(out["frequency"])
    out = out.dropna(subset=["frequency"])
    out = out.groupby(name_col, as_index=False)["frequency"].sum()
    if iterations_executed and not math.isnan(iterations_executed):
        out["normalized_frequency_pct"] = out["frequency"] / float(iterations_executed) * 100.0
    else:
        out["normalized_frequency_pct"] = np.nan
    return out


def _normalize_iteration_frequency_sheet(df: pd.DataFrame, *, name_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out[name_col] = out[name_col].astype(str).str.strip()
    out["iteration"] = _to_numeric(out["iteration"])
    out["frequency"] = _to_numeric(out["frequency"])
    out = out.dropna(subset=["iteration", "frequency"])
    out = out.groupby(["iteration", name_col], as_index=False)["frequency"].sum()
    return out


def _load_workbook(path: Path, logger: logging.Logger) -> Dict[str, pd.DataFrame] | None:
    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        logger.warning("Skip workbook: %s | %s", path, exc)
        return None

    required = {
        SHEET_RUN_SUMMARY,
        SHEET_ROUND_DETAILS,
        SHEET_EARLY_STOP,
        SHEET_OBJECT_FREQ,
        SHEET_FIELD_FREQ,
        SHEET_ITER_OBJECT_FREQ,
        SHEET_ITER_FIELD_FREQ,
    }
    missing = required.difference(set(xls.sheet_names))
    if missing:
        logger.warning("Skip workbook due to missing sheets: %s | missing=%s", path, sorted(missing))
        return None

    data = {
        SHEET_RUN_SUMMARY: pd.read_excel(xls, sheet_name=SHEET_RUN_SUMMARY),
        SHEET_ROUND_DETAILS: pd.read_excel(xls, sheet_name=SHEET_ROUND_DETAILS),
        SHEET_EARLY_STOP: pd.read_excel(xls, sheet_name=SHEET_EARLY_STOP),
        SHEET_OBJECT_FREQ: pd.read_excel(xls, sheet_name=SHEET_OBJECT_FREQ),
        SHEET_FIELD_FREQ: pd.read_excel(xls, sheet_name=SHEET_FIELD_FREQ),
        SHEET_ITER_OBJECT_FREQ: pd.read_excel(xls, sheet_name=SHEET_ITER_OBJECT_FREQ),
        SHEET_ITER_FIELD_FREQ: pd.read_excel(xls, sheet_name=SHEET_ITER_FIELD_FREQ),
    }

    for key, df in data.items():
        data[key] = _filter_success(_normalize_status(df))

    return data


def load_analysis_tables(input_root: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbooks = _discover_workbooks(input_root)
    if not workbooks:
        raise FileNotFoundError(f"No task workbooks found under: {input_root}")

    task_rows: List[Dict[str, object]] = []
    round_rows: List[Dict[str, object]] = []
    early_rows: List[Dict[str, object]] = []
    object_rows: List[Dict[str, object]] = []
    field_rows: List[Dict[str, object]] = []
    iter_rows: List[Dict[str, object]] = []

    for workbook in workbooks:
        payload = _load_workbook(workbook, logger)
        if payload is None:
            continue

        group = _parse_group(workbook)
        task_id = _parse_task_id(workbook)
        city = _parse_city(workbook)

        run_df = payload[SHEET_RUN_SUMMARY].copy()
        round_df = payload[SHEET_ROUND_DETAILS].copy()
        early_df = payload[SHEET_EARLY_STOP].copy()
        obj_df = payload[SHEET_OBJECT_FREQ].copy()
        fld_df = payload[SHEET_FIELD_FREQ].copy()
        iter_obj_df = payload[SHEET_ITER_OBJECT_FREQ].copy()
        iter_fld_df = payload[SHEET_ITER_FIELD_FREQ].copy()

        if run_df.empty:
            logger.warning("Skip workbook with empty run_summary: %s", workbook)
            continue

        run_row = _first_valid_row(run_df)
        iterations_executed = float(_to_numeric(pd.Series([run_row.get("iterations_executed", np.nan)])).iloc[0])
        derived_round = _compute_round_metrics(round_df)

        task_row: Dict[str, object] = {
            "complexity_group": group,
            "complexity_label": GROUP_LABELS[group],
            "complexity_score": GROUP_SCORE[group],
            "task_id": task_id,
            "city": city,
            "workbook": str(workbook),
            "iterations_executed": iterations_executed,
            "best_iteration": _to_numeric(pd.Series([run_row.get("best_iteration", np.nan)])).iloc[0],
            "best_saving_pct_total_site": _to_numeric(pd.Series([run_row.get("best_saving_pct_total_site", np.nan)])).iloc[0],
            "avg_saving_pct_total_site": _to_numeric(pd.Series([run_row.get("avg_saving_pct_total_site", np.nan)])).iloc[0],
            "avg_round_duration_sec": _to_numeric(pd.Series([run_row.get("avg_round_duration_sec", np.nan)])).iloc[0],
            "workflow_total_duration_sec": _to_numeric(pd.Series([run_row.get("workflow_total_duration_sec", np.nan)])).iloc[0],
            "llm_total_duration_sec": _to_numeric(pd.Series([run_row.get("llm_total_duration_sec", np.nan)])).iloc[0],
            "sim_total_duration_sec": _to_numeric(pd.Series([run_row.get("sim_total_duration_sec", np.nan)])).iloc[0],
            "llm_calls_count": _to_numeric(pd.Series([run_row.get("llm_calls_count", np.nan)])).iloc[0],
            "sim_calls_count": _to_numeric(pd.Series([run_row.get("sim_calls_count", np.nan)])).iloc[0],
            "total_tokens": _to_numeric(pd.Series([run_row.get("total_tokens", np.nan)])).iloc[0],
            "input_tokens": _to_numeric(pd.Series([run_row.get("input_tokens", np.nan)])).iloc[0],
            "output_tokens": _to_numeric(pd.Series([run_row.get("output_tokens", np.nan)])).iloc[0],
            "cached_input_tokens": _to_numeric(pd.Series([run_row.get("cached_input_tokens", np.nan)])).iloc[0],
            "run_status": str(run_row.get("run_status", "")).lower().strip(),
            "error_message": run_row.get("error_message", np.nan),
        }
        task_row.update(derived_round)

        if not early_df.empty:
            early_row = _first_valid_row(early_df)
            task_row["early_stop_iteration"] = _to_numeric(pd.Series([early_row.get("early_stop_iteration", np.nan)])).iloc[0]
            task_row["early_stop_reason"] = early_row.get("early_stop_reason", np.nan)
            task_row["early_stop_iterations_executed"] = _to_numeric(pd.Series([early_row.get("iterations_executed", np.nan)])).iloc[0]
        else:
            task_row["early_stop_iteration"] = np.nan
            task_row["early_stop_reason"] = np.nan
            task_row["early_stop_iterations_executed"] = np.nan

        obj_norm = _normalize_frequency_sheet(obj_df, name_col="object_type", iterations_executed=iterations_executed)
        fld_norm = _normalize_frequency_sheet(fld_df, name_col="object_field", iterations_executed=iterations_executed)
        iter_obj_norm = _normalize_iteration_frequency_sheet(iter_obj_df, name_col="object_type")
        iter_fld_norm = _normalize_iteration_frequency_sheet(iter_fld_df, name_col="object_field")

        task_row["unique_object_count"] = float(len(obj_norm))
        task_row["unique_field_count"] = float(len(fld_norm))
        task_row["mean_object_frequency_pct"] = float(obj_norm["normalized_frequency_pct"].mean()) if not obj_norm.empty else np.nan
        task_row["mean_field_frequency_pct"] = float(fld_norm["normalized_frequency_pct"].mean()) if not fld_norm.empty else np.nan
        task_row["max_object_frequency_pct"] = float(obj_norm["normalized_frequency_pct"].max()) if not obj_norm.empty else np.nan
        task_row["max_field_frequency_pct"] = float(fld_norm["normalized_frequency_pct"].max()) if not fld_norm.empty else np.nan

        task_rows.append(task_row)

        for _, row in round_df.iterrows():
            round_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    **row.to_dict(),
                }
            )

        for _, row in early_df.iterrows():
            early_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    **row.to_dict(),
                }
            )

        for _, row in obj_norm.iterrows():
            object_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    **row.to_dict(),
                }
            )

        for _, row in fld_norm.iterrows():
            field_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    **row.to_dict(),
                }
            )

        for _, row in iter_obj_norm.iterrows():
            iter_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    "entity_kind": "object",
                    **row.to_dict(),
                }
            )

        for _, row in iter_fld_norm.iterrows():
            iter_rows.append(
                {
                    "complexity_group": group,
                    "complexity_label": GROUP_LABELS[group],
                    "complexity_score": GROUP_SCORE[group],
                    "task_id": task_id,
                    "city": city,
                    "workbook": str(workbook),
                    "entity_kind": "field",
                    **row.to_dict(),
                }
            )

        logger.info("Loaded workbook: %s | group=%s | task=%s", workbook.name, group, task_id)

    task_df = pd.DataFrame(task_rows)
    round_df = pd.DataFrame(round_rows)
    early_df = pd.DataFrame(early_rows)
    object_df = pd.DataFrame(object_rows)
    field_df = pd.DataFrame(field_rows)
    iteration_df = pd.DataFrame(iter_rows)

    if task_df.empty:
        raise RuntimeError("No valid workbooks were loaded.")

    for col in [
        "iterations_executed",
        "best_iteration",
        "best_saving_pct_total_site",
        "avg_saving_pct_total_site",
        "avg_round_duration_sec",
        "workflow_total_duration_sec",
        "llm_total_duration_sec",
        "sim_total_duration_sec",
        "llm_calls_count",
        "sim_calls_count",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "unique_object_count",
        "unique_field_count",
        "mean_object_frequency_pct",
        "mean_field_frequency_pct",
        "max_object_frequency_pct",
        "max_field_frequency_pct",
        "early_stop_iteration",
        "early_stop_iterations_executed",
        "baseline_total_site_energy_kwh",
        "final_total_site_energy_kwh",
        "best_total_site_energy_kwh",
        "best_energy_iteration",
        "energy_drop_kwh",
        "energy_drop_pct",
        "final_saving_pct",
        "best_round_saving_pct",
        "saving_improvement_span",
        "energy_slope_per_iteration",
        "round_count",
    ]:
        if col in task_df.columns:
            task_df[col] = _to_numeric(task_df[col])

    if "iteration" in round_df.columns:
        round_df["iteration"] = _to_numeric(round_df["iteration"])
    if "iteration" in early_df.columns:
        early_df["iteration"] = _to_numeric(early_df["iteration"])
    if "iteration" in object_df.columns:
        object_df["iteration"] = _to_numeric(object_df["iteration"])
    if "iteration" in field_df.columns:
        field_df["iteration"] = _to_numeric(field_df["iteration"])
    if "iteration" in iteration_df.columns:
        iteration_df["iteration"] = _to_numeric(iteration_df["iteration"])

    return task_df, round_df, early_df, object_df, field_df, iteration_df


def _annotate_bar_values(ax: plt.Axes, bars, formatter) -> None:
    for bar in bars:
        value = bar.get_height() if hasattr(bar, "get_height") else bar.get_width()
        if np.isnan(value):
            continue
        if hasattr(bar, "get_height"):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                formatter(value),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                zorder=4,
            )
        else:
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2.0,
                formatter(value),
                ha="left",
                va="center",
                fontsize=10,
                color="black",
                zorder=4,
            )


def plot_group_bars(
    summary: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    out_path: Path,
    ylim: Tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    plot_df = summary[["complexity_group", value_col]].dropna().copy()
    plot_df["complexity_group"] = pd.Categorical(plot_df["complexity_group"], categories=GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("complexity_group")

    groups = plot_df["complexity_group"].astype(str).tolist()
    colors = _group_colors(groups)
    bar_colors = [colors[g] for g in groups]
    x = np.arange(len(groups))
    values = plot_df[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(x, values, width=0.65, color=bar_colors, alpha=0.45, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=11)
    ax.set_xlabel("Complexity Group")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", visible=False)
    _annotate_bar_values(ax, bars, lambda value: f"{value:,.2f}" if abs(value) >= 100 else f"{value:.2f}")
    _save_figure(fig, out_path)


def _red_blue_palette(n: int) -> List[tuple]:
    base = _blue_to_red_cmap()
    trim = base(np.linspace(0.95, 0.05, max(n, 2)))
    if n <= 1:
        return [tuple(trim[0])]
    return [tuple(trim[i]) for i in range(n)]


def plot_group_iteration_saving_combo(summary: pd.DataFrame, *, out_path: Path) -> None:
    required = ["complexity_group", "best_iteration_mean", "best_saving_pct_total_site_mean"]
    if any(col not in summary.columns for col in required):
        return

    plot_df = summary[required].dropna().copy()
    if plot_df.empty:
        return

    plot_df["complexity_group"] = pd.Categorical(plot_df["complexity_group"], categories=GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("complexity_group").reset_index(drop=True)

    groups = plot_df["complexity_group"].astype(str).tolist()
    labels = [GROUP_LABELS.get(group, group) for group in groups]
    colors = _group_colors(groups)
    bar_colors = [colors[group] for group in groups]

    x = np.arange(len(groups))
    iter_values = plot_df["best_iteration_mean"].to_numpy(dtype=float)
    saving_values = plot_df["best_saving_pct_total_site_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax_right = ax.twinx()
    ax.set_axisbelow(True)
    ax_right.grid(False)

    bars = ax.bar(
        x,
        iter_values,
        width=0.52,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.55,
        hatch="///",
        zorder=3,
    )
    ax_right.plot(
        x,
        saving_values,
        color="#E3606D",
        marker="o",
        linewidth=2.2,
        markersize=6,
        zorder=6,
    )

    ax.set_xlabel("Complexity Group", fontsize=16)
    ax.set_ylabel("Mean Best Iteration", fontsize=16)
    ax_right.set_ylabel("Mean Best Saving (%)", fontsize=16, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=13, color="black")
    ax.tick_params(axis="x", labelsize=14, colors="black")
    ax.tick_params(axis="y", labelsize=14, colors="black")
    ax_right.tick_params(axis="y", labelsize=14, colors="black")
    ax.set_ylim(6, 16)
    ax.set_yticks([6, 8, 10, 12, 14, 16])
    ax_right.set_ylim(60, 75)
    ax_right.set_yticks([60, 63, 66, 69, 72, 75])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#BBBBBB", edgecolor="#333333", hatch="///", label="Best iteration (bar)"),
        plt.Line2D([0], [0], color="#E3606D", marker="o", linewidth=2.2, label="Best saving (line)"),
    ]
    legend = fig.legend(
        handles=legend_handles,
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
    _save_figure(fig, out_path)


def _top_entity_summary(entity_df: pd.DataFrame, *, name_col: str, top_k: int = TOP_K) -> pd.DataFrame:
    plot_df = entity_df.copy()
    plot_df[name_col] = plot_df[name_col].astype(str)
    summary = (
        plot_df.groupby(name_col, as_index=False)["normalized_frequency_pct"]
        .mean()
        .sort_values(["normalized_frequency_pct", name_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary.head(min(top_k, len(summary)))


def plot_top_entity_barh(
    entity_df: pd.DataFrame,
    *,
    name_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    summary = _top_entity_summary(entity_df, name_col=name_col, top_k=top_k)
    if summary.empty:
        return summary

    plot_df = summary.sort_values("normalized_frequency_pct", ascending=True).reset_index(drop=True)
    labels = plot_df[name_col].astype(str).tolist()
    values = plot_df["normalized_frequency_pct"].to_numpy(dtype=float)
    colors = _gradient_hex("#D9E6F2", "#E47C6B", len(plot_df))

    fig, ax = plt.subplots(figsize=(13.5, max(6.2, 0.48 * len(plot_df) + 2.6)))
    y = np.arange(len(plot_df))
    bars = ax.barh(y, values, height=0.62, color=colors, alpha=0.55, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap_label(label) for label in labels], fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entity")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", visible=False)
    max_value = float(np.max(values)) if len(values) else 0.0
    ax.set_xlim(0.0, max_value + max(max_value * 0.12, 1.0))
    ax.invert_yaxis()
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + max(max_value * 0.015, 0.02),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )
    _save_figure(fig, out_path)
    return summary


def _group_entity_matrix(entity_df: pd.DataFrame, *, name_col: str, top_k: int = TOP_K) -> pd.DataFrame:
    if entity_df.empty:
        return pd.DataFrame()
    overall = (
        entity_df.groupby(name_col, as_index=False)["normalized_frequency_pct"]
        .mean()
        .sort_values(["normalized_frequency_pct", name_col], ascending=[False, True])
        .head(top_k)
    )
    top_names = overall[name_col].astype(str).tolist()
    matrix = (
        entity_df[entity_df[name_col].astype(str).isin(top_names)]
        .groupby(["complexity_group", name_col], as_index=False)["normalized_frequency_pct"]
        .mean()
        .pivot(index="complexity_group", columns=name_col, values="normalized_frequency_pct")
        .reindex(GROUP_ORDER)
        .fillna(0.0)
    )
    matrix = matrix.loc[:, top_names]
    return matrix


def _group_entity_cv_specificity(
    entity_df: pd.DataFrame,
    *,
    name_col: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    if entity_df.empty:
        return pd.DataFrame(columns=[name_col, "mean_frequency_pct", "std_frequency_pct", "cv"])

    matrix = (
        entity_df.groupby(["complexity_group", name_col], as_index=False)["normalized_frequency_pct"]
        .mean()
        .pivot(index="complexity_group", columns=name_col, values="normalized_frequency_pct")
        .reindex(GROUP_ORDER)
        .fillna(0.0)
    )
    if matrix.empty:
        return pd.DataFrame(columns=[name_col, "mean_frequency_pct", "std_frequency_pct", "cv"])

    stats = pd.DataFrame(
        {
            name_col: matrix.columns.astype(str),
            "mean_frequency_pct": matrix.mean(axis=0).to_numpy(dtype=float),
            "std_frequency_pct": matrix.std(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=float),
        }
    )
    stats["cv"] = stats["std_frequency_pct"] / stats["mean_frequency_pct"].replace(0, np.nan)
    stats = stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["cv"]) 
    stats = stats.sort_values(["cv", name_col], ascending=[False, True]).head(min(top_k, len(stats))).reset_index(drop=True)
    return stats


def plot_group_heatmap(matrix: pd.DataFrame, *, title: str, out_path: Path) -> None:
    if matrix.empty:
        return
    # 转置矩阵：对象/字段改到纵轴，复杂度改到横轴
    matrix = matrix.T
    labels = matrix.columns.tolist()
    vals = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, max(6.2, len(matrix.index) * 0.36)))
    im = ax.imshow(vals, cmap=_blue_to_red_cmap(), vmin=0, aspect="auto")
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Frequency (%)", rotation=270, labelpad=14)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([GROUP_LABELS.get(label, label) for label in labels])
    ax.set_yticklabels([_wrap_label(label, width=16) for label in matrix.index], fontsize=9)
    ax.set_title(title)
    ax.grid(False)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)

    _save_figure(fig, out_path)


def plot_specificity_cv_bar(stats_df: pd.DataFrame, *, name_col: str, out_path: Path) -> None:
    if stats_df.empty:
        return

    plot_df = stats_df.sort_values("cv", ascending=True).reset_index(drop=True)
    labels = plot_df[name_col].astype(str).tolist()
    values = plot_df["cv"].to_numpy(dtype=float)
    colors = _red_blue_palette(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(6.0, 0.42 * len(plot_df) + 2.2)))
    y = np.arange(len(plot_df))
    bars = ax.barh(y, values, color=colors, alpha=0.65, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap_label(label, width=18) for label in labels], fontsize=11)
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Object" if name_col == "object_type" else "Object Field")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", visible=False)

    max_value = float(np.max(values)) if len(values) else 0.0
    right_pad = max(max_value * 0.12, 0.1)
    ax.set_xlim(0.0, max_value + right_pad)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + max(max_value * 0.015, 0.01),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )

    _save_figure(fig, out_path)


def plot_iteration_curves(round_df: pd.DataFrame, *, metric_col: str, ylabel: str, title: str, out_path: Path) -> None:
    if round_df.empty or metric_col not in round_df.columns:
        return

    plot_df = round_df[["complexity_group", "iteration", metric_col]].dropna().copy()
    if plot_df.empty:
        return

    group_stats = (
        plot_df.groupby(["complexity_group", "iteration"], as_index=False)[metric_col]
        .mean()
        .sort_values(["complexity_group", "iteration"])
    )

    colors = _group_colors(group_stats["complexity_group"].astype(str).tolist())
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for group in GROUP_ORDER:
        sub = group_stats[group_stats["complexity_group"] == group]
        if sub.empty:
            continue
        ax.plot(
            sub["iteration"].to_numpy(dtype=float),
            sub[metric_col].to_numpy(dtype=float),
            color=colors[group],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            label=GROUP_LABELS[group],
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(title="Group", ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.18, 1.0, 0.1), mode="expand", frameon=False, borderaxespad=0.0, handlelength=1.3)
    _save_figure(fig, out_path)


def _group_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_saving_pct_total_site",
        "avg_saving_pct_total_site",
        "iterations_executed",
        "best_iteration",
        "total_tokens",
        "workflow_total_duration_sec",
        "llm_total_duration_sec",
        "sim_total_duration_sec",
        "llm_calls_count",
        "sim_calls_count",
        "unique_object_count",
        "unique_field_count",
        "mean_object_frequency_pct",
        "mean_field_frequency_pct",
        "early_stop_iteration",
        "energy_drop_pct",
        "energy_slope_per_iteration",
    ]
    rows: List[Dict[str, object]] = []
    for group in GROUP_ORDER:
        subset = task_df[task_df["complexity_group"] == group].copy()
        if subset.empty:
            continue
        row: Dict[str, object] = {
            "complexity_group": group,
            "complexity_label": GROUP_LABELS[group],
            "task_count": int(len(subset)),
            "success_rate": float((subset["run_status"].astype(str).str.lower() == "success").mean()),
        }
        for metric in metrics:
            if metric in subset.columns:
                values = _to_numeric(subset[metric])
                row[f"{metric}_mean"] = float(values.mean()) if not values.dropna().empty else np.nan
                row[f"{metric}_median"] = float(values.median()) if not values.dropna().empty else np.nan
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.dropna().shape[0] > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _complexity_trend(task_df: pd.DataFrame, value_col: str) -> float:
    subset = task_df[["complexity_score", value_col]].dropna().copy()
    if len(subset) < 2:
        return np.nan
    return float(subset["complexity_score"].corr(subset[value_col], method="spearman"))


def _top_common_names(task_entity_df: pd.DataFrame, *, name_col: str, top_k: int = TOP_K) -> List[str]:
    summary = (
        task_entity_df.groupby(name_col, as_index=False)["normalized_frequency_pct"].mean()
        .sort_values(["normalized_frequency_pct", name_col], ascending=[False, True])
        .head(top_k)
    )
    return summary[name_col].astype(str).tolist()


def _log_table(logger: logging.Logger, title: str, df: pd.DataFrame, columns: Sequence[str]) -> None:
    logger.info("%s", title)
    if df.empty:
        logger.info("<empty>")
        return
    logger.info("%s", df.loc[:, [col for col in columns if col in df.columns]].to_string(index=False))


def build_summary_artifacts(task_df: pd.DataFrame, round_df: pd.DataFrame, early_df: pd.DataFrame, object_df: pd.DataFrame, field_df: pd.DataFrame, iteration_df: pd.DataFrame, output_root: Path, logger: logging.Logger) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    task_df = task_df.sort_values(["complexity_score", "task_id"]).reset_index(drop=True)
    group_df = _group_summary(task_df)
    task_df.to_csv(summary_dir / "task_summary.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(summary_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
    round_df.to_csv(summary_dir / "round_details_long.csv", index=False, encoding="utf-8-sig")
    early_df.to_csv(summary_dir / "early_stop_long.csv", index=False, encoding="utf-8-sig")
    object_df.to_csv(summary_dir / "object_frequency_long.csv", index=False, encoding="utf-8-sig")
    field_df.to_csv(summary_dir / "field_frequency_long.csv", index=False, encoding="utf-8-sig")
    iteration_df.to_csv(summary_dir / "iteration_frequency_long.csv", index=False, encoding="utf-8-sig")

    logger.info("Task count by complexity group: %s", task_df.groupby("complexity_group").size().to_dict())
    logger.info("Success rate by group: %s", task_df.groupby("complexity_group")["run_status"].apply(lambda s: float((s.astype(str).str.lower() == 'success').mean())).to_dict())

    _log_table(
        logger,
        "Group summary table",
        group_df,
        [
            "complexity_label",
            "task_count",
            "success_rate",
            "best_saving_pct_total_site_mean",
            "avg_saving_pct_total_site_mean",
            "iterations_executed_mean",
            "best_iteration_mean",
            "total_tokens_mean",
            "workflow_total_duration_sec_mean",
            "unique_object_count_mean",
            "unique_field_count_mean",
            "mean_object_frequency_pct_mean",
            "mean_field_frequency_pct_mean",
            "early_stop_iteration_mean",
            "energy_drop_pct_mean",
        ],
    )

    # Correlation summary on task-level data.
    corr_metrics = [
        "best_saving_pct_total_site",
        "avg_saving_pct_total_site",
        "iterations_executed",
        "best_iteration",
        "total_tokens",
        "workflow_total_duration_sec",
        "llm_total_duration_sec",
        "sim_total_duration_sec",
        "llm_calls_count",
        "sim_calls_count",
        "unique_object_count",
        "unique_field_count",
        "mean_object_frequency_pct",
        "mean_field_frequency_pct",
        "early_stop_iteration",
        "energy_drop_pct",
        "energy_slope_per_iteration",
    ]
    corr_rows = []
    for metric in corr_metrics:
        if metric in task_df.columns:
            corr_rows.append({"metric": metric, "spearman_with_complexity": _complexity_trend(task_df, metric)})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_with_complexity", ascending=False)
    corr_df.to_csv(summary_dir / "complexity_spearman_summary.csv", index=False, encoding="utf-8-sig")
    _log_table(logger, "Spearman correlation with complexity score", corr_df, ["metric", "spearman_with_complexity"])

    # Top entity tables.
    object_top = _top_entity_summary(object_df, name_col="object_type", top_k=TOP_K)
    field_top = _top_entity_summary(field_df, name_col="object_field", top_k=TOP_K)
    object_top.to_csv(summary_dir / "top_object_frequency.csv", index=False, encoding="utf-8-sig")
    field_top.to_csv(summary_dir / "top_field_frequency.csv", index=False, encoding="utf-8-sig")

    # Plotting: group means.
    plot_group_iteration_saving_combo(group_df, out_path=figure_dir / "group_best_iteration_saving_combo")
    plot_group_bars(group_df, value_col="iterations_executed_mean", ylabel="Mean iterations executed", out_path=figure_dir / "group_iterations_mean")
    plot_group_bars(group_df, value_col="workflow_total_duration_sec_mean", ylabel="Mean workflow duration (sec)", out_path=figure_dir / "group_workflow_duration_mean")
    plot_group_bars(group_df, value_col="early_stop_iteration_mean", ylabel="Mean early stop iteration", out_path=figure_dir / "group_early_stop_iteration_mean")

    # Top entities by group and overall.
    for group in GROUP_ORDER:
        group_object = object_df[object_df["complexity_group"] == group].copy()
        group_field = field_df[field_df["complexity_group"] == group].copy()
        group_dir = figure_dir / "groups" / group
        plot_top_entity_barh(group_object, name_col="object_type", title="", xlabel="Mean normalized frequency (%)", out_path=group_dir / "top_objects")
        plot_top_entity_barh(group_field, name_col="object_field", title="", xlabel="Mean normalized frequency (%)", out_path=group_dir / "top_fields")

    plot_top_entity_barh(object_df, name_col="object_type", title="", xlabel="Mean normalized frequency (%)", out_path=figure_dir / "overall_top_objects")
    plot_top_entity_barh(field_df, name_col="object_field", title="", xlabel="Mean normalized frequency (%)", out_path=figure_dir / "overall_top_fields")

    object_matrix = _group_entity_matrix(object_df, name_col="object_type", top_k=TOP_K)
    field_matrix = _group_entity_matrix(field_df, name_col="object_field", top_k=TOP_K)
    plot_group_heatmap(object_matrix, title="", out_path=figure_dir / "group_object_frequency_heatmap")
    plot_group_heatmap(field_matrix, title="", out_path=figure_dir / "group_field_frequency_heatmap")

    object_specificity_cv = _group_entity_cv_specificity(object_df, name_col="object_type", top_k=TOP_K)
    field_specificity_cv = _group_entity_cv_specificity(field_df, name_col="object_field", top_k=TOP_K)
    object_specificity_cv.to_csv(summary_dir / "object_frequency_specificity_cv.csv", index=False, encoding="utf-8-sig")
    field_specificity_cv.to_csv(summary_dir / "field_frequency_specificity_cv.csv", index=False, encoding="utf-8-sig")
    plot_specificity_cv_bar(object_specificity_cv, name_col="object_type", out_path=figure_dir / "group_object_frequency_specificity_cv")
    plot_specificity_cv_bar(field_specificity_cv, name_col="object_field", out_path=figure_dir / "group_field_frequency_specificity_cv")

    # Iteration curves.
    plot_iteration_curves(round_df, metric_col="total_site_energy_kwh", ylabel="Total site energy (kWh)", title="", out_path=figure_dir / "iteration_energy_curve")
    plot_iteration_curves(round_df, metric_col="saving_pct_total_site_vs_baseline", ylabel="Saving vs baseline (%)", title="", out_path=figure_dir / "iteration_saving_curve")
    plot_iteration_curves(round_df, metric_col="iteration_total_duration_sec", ylabel="Iteration duration (sec)", title="", out_path=figure_dir / "iteration_duration_curve")

    # Additional high-level log statements.
    if not object_matrix.empty:
        logger.info("Top object names across groups: %s", object_matrix.columns.tolist())
    if not field_matrix.empty:
        logger.info("Top field names across groups: %s", field_matrix.columns.tolist())
    if not object_specificity_cv.empty:
        _log_table(logger, "Top object specificity CV", object_specificity_cv, ["object_type", "mean_frequency_pct", "std_frequency_pct", "cv"])
    if not field_specificity_cv.empty:
        _log_table(logger, "Top field specificity CV", field_specificity_cv, ["object_field", "mean_frequency_pct", "std_frequency_pct", "cv"])

    object_overlap = {}
    field_overlap = {}
    for left in GROUP_ORDER:
        for right in GROUP_ORDER:
            left_object = set(object_df[object_df["complexity_group"] == left]["object_type"].astype(str).tolist())
            right_object = set(object_df[object_df["complexity_group"] == right]["object_type"].astype(str).tolist())
            left_field = set(field_df[field_df["complexity_group"] == left]["object_field"].astype(str).tolist())
            right_field = set(field_df[field_df["complexity_group"] == right]["object_field"].astype(str).tolist())
            object_overlap[(left, right)] = float(len(left_object & right_object) / max(len(left_object | right_object), 1))
            field_overlap[(left, right)] = float(len(left_field & right_field) / max(len(left_field | right_field), 1))

    logger.info("Object frequency Jaccard overlap by group pair: %s", object_overlap)
    logger.info("Field frequency Jaccard overlap by group pair: %s", field_overlap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Task-result outputs across Task-Low / Task-Medium / Task-High / Task-Top.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT, help="Root folder containing Task-result/Task-Top|Task-High|Task-Medium|Task-Low.")
    parser.add_argument("--output-root", type=Path, default=FIGURE_ROOT, help="Root folder for figures and summary tables.")
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT, help="Root folder for analysis log files.")
    parser.add_argument("--run-stamp", type=str, default=None, help="Optional timestamp string for the output subdirectory.")
    return parser


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("task_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_dir / "analysis.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _set_plot_style()

    stamp = args.run_stamp or _run_stamp()
    output_root = _resolve_path(args.output_root) / stamp
    log_root = _resolve_path(args.log_root) / stamp
    logger = setup_logger(log_root)

    input_root = _resolve_path(args.input_root)
    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("Log root: %s", log_root)

    task_df, round_df, early_df, object_df, field_df, iteration_df = load_analysis_tables(input_root, logger)
    build_summary_artifacts(task_df, round_df, early_df, object_df, field_df, iteration_df, output_root, logger)

    logger.info("Analysis complete.")
    logger.info("Figures saved under: %s", (output_root / "figures").resolve())
    logger.info("Tables saved under: %s", (output_root / "tables").resolve())
    logger.info("Log saved under: %s", log_root.resolve())


if __name__ == "__main__":
    main()