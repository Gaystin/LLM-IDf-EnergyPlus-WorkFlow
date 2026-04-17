import argparse
import logging
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Avoid matplotlib tkinter crashes in batch runs on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
from matplotlib import colors as mcolors

# Avoid GUI backend issues on Windows.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DATA_FILE_RE = re.compile(r"^(\d+)__(.+)\.csv$", re.IGNORECASE)
PKL_FILE_RE = re.compile(r"^(\d+)_(.+)\.pkl$", re.IGNORECASE)


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


def _save_figure(fig: plt.Figure, out_file: Path) -> None:
    base = out_file.with_suffix("") if out_file.suffix else out_file
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.2)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _city_color_map(cities: List[str]) -> Dict[str, tuple]:
    ordered = sorted(set(cities))
    n = max(len(ordered), 2)
    start_rgb = np.array(mcolors.to_rgb("#4C78A8"))
    end_rgb = np.array(mcolors.to_rgb("#E97F89"))
    vals = np.linspace(0.0, 1.0, n)

    cmap: Dict[str, tuple] = {}
    for idx, city in enumerate(ordered):
        rgb = start_rgb * (1.0 - vals[idx]) + end_rgb * vals[idx]
        cmap[city] = (*rgb.tolist(), 1.0)
    return cmap


def _blue_to_red_cmap() -> mcolors.Colormap:
    # Match vis v2 style: low value blue, high value red.
    color_list = ["#5B84B1", "#D9E6F2", "#F4E6DE", "#E47C6B"]
    return mcolors.LinearSegmentedColormap.from_list("warm_blue_to_red", color_list, N=256)


def _gradient_hex(start_hex: str, end_hex: str, n: int) -> List[tuple]:
    n = max(n, 1)
    s = np.array(mcolors.to_rgb(start_hex))
    e = np.array(mcolors.to_rgb(end_hex))
    vals = np.linspace(0.0, 1.0, n)
    return [tuple((s * (1.0 - v) + e * v).tolist()) for v in vals]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def strip_idf_comments(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_idf_zone_count(idf_path: Path) -> Tuple[int, List[str]]:
    text = idf_path.read_text(encoding="utf-8", errors="ignore")
    clean = strip_idf_comments(text)
    objects = clean.split(";")
    zone_names: List[str] = []

    for block in objects:
        block = block.strip()
        if not block:
            continue
        fields = [f.strip() for f in block.replace("\n", " ").split(",")]
        fields = [f for f in fields if f]
        if not fields:
            continue
        obj_type = fields[0].upper()
        if obj_type == "ZONE":
            zone_name = fields[1] if len(fields) > 1 else f"ZONE_{len(zone_names) + 1}"
            zone_names.append(zone_name)

    return len(zone_names), zone_names


def parse_epw_metrics(epw_path: Path) -> Dict[str, float]:
    lines = epw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 9:
        raise ValueError(f"EPW file too short: {epw_path}")

    rows = []
    for line in lines[8:]:
        parts = line.split(",")
        if len(parts) < 16:
            continue
        try:
            db = float(parts[6])
            rh = float(parts[8])
            ghr = float(parts[13])
            dnr = float(parts[14])
            dhr = float(parts[15])
        except ValueError:
            continue
        rows.append((db, rh, ghr, dnr, dhr))

    if not rows:
        raise ValueError(f"No valid hourly rows in EPW: {epw_path}")

    arr = np.asarray(rows, dtype=float)
    db = arr[:, 0]
    rh = arr[:, 1]
    ghr = arr[:, 2]
    dnr = arr[:, 3]
    dhr = arr[:, 4]

    cdd18 = np.maximum(db - 18.0, 0.0).sum()
    hdd18 = np.maximum(18.0 - db, 0.0).sum()

    return {
        "weather_hours": float(len(arr)),
        "db_mean": float(db.mean()),
        "db_std": float(db.std()),
        "rh_mean": float(rh.mean()),
        "ghr_sum": float(ghr.sum()),
        "dnr_sum": float(dnr.sum()),
        "dhr_sum": float(dhr.sum()),
        "cdd18": float(cdd18),
        "hdd18": float(hdd18),
    }


def parse_city_from_data_file(path: Path) -> Tuple[str, str]:
    m = DATA_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected data filename format: {path.name}")
    building_id, city = m.group(1), m.group(2)
    return building_id, city


def parse_city_from_pkl_file(path: Path) -> Tuple[str, str]:
    m = PKL_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected pkl filename format: {path.name}")
    building_id, city = m.group(1), m.group(2)
    return building_id, city


def load_city_csv_metrics(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    zone_cols = [c for c in df.columns if c.lower() != "hour"]
    if not zone_cols:
        raise ValueError(f"No zone load columns found in CSV: {csv_path}")

    hourly_total = df[zone_cols].sum(axis=1)
    annual_net = float(hourly_total.sum())
    annual_positive = float(hourly_total.clip(lower=0).sum())
    annual_negative = float(hourly_total.clip(upper=0).sum())
    annual_abs = float(hourly_total.abs().sum())

    return {
        "hours": float(len(df)),
        "zone_count_csv": float(len(zone_cols)),
        "annual_net": annual_net,
        "annual_positive": annual_positive,
        "annual_negative": annual_negative,
        "annual_abs": annual_abs,
        "peak_positive": float(hourly_total.max()),
        "peak_negative": float(hourly_total.min()),
    }


def load_pkl_space_count(pkl_path: Path) -> int:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    building = data.get("building", {})
    spaces = building.get("valid_energy_spaces", [])
    return int(len(spaces))


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def scatter_with_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    a, b = fit_line(x, y)
    color_map = _city_color_map(df["city"].astype(str).tolist())

    fig, ax = plt.subplots(figsize=(11, 6))
    for city in sorted(df["city"].astype(str).unique()):
        sub = df[df["city"].astype(str) == city]
        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=34,
            color=color_map[city],
            label=city,
            alpha=0.95,
            edgecolors="none",
        )

    if xlim is not None:
        x_min, x_max = float(xlim[0]), float(xlim[1])
    elif len(x) > 0:
        x_min, x_max = float(np.min(x)), float(np.max(x))
    else:
        x_min, x_max = 0.0, 1.0

    if len(x) > 0:
        x_min = min(x_min, float(np.min(x)))
        x_max = max(x_max, float(np.max(x)))
        if abs(a) > 1e-12:
            x_from_y = (y - b) / a
            x_min = min(x_min, float(np.min(x_from_y)))
            x_max = max(x_max, float(np.max(x_from_y)))

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    span = x_max - x_min
    pad = 0.05 * span
    line_x = np.asarray([x_min - pad, x_max + pad], dtype=float)
    line_y = a * line_x + b

    # Scale axes with fitted-line endpoints so the fit appears on the chart diagonal.
    ax.set_xlim(float(line_x[0]), float(line_x[1]))
    ax.set_ylim(float(line_y[0]), float(line_y[1]))

    ax.plot(
        line_x,
        line_y,
        linewidth=1.8,
        color="#D62728",
        linestyle="--",
        label="Linear fit",
    )

    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    if len(x) > 0 and not np.isnan(corr):
        ax.text(
            0.02,
            0.98,
            f"r = {corr:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="black",
            fontfamily="DejaVu Sans",
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.25,rounding_size=0.12",
                facecolor="white",
                edgecolor="#B0B0B0",
                linewidth=0.9,
                alpha=1.0,
            ),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(
        title="City",
        ncol=6,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22, 1.0, 0.1),
        mode="expand",
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.3,
    )
    _save_figure(fig, out_path)


def plot_bar_sorted(df: pd.DataFrame, value_col: str, title: str, y_label: str, out_path: Path) -> None:
    xdf = df.sort_values(value_col).reset_index(drop=True)
    vals = xdf[value_col].to_numpy(dtype=float)
    if value_col == "annual_abs":
        colors = _gradient_hex("#F3AEA8", "#C8454B", len(xdf))
    elif value_col == "annual_net":
        neg_idx = [i for i, v in enumerate(vals) if v < 0]
        pos_idx = [i for i, v in enumerate(vals) if v >= 0]
        neg_cols = _gradient_hex("#2E6EA6", "#A9C8E6", len(neg_idx))
        pos_cols = _gradient_hex("#F3AEA8", "#C8454B", len(pos_idx))
        colors = [None] * len(vals)
        for k, i in enumerate(neg_idx):
            colors[i] = neg_cols[k]
        for k, i in enumerate(pos_idx):
            colors[i] = pos_cols[k]
    else:
        colors = _gradient_hex("#D9E6F2", "#E47C6B", len(xdf))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(xdf["city"], xdf[value_col], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.35, alpha=0.25, zorder=0)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", labelrotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)
    if value_col == "annual_abs":
        ax.set_ylim(0, 3000)
    if value_col == "annual_net":
        ax.set_ylim(-3000, 2000)
    _save_figure(fig, out_path)


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = corr_df.columns.tolist()
    vals = corr_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(vals, cmap=_blue_to_red_cmap(), vmin=-1, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=14)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(False)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            text = f"{vals[i, j]:.2f}" if not math.isnan(vals[i, j]) else "nan"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    _save_figure(fig, out_path)


def top_correlations(df: pd.DataFrame, load_col: str, weather_cols: List[str]) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for w in weather_cols:
        pearson = float(df[[load_col, w]].corr(method="pearson").iloc[0, 1])
        spearman = float(df[[load_col, w]].corr(method="spearman").iloc[0, 1])
        rows.append((w, pearson, spearman))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze city-wise building load and weather relationship.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of city load CSV files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Directory of city PKL files.")
    parser.add_argument("--weather-dir", type=Path, default=Path("weather"), help="Directory of city EPW files.")
    parser.add_argument("--idf-path", type=Path, default=Path("idf") / "10011.idf", help="IDF file path.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_output"), help="Output directory.")
    args = parser.parse_args()

    root = args.root.resolve()
    data_dir = (root / args.data_dir).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    weather_dir = (root / args.weather_dir).resolve()
    idf_path = (root / args.idf_path).resolve()
    out_dir = (root / args.out_dir).resolve()

    _set_plot_style()

    logger = setup_logger(out_dir)
    logger.info("Analysis started.")
    logger.info("root=%s", root)
    logger.info("data_dir=%s", data_dir)
    logger.info("dataset_dir=%s", dataset_dir)
    logger.info("weather_dir=%s", weather_dir)
    logger.info("idf_path=%s", idf_path)

    if not data_dir.exists() or not dataset_dir.exists() or not weather_dir.exists() or not idf_path.exists():
        raise FileNotFoundError("One or more required inputs are missing.")

    idf_zone_count, idf_zone_names = parse_idf_zone_count(idf_path)
    logger.info("IDF zone count=%d", idf_zone_count)

    data_files = sorted(data_dir.glob("*.csv"))
    pkl_files = sorted(dataset_dir.glob("*.pkl"))
    weather_files = sorted(weather_dir.glob("*.epw"))

    if not data_files:
        raise RuntimeError("No CSV files found in data directory.")

    pkl_city_map: Dict[str, Path] = {}
    for p in pkl_files:
        _, c = parse_city_from_pkl_file(p)
        pkl_city_map[c.lower()] = p

    weather_city_map: Dict[str, Path] = {p.stem.lower(): p for p in weather_files}

    rows = []
    mismatch_notes = []

    for csv_path in data_files:
        bld_id, city = parse_city_from_data_file(csv_path)
        city_key = city.lower()

        logger.info("Processing city=%s", city)

        load_metrics = load_city_csv_metrics(csv_path)

        pkl_path = pkl_city_map.get(city_key)
        if pkl_path is None:
            raise FileNotFoundError(f"Missing PKL for city: {city}")
        pkl_space_count = load_pkl_space_count(pkl_path)

        epw_path = weather_city_map.get(city_key)
        if epw_path is None:
            raise FileNotFoundError(f"Missing EPW for city: {city}")
        weather_metrics = parse_epw_metrics(epw_path)

        zone_count_csv = int(load_metrics["zone_count_csv"])
        hour_count = int(load_metrics["hours"])

        note = []
        if hour_count != 8760:
            note.append(f"CSV hours={hour_count} (expected 8760)")
        if int(weather_metrics["weather_hours"]) != 8760:
            note.append(f"EPW hours={int(weather_metrics['weather_hours'])} (expected 8760)")
        if zone_count_csv != pkl_space_count:
            note.append(f"CSV zone_count={zone_count_csv} != PKL valid_energy_spaces={pkl_space_count}")
        if idf_zone_count > 0 and zone_count_csv != idf_zone_count:
            note.append(f"CSV zone_count={zone_count_csv} != IDF zone_count={idf_zone_count}")

        note_text = " | ".join(note)
        if note_text:
            mismatch_notes.append((city, note_text))
            logger.warning("%s: %s", city, note_text)

        row = {
            "building_id": bld_id,
            "city": city,
            "zone_count_csv": zone_count_csv,
            "zone_count_pkl": pkl_space_count,
            "zone_count_idf": idf_zone_count,
            "hours_csv": hour_count,
        }
        row.update(load_metrics)
        row.update(weather_metrics)
        row["quality_note"] = note_text
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)

    # Save raw summary.
    summary_csv = out_dir / "city_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved summary: %s", summary_csv)

    load_cols = ["annual_net", "annual_positive", "annual_negative", "annual_abs"]
    weather_cols = ["db_mean", "db_std", "rh_mean", "ghr_sum", "dnr_sum", "dhr_sum", "cdd18", "hdd18"]

    pearson_corr = summary[load_cols + weather_cols].corr(method="pearson")
    spearman_corr = summary[load_cols + weather_cols].corr(method="spearman")

    pearson_csv = out_dir / "correlation_pearson.csv"
    spearman_csv = out_dir / "correlation_spearman.csv"
    pearson_corr.to_csv(pearson_csv, encoding="utf-8-sig")
    spearman_corr.to_csv(spearman_csv, encoding="utf-8-sig")
    logger.info("Saved correlations: %s, %s", pearson_csv, spearman_csv)

    # Plots.
    plot_bar_sorted(
        summary,
        value_col="annual_net",
        title="Annual Net Load by City (sum of all zones over 8760h)",
        y_label="Annual Net Load (sum of zone unit-area loads)",
        out_path=out_dir / "annual_net_bar.png",
    )

    plot_bar_sorted(
        summary,
        value_col="annual_abs",
        title="Annual Absolute Load by City",
        y_label="Annual Absolute Load",
        out_path=out_dir / "annual_abs_bar.png",
    )

    scatter_with_fit(
        summary,
        x_col="cdd18",
        y_col="annual_positive",
        x_label="Cooling Degree Hours (base 18C)",
        y_label="Annual Positive Load",
        title="City Weather vs Annual Positive Load",
        out_path=out_dir / "scatter_cdd18_vs_positive.png",
        xlim=(0, 100000),
        ylim=(0, 1800),
    )

    summary = summary.copy()
    summary["annual_heating_magnitude"] = -summary["annual_negative"]
    scatter_with_fit(
        summary,
        x_col="hdd18",
        y_col="annual_heating_magnitude",
        x_label="Heating Degree Hours (base 18C)",
        y_label="Annual Heating Magnitude (-annual_negative)",
        title="City Weather vs Annual Heating Magnitude",
        out_path=out_dir / "scatter_hdd18_vs_heating.png",
        xlim=(0, 250000),
        ylim=(0, 3000),
    )

    scatter_with_fit(
        summary,
        x_col="db_mean",
        y_col="annual_net",
        x_label="Annual Mean Dry Bulb Temperature (C)",
        y_label="Annual Net Load",
        title="Annual Mean DB vs Annual Net Load",
        out_path=out_dir / "scatter_dbmean_vs_net.png",
        xlim=(-10, 30),
        ylim=(-3000, 2000),
    )

    plot_correlation_heatmap(
        pearson_corr,
        title="Pearson Correlation Matrix (Load + Weather Metrics)",
        out_path=out_dir / "heatmap_load_weather_pearson.png",
    )

    # Text report with key evidence.
    report_path = out_dir / "analysis_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Building Load vs Weather Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Cities analyzed: {len(summary)}\n")
        f.write(f"IDF zone count: {idf_zone_count}\n")
        f.write(f"Zone names in IDF (first 20): {idf_zone_names[:20]}\n\n")

        if mismatch_notes:
            f.write("Data quality notes:\n")
            for city, note in mismatch_notes:
                f.write(f"- {city}: {note}\n")
        else:
            f.write("Data quality checks passed for all cities (8760h and zone count consistency).\n")

        f.write("\nTop Pearson correlations by load metric:\n")
        for load_col in load_cols:
            f.write(f"\n[{load_col}]\n")
            top = top_correlations(summary, load_col, weather_cols)[:5]
            for w, p, s in top:
                f.write(f"- {w}: pearson={p:.4f}, spearman={s:.4f}\n")

        f.write("\nMethod notes:\n")
        f.write("- Annual building load is computed as the sum of all zone columns over 8760h.\n")
        f.write("- Zone-level values are treated as provided (unit-area load per user description).\n")
        f.write("- Weather indicators are extracted from EPW hourly fields: DB, RH, GHR, DNR, DHR.\n")
        f.write("- Additional climate indicators use base-18C degree-hours: CDD18 and HDD18.\n")

    logger.info("Saved report: %s", report_path)
    logger.info("Analysis completed.")


if __name__ == "__main__":
    main()
