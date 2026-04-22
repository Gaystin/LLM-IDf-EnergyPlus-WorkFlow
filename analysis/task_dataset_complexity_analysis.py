import argparse
import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Avoid matplotlib tkinter crashes in multi-threaded/batch environments on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


LEVEL_ORDER = ["Low", "Medium", "High", "Top"]
LEVEL_TO_DIR = {
    "Low": "Task-Low",
    "Medium": "Task-Medium",
    "High": "Task-High",
    "Top": "Task-Top",
}

# For architecture-aware complexity grouping.
HVAC_KEYWORDS = (
    "AIRLOOP",
    "PLANTLOOP",
    "BRANCH",
    "CONNECTOR",
    "NODE",
    "FAN",
    "COIL",
    "BOILER",
    "CHILLER",
    "COOLINGTOWER",
    "PUMP",
    "HEATEXCHANGER",
    "SETPOINTMANAGER",
    "CONTROLLER",
    "HVACTEMPLATE",
    "UNITARY",
    "TERMINAL",
)

ENVELOPE_KEYWORDS = (
    "MATERIAL",
    "CONSTRUCTION",
    "BUILDINGSURFACE",
    "FENESTRATIONSURFACE",
    "WINDOW",
    "DOOR",
    "SHADING",
    "ROOF",
    "WALL",
    "FLOOR",
)

CONTROL_KEYWORDS = (
    "THERMOSTAT",
    "SETPOINT",
    "CONTROLLER",
    "SCHEDULETYPELIMITS",
    "AVAILABILITYMANAGER",
    "ZONECONTROL",
)

SIZING_KEYWORDS = (
    "SIZING:",
    "DESIGNSPECIFICATION",
)


@dataclass
class ParsedIDF:
    total_lines: int
    blank_lines: int
    comment_lines: int
    code_lines: int
    object_counter: Counter
    field_slots_counter: Counter
    non_empty_fields_counter: Counter
    string_like_fields: int
    numeric_like_fields: int
    long_fields: int


class DatasetComplexityAnalyzer:
    def __init__(self, task_root: Path, out_root: Path, top_families: int = 12):
        self.task_root = task_root
        self.out_root = out_root
        self.top_families = top_families

        self.fig_dir = self.out_root / "figures"
        self.data_dir = self.out_root / "data"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._init_logger()

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"task_complexity_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Prevent duplicate handlers in repeated runs within the same process.
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(self.out_root / "analysis.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    @staticmethod
    def _strip_inline_comment(line: str) -> str:
        return line.split("!", 1)[0].strip()

    @staticmethod
    def _is_numeric_text(text: str) -> bool:
        if text is None:
            return False
        t = str(text).strip()
        if not t:
            return False
        # Remove common scientific notation and signs.
        try:
            float(t)
            return True
        except ValueError:
            return False

    def parse_idf(self, path: Path) -> ParsedIDF:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        total_lines = len(lines)
        blank_lines = 0
        comment_lines = 0
        code_lines = 0

        raw_blocks: List[str] = []
        buffer: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
                continue
            if stripped.startswith("!"):
                comment_lines += 1
                continue

            cleaned = self._strip_inline_comment(line)
            if not cleaned:
                comment_lines += 1
                continue

            code_lines += 1
            buffer.append(cleaned)
            if ";" in cleaned:
                raw_blocks.append(" ".join(buffer))
                buffer = []

        object_counter: Counter = Counter()
        field_slots_counter: Counter = Counter()
        non_empty_fields_counter: Counter = Counter()

        string_like_fields = 0
        numeric_like_fields = 0
        long_fields = 0

        for raw in raw_blocks:
            block = raw.split(";", 1)[0].strip()
            if not block:
                continue
            parts = [p.strip() for p in block.split(",")]
            if not parts:
                continue

            object_type = parts[0].upper()
            if not object_type:
                continue

            fields = parts[1:]
            field_slots = len(fields)
            non_empty_fields = [f for f in fields if f != ""]

            object_counter[object_type] += 1
            field_slots_counter[object_type] += field_slots
            non_empty_fields_counter[object_type] += len(non_empty_fields)

            for field in non_empty_fields:
                if len(field) >= 24:
                    long_fields += 1
                if self._is_numeric_text(field):
                    numeric_like_fields += 1
                else:
                    string_like_fields += 1

        return ParsedIDF(
            total_lines=total_lines,
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            code_lines=code_lines,
            object_counter=object_counter,
            field_slots_counter=field_slots_counter,
            non_empty_fields_counter=non_empty_fields_counter,
            string_like_fields=string_like_fields,
            numeric_like_fields=numeric_like_fields,
            long_fields=long_fields,
        )

    @staticmethod
    def _entropy_from_counter(counter: Counter) -> float:
        total = sum(counter.values())
        if total <= 0:
            return 0.0
        probs = np.array([v / total for v in counter.values()], dtype=float)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _count_keyword_objects(counter: Counter, keywords: Iterable[str]) -> int:
        total = 0
        for obj_type, cnt in counter.items():
            if any(k in obj_type for k in keywords):
                total += cnt
        return int(total)

    @staticmethod
    def _object_family_counter(counter: Counter) -> Counter:
        fam = Counter()
        for obj_type, cnt in counter.items():
            family = obj_type.split(":", 1)[0].strip() if ":" in obj_type else obj_type.strip()
            fam[family] += cnt
        return fam

    def compute_file_metrics(self, file_path: Path, level: str) -> Dict[str, float]:
        parsed = self.parse_idf(file_path)
        object_counter = parsed.object_counter
        field_slots_counter = parsed.field_slots_counter
        non_empty_fields_counter = parsed.non_empty_fields_counter

        total_instances = int(sum(object_counter.values()))
        unique_types = int(len(object_counter))

        total_field_slots = int(sum(field_slots_counter.values()))
        total_non_empty_fields = int(sum(non_empty_fields_counter.values()))

        zone_count = int(object_counter.get("ZONE", 0))
        people_count = int(object_counter.get("PEOPLE", 0))
        lights_count = int(object_counter.get("LIGHTS", 0))
        electric_equip_count = int(object_counter.get("ELECTRICEQUIPMENT", 0))
        surface_count = int(
            object_counter.get("BUILDINGSURFACE:DETAILED", 0)
            + object_counter.get("FENESTRATIONSURFACE:DETAILED", 0)
            + object_counter.get("INTERNALMASS", 0)
        )

        schedule_count = self._count_keyword_objects(object_counter, ("SCHEDULE",))
        hvac_count = self._count_keyword_objects(object_counter, HVAC_KEYWORDS)
        envelope_count = self._count_keyword_objects(object_counter, ENVELOPE_KEYWORDS)
        control_count = self._count_keyword_objects(object_counter, CONTROL_KEYWORDS)
        sizing_count = self._count_keyword_objects(object_counter, SIZING_KEYWORDS)
        output_req_count = self._count_keyword_objects(object_counter, ("OUTPUT:",))

        max_type_instance_count = int(max(object_counter.values())) if object_counter else 0
        entropy = self._entropy_from_counter(object_counter)

        avg_fields_per_object = float(total_field_slots / total_instances) if total_instances > 0 else 0.0
        non_empty_ratio = float(total_non_empty_fields / total_field_slots) if total_field_slots > 0 else 0.0

        total_typed_fields = parsed.numeric_like_fields + parsed.string_like_fields
        numeric_field_ratio = (
            float(parsed.numeric_like_fields / total_typed_fields) if total_typed_fields > 0 else 0.0
        )
        string_field_ratio = (
            float(parsed.string_like_fields / total_typed_fields) if total_typed_fields > 0 else 0.0
        )
        long_field_ratio = (
            float(parsed.long_fields / total_non_empty_fields) if total_non_empty_fields > 0 else 0.0
        )

        hvac_object_ratio = float(hvac_count / total_instances) if total_instances > 0 else 0.0
        envelope_object_ratio = float(envelope_count / total_instances) if total_instances > 0 else 0.0
        control_object_ratio = float(control_count / total_instances) if total_instances > 0 else 0.0

        comment_density = float(parsed.comment_lines / parsed.total_lines) if parsed.total_lines > 0 else 0.0
        code_density = float(parsed.code_lines / parsed.total_lines) if parsed.total_lines > 0 else 0.0

        return {
            "level": level,
            "file": file_path.name,
            "path": str(file_path),
            "file_size_kb": file_path.stat().st_size / 1024.0,
            "total_lines": parsed.total_lines,
            "blank_lines": parsed.blank_lines,
            "comment_lines": parsed.comment_lines,
            "code_lines": parsed.code_lines,
            "comment_density": comment_density,
            "code_density": code_density,
            "object_instances": total_instances,
            "unique_object_types": unique_types,
            "max_object_type_count": max_type_instance_count,
            "object_type_entropy": entropy,
            "total_field_slots": total_field_slots,
            "total_non_empty_fields": total_non_empty_fields,
            "avg_fields_per_object": avg_fields_per_object,
            "non_empty_field_ratio": non_empty_ratio,
            "numeric_field_ratio": numeric_field_ratio,
            "string_field_ratio": string_field_ratio,
            "long_field_ratio": long_field_ratio,
            "zone_count": zone_count,
            "people_count": people_count,
            "lights_count": lights_count,
            "electric_equipment_count": electric_equip_count,
            "surface_count": surface_count,
            "schedule_object_count": schedule_count,
            "hvac_object_count": hvac_count,
            "envelope_object_count": envelope_count,
            "control_object_count": control_count,
            "sizing_object_count": sizing_count,
            "output_request_count": output_req_count,
            "hvac_object_ratio": hvac_object_ratio,
            "envelope_object_ratio": envelope_object_ratio,
            "control_object_ratio": control_object_ratio,
        }

    def collect_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[Dict[str, float]] = []
        family_rows: List[Dict[str, float]] = []

        for level in LEVEL_ORDER:
            sub = LEVEL_TO_DIR[level]
            level_dir = self.task_root / sub
            if not level_dir.exists():
                self.logger.warning("Missing level directory: %s", level_dir)
                continue

            files = sorted(level_dir.glob("*.idf"))
            self.logger.info("Scanning level=%s | files=%d", level, len(files))

            for idx, idf_file in enumerate(files, start=1):
                metrics = self.compute_file_metrics(idf_file, level)
                rows.append(metrics)

                parsed = self.parse_idf(idf_file)
                fam_counter = self._object_family_counter(parsed.object_counter)
                total_instances = max(1, sum(parsed.object_counter.values()))
                for fam, cnt in fam_counter.items():
                    family_rows.append(
                        {
                            "level": level,
                            "file": idf_file.name,
                            "family": fam,
                            "count": cnt,
                            "ratio": cnt / total_instances,
                        }
                    )

                if idx % 100 == 0:
                    self.logger.info("Progress level=%s | parsed=%d/%d", level, idx, len(files))

        file_df = pd.DataFrame(rows)
        fam_df = pd.DataFrame(family_rows)

        if file_df.empty:
            raise RuntimeError(f"No IDF files parsed from {self.task_root}")

        return file_df, fam_df

    @staticmethod
    def _robust_zscore(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        median = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            std = s.std(ddof=0)
            if pd.isna(std) or std == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - s.mean()) / (std + 1e-9)
        return (s - median) / (iqr + 1e-9)

    def add_complexity_score(self, file_df: pd.DataFrame) -> pd.DataFrame:
        out = file_df.copy()
        metrics = [
            "code_lines",
            "object_instances",
            "unique_object_types",
            "total_field_slots",
            "zone_count",
            "surface_count",
            "hvac_object_count",
            "control_object_count",
            "schedule_object_count",
            "object_type_entropy",
            "avg_fields_per_object",
            "long_field_ratio",
        ]

        z_cols = []
        for m in metrics:
            z_name = f"z_{m}"
            out[z_name] = self._robust_zscore(out[m])
            z_cols.append(z_name)

        out["complexity_score_raw"] = out[z_cols].mean(axis=1)

        raw = out["complexity_score_raw"].to_numpy(dtype=float)
        p5 = np.nanpercentile(raw, 5)
        p95 = np.nanpercentile(raw, 95)
        if not np.isfinite(p5) or not np.isfinite(p95) or p95 <= p5:
            out["complexity_score"] = 50.0
        else:
            clipped = np.clip(raw, p5, p95)
            out["complexity_score"] = (clipped - p5) / (p95 - p5) * 100.0

        return out

    @staticmethod
    def _eta_squared(df: pd.DataFrame, metric: str, group_col: str = "level") -> float:
        x = pd.to_numeric(df[metric], errors="coerce")
        g = df[group_col]
        valid = x.notna() & g.notna()
        x = x[valid]
        g = g[valid]
        if len(x) <= 1:
            return float("nan")

        overall = x.mean()
        ss_total = float(((x - overall) ** 2).sum())
        if ss_total <= 0:
            return 0.0

        ss_between = 0.0
        for lv, grp in x.groupby(g):
            n = len(grp)
            if n == 0:
                continue
            ss_between += n * float((grp.mean() - overall) ** 2)
        return float(ss_between / ss_total)

    def build_level_summary(self, file_df: pd.DataFrame) -> pd.DataFrame:
        key_metrics = [
            "complexity_score",
            "code_lines",
            "object_instances",
            "unique_object_types",
            "total_field_slots",
            "zone_count",
            "surface_count",
            "hvac_object_count",
            "object_type_entropy",
            "non_empty_field_ratio",
            "numeric_field_ratio",
        ]

        rows = []
        for lv in LEVEL_ORDER:
            sub = file_df[file_df["level"] == lv]
            if sub.empty:
                continue
            row = {
                "level": lv,
                "file_count": len(sub),
            }
            for m in key_metrics:
                s = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(s.mean())
                row[f"{m}_median"] = float(s.median())
                row[f"{m}_std"] = float(s.std(ddof=0))
                row[f"{m}_q25"] = float(s.quantile(0.25))
                row[f"{m}_q75"] = float(s.quantile(0.75))
            rows.append(row)

        return pd.DataFrame(rows)

    def build_separation_table(self, file_df: pd.DataFrame) -> pd.DataFrame:
        candidate_metrics = [
            "complexity_score",
            "code_lines",
            "object_instances",
            "unique_object_types",
            "total_field_slots",
            "total_non_empty_fields",
            "avg_fields_per_object",
            "zone_count",
            "surface_count",
            "schedule_object_count",
            "hvac_object_count",
            "envelope_object_count",
            "control_object_count",
            "sizing_object_count",
            "output_request_count",
            "object_type_entropy",
            "max_object_type_count",
            "comment_density",
            "numeric_field_ratio",
            "string_field_ratio",
            "long_field_ratio",
        ]

        rows = []
        for m in candidate_metrics:
            eta2 = self._eta_squared(file_df, m, "level")
            medians = (
                file_df.groupby("level")[m]
                .median()
                .reindex(LEVEL_ORDER)
                .to_dict()
            )
            monotonic = (
                medians.get("Low", np.nan) <= medians.get("Medium", np.nan)
                <= medians.get("High", np.nan)
            )

            rows.append(
                {
                    "metric": m,
                    "eta_squared": eta2,
                    "low_median": medians.get("Low", np.nan),
                    "medium_median": medians.get("Medium", np.nan),
                    "high_median": medians.get("High", np.nan),
                    "is_monotonic_low_medium_high": bool(monotonic),
                }
            )

        sep_df = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
        return sep_df

    @staticmethod
    def _save_figure(fig: plt.Figure, out_file_stem: Path) -> None:
        fig.savefig(out_file_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.2)
        fig.savefig(out_file_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    @staticmethod
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

    @staticmethod
    def _blue_to_red_cmap() -> mcolors.Colormap:
        color_list = ["#5B84B1", "#D9E6F2", "#F4E6DE", "#E47C6B"]
        return mcolors.LinearSegmentedColormap.from_list("warm_blue_to_red", color_list, N=256)

    @staticmethod
    def _gradient_hex(start_hex: str, end_hex: str, n: int) -> List[tuple]:
        n = max(n, 1)
        s = np.array(mcolors.to_rgb(start_hex))
        e = np.array(mcolors.to_rgb(end_hex))
        vals = np.linspace(0.0, 1.0, n)
        return [tuple((s * (1.0 - v) + e * v).tolist()) for v in vals]

    @classmethod
    def _level_color_map(cls) -> Dict[str, tuple]:
        cols = cls._gradient_hex("#4C78A8", "#E97F89", len(LEVEL_ORDER))
        return {lv: cols[i] for i, lv in enumerate(LEVEL_ORDER)}

    def plot_complexity_distribution(self, file_df: pd.DataFrame) -> None:
        self._set_plot_style()
        fig, ax = plt.subplots(figsize=(11, 7))

        colors = self._level_color_map()
        x_positions = np.arange(1, len(LEVEL_ORDER) + 1)

        box_data = [file_df.loc[file_df["level"] == lv, "complexity_score"].dropna() for lv in LEVEL_ORDER]
        bp = ax.boxplot(
            box_data,
            tick_labels=LEVEL_ORDER,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "#111111", "linewidth": 1.6},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, lv in zip(bp["boxes"], LEVEL_ORDER):
            patch.set_facecolor(mcolors.to_rgba(colors[lv], alpha=0.45))
            patch.set_edgecolor(colors[lv])

        rng = np.random.default_rng(42)
        for i, lv in enumerate(LEVEL_ORDER):
            y = file_df.loc[file_df["level"] == lv, "complexity_score"].dropna().to_numpy()
            if y.size == 0:
                continue
            jitter = rng.normal(0, 0.06, size=y.size)
            ax.scatter(
                np.full_like(y, x_positions[i], dtype=float) + jitter,
                y,
                s=16,
                alpha=0.35,
                color=colors[lv],
                edgecolors="none",
            )

        ax.set_xlabel("Task Level", fontsize=12)
        ax.set_ylabel("Complexity Score (0-100)", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)

        self._save_figure(fig, self.fig_dir / "01_complexity_score_distribution")

    def plot_size_structure_scatter(self, file_df: pd.DataFrame) -> None:
        self._set_plot_style()
        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        color_map = self._level_color_map()

        for lv in LEVEL_ORDER:
            sub = file_df[file_df["level"] == lv]
            if sub.empty:
                continue
            sizes = 18 + np.sqrt(np.maximum(sub["zone_count"].to_numpy(dtype=float), 0.0)) * 18
            ax.scatter(
                sub["object_instances"],
                sub["unique_object_types"],
                s=sizes,
                alpha=0.45,
                color=color_map[lv],
                edgecolors="none",
                label=lv,
            )

        ax.set_xlabel("Total Object Instances", fontsize=12)
        ax.set_ylabel("Unique Object Types", fontsize=12)
        ax.legend(title="Level", frameon=False)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.grid(axis="x", visible=False)

        self._save_figure(fig, self.fig_dir / "02_size_vs_diversity_scatter")

    def plot_level_metric_heatmap(self, file_df: pd.DataFrame) -> None:
        self._set_plot_style()

        metrics = [
            "complexity_score",
            "code_lines",
            "object_instances",
            "unique_object_types",
            "total_field_slots",
            "zone_count",
            "surface_count",
            "schedule_object_count",
            "hvac_object_count",
            "control_object_count",
            "object_type_entropy",
            "non_empty_field_ratio",
            "numeric_field_ratio",
        ]

        grouped = (
            file_df.groupby("level")[metrics]
            .median()
            .reindex(LEVEL_ORDER)
        )

        # Normalize each metric to [0,1] for visual comparability.
        heat = grouped.copy()
        for m in metrics:
            col = grouped[m]
            mn, mx = col.min(), col.max()
            if pd.isna(mn) or pd.isna(mx) or mx <= mn:
                heat[m] = 0.5
            else:
                heat[m] = (col - mn) / (mx - mn)

        fig, ax = plt.subplots(figsize=(13.5, 5.8))
        im = ax.imshow(
            heat.to_numpy(dtype=float),
            aspect="auto",
            cmap=self._blue_to_red_cmap(),
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(LEVEL_ORDER)))
        ax.set_yticklabels(LEVEL_ORDER, fontsize=11)
        ax.grid(False)

        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                val = heat.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#111111")

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized median", fontsize=10)

        self._save_figure(fig, self.fig_dir / "03_level_metric_heatmap")

    def plot_family_composition(self, family_df: pd.DataFrame) -> None:
        self._set_plot_style()
        if family_df.empty:
            self.logger.warning("Skip family composition plot: empty family dataframe.")
            return

        top_fams = (
            family_df.groupby("family")["count"].sum()
            .sort_values(ascending=False)
            .head(self.top_families)
            .index
            .tolist()
        )

        sub = family_df[family_df["family"].isin(top_fams)].copy()
        pivot = (
            sub.groupby(["level", "family"])["ratio"]
            .mean()
            .unstack(fill_value=0.0)
            .reindex(index=LEVEL_ORDER)
        )

        fig, ax = plt.subplots(figsize=(12.5, 7.2))
        bottoms = np.zeros(len(pivot.index), dtype=float)

        palette = self._gradient_hex("#4C78A8", "#E97F89", max(1, len(pivot.columns)))
        for idx, fam in enumerate(pivot.columns):
            vals = pivot[fam].to_numpy(dtype=float)
            ax.bar(pivot.index, vals, bottom=bottoms, color=palette[idx], width=0.58, label=fam)
            bottoms += vals

        ax.set_ylabel("Mean family ratio in file", fontsize=12)
        ax.set_xlabel("Task Level", fontsize=12)
        ax.legend(
            title="Object Family",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=9,
            title_fontsize=10,
            frameon=False,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.grid(axis="x", visible=False)

        self._save_figure(fig, self.fig_dir / "04_family_composition_stacked")

    def plot_correlation_heatmap(self, file_df: pd.DataFrame) -> None:
        self._set_plot_style()
        metrics = [
            "complexity_score",
            "code_lines",
            "object_instances",
            "unique_object_types",
            "total_field_slots",
            "zone_count",
            "surface_count",
            "schedule_object_count",
            "hvac_object_count",
            "control_object_count",
            "object_type_entropy",
            "non_empty_field_ratio",
            "numeric_field_ratio",
            "long_field_ratio",
        ]

        corr = file_df[metrics].corr(method="spearman")
        mat = corr.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(12.5, 10.0))
        im = ax.imshow(mat, cmap=self._blue_to_red_cmap(), vmin=-1.0, vmax=1.0, aspect="equal")

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(metrics, fontsize=9)
        ax.grid(False)

        for i in range(len(metrics)):
            for j in range(len(metrics)):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="#111111")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spearman rho", fontsize=10)

        self._save_figure(fig, self.fig_dir / "05_metric_correlation_heatmap")

    def plot_level_file_counts(self, file_df: pd.DataFrame) -> None:
        self._set_plot_style()
        
        # Count files per level
        level_counts = file_df["level"].value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        color_map = self._level_color_map()
        colors = [color_map[lv] for lv in LEVEL_ORDER]
        
        bars = ax.bar(
            LEVEL_ORDER,
            level_counts.values,
            width=0.6,
            color=colors,
            alpha=0.45,
            edgecolor=[color_map[lv] for lv in LEVEL_ORDER],
            linewidth=1.5,
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
        
        ax.set_xlabel("Task Level", fontsize=16)
        ax.set_ylabel("Number of IDF Files", fontsize=16)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="x", visible=False)
        
        fig.tight_layout()
        output_png = self.fig_dir / "00_level_file_counts_bar"
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

    def write_report(
        self,
        file_df: pd.DataFrame,
        level_summary_df: pd.DataFrame,
        separation_df: pd.DataFrame,
    ) -> None:
        report_path = self.out_root / "dataset_report.md"

        top_sep = separation_df.head(10)
        n_files = len(file_df)
        per_level_counts = file_df["level"].value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int)

        lines = []
        lines.append("# Task-IDF 数据集复杂度分析报告")
        lines.append("")
        lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 数据目录: {self.task_root}")
        lines.append(f"- 样本总数: {n_files}")
        lines.append(f"- Low/Medium/High 样本数: {per_level_counts.to_dict()}")
        lines.append("")

        lines.append("## 1) 本次统计了哪些复杂度维度")
        lines.append("")
        lines.append("- 规模维度: code_lines, object_instances, total_field_slots, file_size_kb")
        lines.append("- 结构维度: unique_object_types, object_type_entropy, max_object_type_count")
        lines.append("- 建筑/负荷维度: zone_count, surface_count, people_count, lights_count")
        lines.append("- 系统维度: hvac_object_count, schedule_object_count, control_object_count, sizing_object_count")
        lines.append("- 字段复杂度: avg_fields_per_object, non_empty_field_ratio, numeric_field_ratio, long_field_ratio")
        lines.append("- 信息组织维度: comment_density, code_density")
        lines.append("")

        lines.append("## 2) 分级区分能力最强的指标（按 eta² 排序）")
        lines.append("")
        lines.append("| Rank | Metric | eta² | Low median | Medium median | High median | Monotonic (L<=M<=H) |")
        lines.append("|---:|---|---:|---:|---:|---:|:---:|")
        for i, row in enumerate(top_sep.itertuples(index=False), start=1):
            lines.append(
                f"| {i} | {row.metric} | {row.eta_squared:.4f} | {row.low_median:.3f} | {row.medium_median:.3f} | {row.high_median:.3f} | {'Y' if row.is_monotonic_low_medium_high else 'N'} |"
            )
        lines.append("")

        lines.append("## 3) 三个等级的核心统计")
        lines.append("")
        key_show = [
            "complexity_score_median",
            "code_lines_median",
            "object_instances_median",
            "unique_object_types_median",
            "zone_count_median",
            "hvac_object_count_median",
            "object_type_entropy_median",
            "non_empty_field_ratio_median",
            "numeric_field_ratio_median",
        ]

        show_df = level_summary_df[["level", "file_count", *key_show]].copy()
        lines.extend(self._df_to_markdown_table(show_df))
        lines.append("")

        lines.append("## 4) 复杂度总分构建方式")
        lines.append("")
        lines.append("复杂度总分由以下指标的 robust z-score 均值构成，并裁剪到 5%-95% 分位后映射到 0~100：")
        lines.append("")
        lines.append(
            "- code_lines, object_instances, unique_object_types, total_field_slots, zone_count, "
            "surface_count, hvac_object_count, control_object_count, schedule_object_count, "
            "object_type_entropy, avg_fields_per_object, long_field_ratio"
        )
        lines.append("")
        lines.append("## 5) 产出文件说明")
        lines.append("")
        lines.append("- data/file_metrics.csv: 每个 IDF 的全量指标")
        lines.append("- data/level_summary.csv: 分等级描述统计")
        lines.append("- data/metric_separation_eta2.csv: 指标区分度排序")
        lines.append("- figures/*.png / *.svg: 可视化图")
        lines.append("- analysis.log: 运行日志")

        report_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _df_to_markdown_table(df: pd.DataFrame) -> List[str]:
        if df.empty:
            return ["(空表)"]

        headers = [str(c) for c in df.columns]
        rows = []
        for row in df.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    if np.isnan(v):
                        vals.append("")
                    else:
                        vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            rows.append(vals)

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return lines

    def run(self) -> None:
        self.logger.info("Task complexity analysis started.")
        self.logger.info("task_root=%s", self.task_root)
        self.logger.info("out_root=%s", self.out_root)

        file_df, family_df = self.collect_dataset()
        self.logger.info("Parsed files: %d", len(file_df))

        file_df = self.add_complexity_score(file_df)
        level_summary_df = self.build_level_summary(file_df)
        separation_df = self.build_separation_table(file_df)

        file_df.to_csv(self.data_dir / "file_metrics.csv", index=False, encoding="utf-8-sig")
        family_df.to_csv(self.data_dir / "family_metrics.csv", index=False, encoding="utf-8-sig")
        level_summary_df.to_csv(self.data_dir / "level_summary.csv", index=False, encoding="utf-8-sig")
        separation_df.to_csv(self.data_dir / "metric_separation_eta2.csv", index=False, encoding="utf-8-sig")

        self.logger.info("Saved CSV outputs.")

        self.plot_complexity_distribution(file_df)
        self.plot_size_structure_scatter(file_df)
        self.plot_level_metric_heatmap(file_df)
        self.plot_family_composition(family_df)
        self.plot_correlation_heatmap(file_df)
        self.plot_level_file_counts(file_df)

        self.logger.info("Saved figures to %s", self.fig_dir)

        self.write_report(file_df, level_summary_df, separation_df)
        self.logger.info("Saved markdown report.")
        self.logger.info("Task complexity analysis finished.")


def build_output_dir(base_dir: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "log" / "task_complexity_analysis"
    return base_dir / ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Task-Low/Medium/High/Top IDF dataset complexity with logs, CSV summaries and plots."
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).parent.parent / "Task-new",
        help="Root directory containing Task-Low/Task-Medium/Task-High/Task-Top",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, auto-create under log/ with timestamp.",
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=12,
        help="Top N object families used in composition stacked bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else build_output_dir()

    analyzer = DatasetComplexityAnalyzer(
        task_root=args.task_root,
        out_root=out_dir,
        top_families=max(3, int(args.top_families)),
    )
    analyzer.run()

    print("=" * 92)
    print("Task dataset complexity analysis completed.")
    print(f"Output directory: {out_dir.resolve()}")
    print("Main outputs:")
    print(f"- Log: {(out_dir / 'analysis.log').resolve()}")
    print(f"- Report: {(out_dir / 'dataset_report.md').resolve()}")
    print(f"- CSV: {(out_dir / 'data').resolve()}")
    print(f"- Figures: {(out_dir / 'figures').resolve()}")
    print("=" * 92)


if __name__ == "__main__":
    main()
