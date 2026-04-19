from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")


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

LEVEL_ORDER = ["Task-Low", "Task-Medium", "Task-High", "Task-Top"]
LEVEL_RULES = (
    ("Task-Low", 0.0, 25.0),
    ("Task-Medium", 25.0, 50.0),
    ("Task-High", 50.0, 75.0),
    ("Task-Top", 75.0, 100.000001),
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_task_root() -> Path:
    return _repo_root() / "Task-new"


def _default_result_root() -> Path:
    return _repo_root() / "Task-result"


def _default_output_root() -> Path:
    return _repo_root() / "Task-result-new"


def _strip_inline_comment(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _is_numeric_text(text: str) -> bool:
    if text is None:
        return False
    value = str(text).strip()
    if not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _count_keyword_objects(counter: Counter, keywords: Iterable[str]) -> int:
    total = 0
    for object_type, count in counter.items():
        if any(keyword in object_type for keyword in keywords):
            total += count
    return int(total)


class TaskComplexityScorer:
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

            cleaned = _strip_inline_comment(line)
            if not cleaned:
                comment_lines += 1
                continue

            code_lines += 1
            buffer.append(cleaned)
            if ";" in cleaned:
                raw_blocks.append(" ".join(buffer))
                buffer = []

        object_counter = Counter()
        field_slots_counter = Counter()
        non_empty_fields_counter = Counter()
        string_like_fields = 0
        numeric_like_fields = 0
        long_fields = 0

        for raw in raw_blocks:
            block = raw.split(";", 1)[0].strip()
            if not block:
                continue

            parts = [piece.strip() for piece in block.split(",")]
            if not parts:
                continue

            object_type = parts[0].upper()
            if not object_type:
                continue

            fields = parts[1:]
            non_empty_fields = [field for field in fields if field != ""]

            object_counter[object_type] += 1
            field_slots_counter[object_type] += len(fields)
            non_empty_fields_counter[object_type] += len(non_empty_fields)

            for field in non_empty_fields:
                if len(field) >= 24:
                    long_fields += 1
                if _is_numeric_text(field):
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
        probs = np.array([value / total for value in counter.values()], dtype=float)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def compute_file_metrics(self, file_path: Path) -> dict:
        parsed = self.parse_idf(file_path)
        object_counter = parsed.object_counter
        field_slots_counter = parsed.field_slots_counter
        non_empty_fields_counter = parsed.non_empty_fields_counter

        total_instances = int(sum(object_counter.values()))
        unique_types = int(len(object_counter))
        total_field_slots = int(sum(field_slots_counter.values()))
        total_non_empty_fields = int(sum(non_empty_fields_counter.values()))

        zone_count = int(object_counter.get("ZONE", 0))
        surface_count = int(
            object_counter.get("BUILDINGSURFACE:DETAILED", 0)
            + object_counter.get("FENESTRATIONSURFACE:DETAILED", 0)
            + object_counter.get("INTERNALMASS", 0)
        )

        schedule_count = _count_keyword_objects(object_counter, ("SCHEDULE",))
        hvac_count = _count_keyword_objects(object_counter, HVAC_KEYWORDS)
        control_count = _count_keyword_objects(object_counter, CONTROL_KEYWORDS)
        sizing_count = _count_keyword_objects(object_counter, SIZING_KEYWORDS)
        output_req_count = _count_keyword_objects(object_counter, ("OUTPUT:",))

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
        control_object_ratio = float(control_count / total_instances) if total_instances > 0 else 0.0

        comment_density = float(parsed.comment_lines / parsed.total_lines) if parsed.total_lines > 0 else 0.0
        code_density = float(parsed.code_lines / parsed.total_lines) if parsed.total_lines > 0 else 0.0

        return {
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
            "surface_count": surface_count,
            "schedule_object_count": schedule_count,
            "hvac_object_count": hvac_count,
            "control_object_count": control_count,
            "sizing_object_count": sizing_count,
            "output_request_count": output_req_count,
            "hvac_object_ratio": hvac_object_ratio,
            "control_object_ratio": control_object_ratio,
        }

    @staticmethod
    def _robust_zscore(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            std = values.std(ddof=0)
            if pd.isna(std) or std == 0:
                return pd.Series(np.zeros(len(values)), index=values.index)
            return (values - values.mean()) / (std + 1e-9)
        return (values - median) / (iqr + 1e-9)

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
        for metric in metrics:
            z_name = f"z_{metric}"
            out[z_name] = self._robust_zscore(out[metric])
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


def _assign_level_by_score(score: float) -> str:
    value = float(score)
    for level_name, lower, upper in LEVEL_RULES:
        if lower <= value < upper:
            return level_name
    return "Task-Top"


def _build_idf_lookup(task_root: Path) -> dict[str, tuple[Path, str]]:
    lookup: dict[str, tuple[Path, str]] = {}
    for level_dir in LEVEL_ORDER:
        level_root = task_root / level_dir
        if not level_root.exists():
            continue
        for idf_path in sorted(level_root.glob("*.idf")):
            lookup[idf_path.stem] = (idf_path, level_dir)
        for subdir in sorted([path for path in level_root.iterdir() if path.is_dir()], key=lambda path: path.name):
            for idf_path in sorted(subdir.glob("*.idf")):
                lookup[idf_path.stem] = (idf_path, level_dir)
    return lookup


def _discover_result_task_dirs(result_root: Path) -> dict[str, Path]:
    task_dirs: dict[str, Path] = {}
    for level_dir in ["Task-Low", "Task-Medium", "Task-High"]:
        level_root = result_root / level_dir
        if not level_root.exists():
            continue
        for task_dir in sorted([path for path in level_root.iterdir() if path.is_dir()], key=lambda path: path.name):
            task_dirs[task_dir.name] = task_dir
    return task_dirs


def _ensure_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for level_dir in LEVEL_ORDER:
        (output_root / level_dir).mkdir(parents=True, exist_ok=True)


def _copy_result_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def repartition_results(task_root: Path, result_root: Path, output_root: Path, overwrite: bool = False) -> pd.DataFrame:
    source_result_dirs = _discover_result_task_dirs(result_root)
    if not source_result_dirs:
        raise RuntimeError(f"No task result directories found under {result_root}")

    idf_lookup = _build_idf_lookup(task_root)
    if not idf_lookup:
        raise RuntimeError(f"No IDF files found under {task_root}")

    _ensure_output_dirs(output_root, overwrite=overwrite)

    scorer = TaskComplexityScorer()
    rows: list[dict] = []

    for task_id in sorted(source_result_dirs.keys(), key=lambda value: int(value) if str(value).isdigit() else str(value)):
        found = idf_lookup.get(task_id)
        if found is None:
            rows.append(
                {
                    "task_id": task_id,
                    "old_level": "",
                    "path": "",
                    "source_result_found": True,
                    "source_result_path": str(source_result_dirs[task_id]),
                    "copied": False,
                    "missing_idf": True,
                }
            )
            continue

        idf_path, old_level = found
        metrics = scorer.compute_file_metrics(idf_path)
        metrics["task_id"] = task_id
        metrics["old_level"] = old_level
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = scorer.add_complexity_score(df)
    df["new_level"] = df["complexity_score"].apply(_assign_level_by_score)

    for idx, row in df.iterrows():
        task_id = str(row["task_id"])
        src_task_dir = source_result_dirs.get(task_id)
        if src_task_dir is None:
            df.at[idx, "source_result_found"] = False
            df.at[idx, "source_result_path"] = ""
            df.at[idx, "copied"] = False
            continue

        if bool(row.get("missing_idf", False)):
            df.at[idx, "copied"] = False
            continue

        df.at[idx, "source_result_found"] = True
        df.at[idx, "source_result_path"] = str(src_task_dir)
        dst_task_dir = output_root / str(row["new_level"]) / task_id
        _copy_result_tree(src_task_dir, dst_task_dir, overwrite=overwrite)
        df.at[idx, "copied"] = True

    df = df.sort_values(["complexity_score", "task_id"], ascending=[True, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def build_report(df: pd.DataFrame, task_root: Path, result_root: Path, output_root: Path) -> str:
    lines: list[str] = []
    lines.append("Task-result reassignment report")
    lines.append("=" * 72)
    lines.append(f"Task root: {task_root.resolve()}")
    lines.append(f"Source result root: {result_root.resolve()}")
    lines.append(f"Output root: {output_root.resolve()}")
    lines.append(f"Total IDFs: {len(df)}")
    lines.append(f"Copied task result dirs: {int(df['copied'].sum()) if 'copied' in df.columns else 0}")
    lines.append("")
    lines.append("Counts by new level")
    counts = df["new_level"].value_counts().reindex(LEVEL_ORDER, fill_value=0)
    lines.append(counts.to_string())
    lines.append("")
    lines.append("Missing source results")
    if "source_result_found" in df.columns:
        mask = df["source_result_found"].astype("boolean").fillna(False)
        missing = df.loc[~mask, ["task_id", "old_level", "new_level", "path"]]
    else:
        missing = pd.DataFrame()
    lines.append(missing.to_string(index=False) if not missing.empty else "<none>")
    lines.append("")
    lines.append("Preview")
    preview_cols = [
        "rank",
        "task_id",
        "old_level",
        "new_level",
        "complexity_score",
        "source_result_found",
        "copied",
    ]
    lines.append(df.loc[:, [col for col in preview_cols if col in df.columns]].to_string(index=False))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassign Task-result directories to the new four-level complexity split."
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=_default_task_root(),
        help="Root directory with Task-Low/Task-Medium/Task-High/Task-Top IDFs.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=_default_result_root(),
        help="Source Task-result directory containing old run outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Destination directory for the reassigned task results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing folders in the output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = repartition_results(
        task_root=args.task_root,
        result_root=args.result_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )

    summary_csv = args.output_root / "complexity_debug_summary.csv"
    report_txt = args.output_root / "complexity_debug_report.txt"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(df.to_csv(index=False))
    report_text = build_report(df, args.task_root, args.result_root, args.output_root)
    with report_txt.open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    print("=" * 88)
    print("Complexity reassignment completed.")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Report TXT: {report_txt.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()