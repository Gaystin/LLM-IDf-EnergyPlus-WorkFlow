import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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


class TaskComplexityScorer:
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

        raw_blocks = []
        buffer = []

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

            parts = [p.strip() for p in block.split(",")]
            if not parts:
                continue

            object_type = parts[0].upper()
            if not object_type:
                continue

            fields = parts[1:]
            non_empty_fields = [f for f in fields if f != ""]

            object_counter[object_type] += 1
            field_slots_counter[object_type] += len(fields)
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

    def compute_file_metrics(self, file_path: Path, level: str) -> dict:
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


LEVEL_DIRS = ["Task-Low", "Task-Medium", "Task-High"]


def discover_task_idfs(task_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for level_dir in LEVEL_DIRS:
        folder = task_root / level_dir
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.idf")):
            files.append((p, level_dir))
    return files


def balanced_split_sizes(n: int) -> tuple[int, int, int]:
    base = n // 3
    rem = n % 3
    low_n = base + (1 if rem >= 1 else 0)
    med_n = base + (1 if rem >= 2 else 0)
    high_n = base
    return low_n, med_n, high_n


def ensure_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        for level_dir in LEVEL_DIRS:
            target = output_root / level_dir
            if target.exists():
                shutil.rmtree(target)

    output_root.mkdir(parents=True, exist_ok=True)
    for level_dir in LEVEL_DIRS:
        (output_root / level_dir).mkdir(parents=True, exist_ok=True)


def repartition(task_root: Path, output_root: Path, overwrite: bool = False) -> Path:
    idf_items = discover_task_idfs(task_root)
    if not idf_items:
        raise RuntimeError(f"No IDF files found under {task_root}")

    ensure_output_dirs(output_root, overwrite=overwrite)

    scorer = TaskComplexityScorer()

    rows = []
    for p, old_level in idf_items:
        m = scorer.compute_file_metrics(p, level="All")
        m["old_level"] = old_level
        rows.append(m)

    df = pd.DataFrame(rows)
    df = scorer.add_complexity_score(df)

    # Lowest score -> low complexity, highest score -> high complexity.
    df = df.sort_values(["complexity_score", "file"], ascending=[True, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    low_n, med_n, high_n = balanced_split_sizes(len(df))

    new_level = np.empty(len(df), dtype=object)
    new_level[:low_n] = "Task-Low"
    new_level[low_n:low_n + med_n] = "Task-Medium"
    new_level[low_n + med_n:] = "Task-High"
    df["new_level"] = new_level

    copied = 0
    for row in df.itertuples(index=False):
        src = Path(row.path)
        dst = output_root / row.new_level / src.name
        shutil.copy2(src, dst)
        copied += 1

    summary_cols = [
        "rank",
        "file",
        "old_level",
        "new_level",
        "complexity_score",
        "complexity_score_raw",
        "code_lines",
        "object_instances",
        "unique_object_types",
        "total_field_slots",
        "zone_count",
        "hvac_object_count",
        "surface_count",
        "object_type_entropy",
        "path",
    ]
    summary_df = df[summary_cols].copy()
    summary_csv = output_root / "repartition_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    count_by_level = summary_df["new_level"].value_counts().to_dict()
    score_stats = summary_df.groupby("new_level")["complexity_score"].agg(["min", "median", "max"])

    report = []
    report.append("Task IDF repartition by complexity score")
    report.append("=" * 72)
    report.append(f"Source task root: {task_root.resolve()}")
    report.append(f"Output root: {output_root.resolve()}")
    report.append(f"Total IDF files: {len(summary_df)}")
    report.append(f"Copied files: {copied}")
    report.append(f"Counts by new level: {count_by_level}")
    report.append("")
    report.append("Score range by new level (complexity_score)")
    report.append(score_stats.to_string())
    report.append("")
    report.append("Split rule:")
    report.append("- Sort by complexity_score ascending")
    report.append("- First third -> Task-Low")
    report.append("- Middle third -> Task-Medium")
    report.append("- Last third -> Task-High")

    report_path = output_root / "repartition_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    return summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repartition all IDF files in Task into low/medium/high by computed complexity score."
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).parent.parent / "Task",
        help="Original task root containing Task-Low/Task-Medium/Task-High",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent.parent / "Task_new",
        help="Output folder for new partitions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Task_new level folders if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = repartition(
        task_root=args.task_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print("=" * 88)
    print("Repartition completed.")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
