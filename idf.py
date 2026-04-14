import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _strip_inline_comment(line: str) -> str:
    """Remove inline IDF comments started by '!' and trim spaces."""
    return line.split("!", 1)[0].strip()


def parse_idf_objects(idf_path: Path):
    """Parse IDF into object blocks and return object statistics."""
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    raw_blocks = []
    buffer = []

    for line in lines:
        cleaned = _strip_inline_comment(line)
        if not cleaned:
            continue

        buffer.append(cleaned)
        if ";" in cleaned:
            block = " ".join(buffer)
            raw_blocks.append(block)
            buffer = []

    object_counter = Counter()
    field_slots_counter = Counter()
    non_empty_fields_counter = Counter()
    per_object_field_slots = defaultdict(list)

    for raw in raw_blocks:
        block = raw.split(";", 1)[0].strip()
        if not block:
            continue

        parts = [p.strip() for p in block.split(",")]
        if not parts:
            continue

        object_type = parts[0]
        if not object_type:
            continue
        object_type = object_type.upper()

        fields = parts[1:]
        field_slots = len(fields)
        non_empty_fields = sum(1 for f in fields if f != "")

        object_counter[object_type] += 1
        field_slots_counter[object_type] += field_slots
        non_empty_fields_counter[object_type] += non_empty_fields
        per_object_field_slots[object_type].append(field_slots)

    return {
        "object_counter": object_counter,
        "field_slots_counter": field_slots_counter,
        "non_empty_fields_counter": non_empty_fields_counter,
        "per_object_field_slots": per_object_field_slots,
        "total_instances": sum(object_counter.values()),
        "total_unique_types": len(object_counter),
        "total_field_slots": sum(field_slots_counter.values()),
        "total_non_empty_fields": sum(non_empty_fields_counter.values()),
    }


def build_report(idf_path: Path, stats: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 88)
    lines.append("IDF Analysis Report")
    lines.append("=" * 88)
    lines.append(f"File: {idf_path}")
    lines.append(f"Generated: {now}")
    lines.append("")

    lines.append("[Summary]")
    lines.append(f"Total object instances: {stats['total_instances']}")
    lines.append(f"Total unique object types: {stats['total_unique_types']}")
    lines.append(f"Total field slots (all instances): {stats['total_field_slots']}")
    lines.append(f"Total non-empty fields (all instances): {stats['total_non_empty_fields']}")
    lines.append("")

    lines.append("[All object types and occurrence counts]")
    lines.append(
        "No. | Object Type                                  | Occurrences | "
        "Total Fields | Non-empty Fields | Avg/Obj | Min/Obj | Max/Obj"
    )
    lines.append("-" * 88)

    sorted_items = sorted(
        stats["object_counter"].items(),
        key=lambda kv: (-kv[1], kv[0].lower()),
    )

    for idx, (obj_type, count) in enumerate(sorted_items, start=1):
        total_fields = stats["field_slots_counter"][obj_type]
        non_empty_fields = stats["non_empty_fields_counter"][obj_type]
        field_counts = stats["per_object_field_slots"][obj_type]
        avg_fields = total_fields / count if count else 0.0
        min_fields = min(field_counts) if field_counts else 0
        max_fields = max(field_counts) if field_counts else 0
        lines.append(
            f"{idx:>3} | {obj_type:<44} | {count:>11} | "
            f"{total_fields:>12} | {non_empty_fields:>16} | {avg_fields:>7.2f} | "
            f"{min_fields:>7} | {max_fields:>7}"
        )

    lines.append("")
    lines.append("End of report")
    lines.append("=" * 88)
    return "\n".join(lines)


def analyze_idf(idf_file: str, log_dir: str = "log") -> Path:
    idf_path = Path(idf_file).resolve()
    if not idf_path.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")

    stats = parse_idf_objects(idf_path)
    report = build_report(idf_path, stats)

    output_dir = Path(log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"{idf_path.stem}.log"
    log_path.write_text(report, encoding="utf-8")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an IDF file and write object/field statistics into log folder."
    )
    parser.add_argument(
        "idf_file",
        nargs="?",
        default="Task/Task-High/10751.idf",
        help="Path to target IDF file. Default: Task/Task-High/10751.idf",
    )
    parser.add_argument(
        "--log-dir",
        default="log",
        help="Directory to save logs. Default: log",
    )
    args = parser.parse_args()

    log_path = analyze_idf(args.idf_file, args.log_dir)
    print(f"Analysis complete. Log saved to: {log_path}")


if __name__ == "__main__":
    main()