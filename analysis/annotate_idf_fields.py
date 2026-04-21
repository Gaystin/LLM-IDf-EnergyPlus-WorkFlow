#!/usr/bin/env python3
"""Annotate IDF fields with names from an EnergyPlus IDD file.

Example:
    python analysis/annotate_idf_fields.py --idd "Energy+v8.9.idd" --idf "base.idf" --out "base_annotated.idf"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FIELD_DECL_RE = re.compile(r"^\s*([AN]\d+)\s*[,;]\s*(.*)$", re.IGNORECASE)
FIELD_NAME_RE = re.compile(r"\\field\s+(.+)$", re.IGNORECASE)
EXTENSIBLE_RE = re.compile(r"\\extensible\s*:\s*(\d+)", re.IGNORECASE)


@dataclass
class IddObject:
    name: str
    fields: List[str] = field(default_factory=list)
    begin_extensible_index: Optional[int] = None  # 1-based index
    extensible_group_size: Optional[int] = None


def _normalize_object_name(name: str) -> str:
    return re.sub(r"\s+", "", name.strip()).upper()


def parse_idd(idd_text: str) -> Dict[str, IddObject]:
    objects: Dict[str, IddObject] = {}
    current: Optional[IddObject] = None
    pending_field_pos: Optional[int] = None

    for raw in idd_text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped or stripped.startswith("!"):
            continue

        # Top-level object declaration line: "ObjectName,"
        if line == line.lstrip() and stripped.endswith(",") and "\\" not in stripped:
            obj_name = stripped[:-1].strip()
            if obj_name:
                current = IddObject(name=obj_name)
                objects[_normalize_object_name(obj_name)] = current
                pending_field_pos = None
            continue

        if current is None:
            continue

        ext_match = EXTENSIBLE_RE.search(stripped)
        if ext_match:
            current.extensible_group_size = int(ext_match.group(1))

        if "\\begin-extensible" in stripped.lower():
            # The marker refers to the next field in the object.
            current.begin_extensible_index = len(current.fields) + 1

        field_decl = FIELD_DECL_RE.match(line)
        if field_decl:
            rest = field_decl.group(2)
            current.fields.append("")
            field_pos = len(current.fields) - 1

            field_name_match = FIELD_NAME_RE.search(rest)
            if field_name_match:
                current.fields[field_pos] = field_name_match.group(1).strip()
                pending_field_pos = None
            else:
                pending_field_pos = field_pos
            continue

        # Some IDD definitions place \field on a following line.
        if pending_field_pos is not None:
            field_name_match = FIELD_NAME_RE.search(stripped)
            if field_name_match:
                current.fields[pending_field_pos] = field_name_match.group(1).strip()
                pending_field_pos = None

    return objects


def split_object_tokens(object_text: str) -> Tuple[List[str], List[str]]:
    """Split object text into tokens and delimiters.

    Returns:
      tokens: each token before delimiter (includes empty tokens for blank fields)
      delims: delimiter for each token in tokens (',' or ';')
    """
    tokens: List[str] = []
    delims: List[str] = []
    buf: List[str] = []

    for ch in object_text:
        if ch in ",;":
            tokens.append("".join(buf).strip())
            delims.append(ch)
            buf = []
        else:
            buf.append(ch)

    # trailing content after final delimiter is ignored for normal IDF objects
    trailing = "".join(buf).strip()
    if trailing:
        # Keep as a token without delimiter only when input is malformed.
        tokens.append(trailing)

    return tokens, delims


def get_field_name(obj: Optional[IddObject], field_index: int) -> str:
    if obj is None:
        return f"Field {field_index}"

    if field_index <= len(obj.fields) and obj.fields[field_index - 1]:
        return obj.fields[field_index - 1]

    if (
        obj.begin_extensible_index is not None
        and obj.extensible_group_size
        and obj.begin_extensible_index <= len(obj.fields)
    ):
        start = obj.begin_extensible_index - 1
        end = start + obj.extensible_group_size
        pattern = [f for f in obj.fields[start:end] if f]
        if pattern:
            return pattern[(field_index - obj.begin_extensible_index) % len(pattern)]

    return f"Field {field_index}"


def format_annotated_object(obj_name: str, fields: List[str], idd_obj: Optional[IddObject]) -> str:
    out: List[str] = [f"{obj_name},"]

    comment_col = 42
    for idx, value in enumerate(fields, start=1):
        is_last = idx == len(fields)
        sep = ";" if is_last else ","
        left = f"  {value}{sep}"
        if len(left) < comment_col:
            left = left.ljust(comment_col)
        field_name = get_field_name(idd_obj, idx)
        out.append(f"{left}!- {field_name}")

    return "\n".join(out)


def annotate_idf(idf_text: str, idd_objects: Dict[str, IddObject]) -> str:
    lines = idf_text.splitlines()
    out_lines: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Preserve blank and full-line comments as is.
        if not stripped or stripped.startswith("!"):
            out_lines.append(line)
            i += 1
            continue

        # Collect one full object until ';' (ignoring text after inline '!').
        raw_object_lines: List[str] = []
        while i < len(lines):
            current = lines[i]
            value_part = current.split("!", 1)[0]
            raw_object_lines.append(value_part)
            if ";" in value_part:
                i += 1
                break
            i += 1

        object_text = "\n".join(raw_object_lines)
        tokens, delims = split_object_tokens(object_text)

        # Minimal validity check: object must start with name and first delimiter must be comma.
        if not tokens or not delims or delims[0] != ",":
            out_lines.extend(raw_object_lines)
            continue

        obj_name = tokens[0]
        fields = tokens[1 : len(delims)]

        norm_name = _normalize_object_name(obj_name)
        idd_obj = idd_objects.get(norm_name)
        out_lines.append(format_annotated_object(obj_name, fields, idd_obj))
        out_lines.append("")

    return "\n".join(out_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add field-name comments to an IDF based on an IDD file.")
    parser.add_argument("--idd", required=True, help="Path to EnergyPlus IDD file")
    parser.add_argument("--idf", required=True, help="Path to source IDF file")
    parser.add_argument("--out", required=True, help="Path to output annotated IDF file")
    args = parser.parse_args()

    idd_path = Path(args.idd)
    idf_path = Path(args.idf)
    out_path = Path(args.out)

    idd_text = idd_path.read_text(encoding="utf-8", errors="ignore")
    idf_text = idf_path.read_text(encoding="utf-8", errors="ignore")

    idd_objects = parse_idd(idd_text)
    annotated = annotate_idf(idf_text, idd_objects)

    out_path.write_text(annotated, encoding="utf-8", newline="\n")

    print(f"Parsed IDD objects: {len(idd_objects)}")
    print(f"Annotated IDF written: {out_path}")


if __name__ == "__main__":
    main()
