import html
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import EnergyPlusOptimizer


RUNTIME_DIR = os.path.join(CURRENT_DIR, ".web_runtime")
WORKFLOW_REGEX = re.compile(r"workflow_(\d+)", flags=re.IGNORECASE)
ROUND_REGEX = re.compile(r"【第(\d+)轮/(\d+)】")
REASONING_HEADER_REGEX = re.compile(r"【LLM推理过程\s*-\s*(.*?)】")
PERCENT_REGEX = re.compile(r"([+-]?\d+(?:\.\d+)?)%")
LOG_LINE_START_REGEX = re.compile(r"(?<!\n)(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s-\s[A-Z]+\s-\s)")

DEFAULT_IDD_PATH = "Energy+.idd"
DEFAULT_EPW_PATH = "weather.epw"
DEFAULT_LOG_DIR = "optimization_logs_并行2_web"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_EARLY_STOP_TARGET_SAVING_PCT = 60.0
DEFAULT_NUM_WORKFLOWS = 2
AUTO_REFRESH_SECONDS = 1.0
UI_ACTION_COOLDOWN_SECONDS = 3.0
WORKFLOW_PALETTE = [
    "#1F77B4", "#E67E22", "#2A9D8F", "#C8553D", "#7A5CFA",
    "#6C8A2B", "#D62728", "#17BECF", "#9467BD", "#8C564B",
    "#BCBD22", "#FF7F0E", "#4E79A7", "#F28E2B", "#59A14F",
    "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7",
]


def _real_to_display_iteration(real_iteration: int) -> int:
    return max(int(real_iteration or 0) - 1, 0)


def _real_to_display_max_iterations(real_max_iterations: int) -> int:
    return max(int(real_max_iterations or 0) - 1, 0)


def _build_default_capture_state() -> dict[str, Any]:
    return {
        "current_iteration": 0,
        "max_iterations": 5,
        "latest_reasoning": None,
        "latest_summary": None,
        "reasoning_history": [],
        "summary_history": [],
        "latest_plan_metrics": None,
        "plan_metrics_history": [],
        "latest_round_stats": None,
        "round_stats_history": [],
        "latest_parameter_details": None,
        "parameter_details_history": [],
        "baseline_log": None,
        "baseline_log_history": [],
        "early_stopped": False,
        "latest_status": "等待启动",
        "status_updated_at": None,
        "collecting_summary": False,
        "summary_lines": [],
        "collecting_parameter_details": False,
        "parameter_detail_lines": [],
        "collecting_baseline_log": False,
        "baseline_log_lines": [],
    }


def _build_global_capture_state() -> dict[str, Any]:
    return {
        "summary_log": None,
        "summary_log_lines": [],
        "updated_at": None,
        "collecting_summary_log": False,
    }


def _build_empty_snapshot(config: dict[str, Any] | None = None, status: str = "idle", error: str | None = None) -> dict[str, Any]:
    config = config or {}
    max_iterations = int(config.get("max_iterations", 5) or 5)
    optimization_rounds = int(config.get("optimization_rounds", _real_to_display_max_iterations(max_iterations)) or _real_to_display_max_iterations(max_iterations))
    max_iterations_cap = int(config.get("max_iterations_cap", optimization_rounds) or optimization_rounds)
    early_stop_target_total_saving_pct = float(
        config.get("early_stop_target_total_saving_pct", DEFAULT_EARLY_STOP_TARGET_SAVING_PCT)
        or DEFAULT_EARLY_STOP_TARGET_SAVING_PCT
    )
    num_workflows = int(config.get("num_workflows", 2) or 2)
    workflows = {
        f"workflow_{index + 1}": {
            "current_iteration": 0,
            "max_iterations": optimization_rounds,
            "progress": 0.0,
            "latest_reasoning": None,
            "latest_summary": None,
            "reasoning_history": [],
            "summary_history": [],
            "latest_plan_metrics": None,
            "plan_metrics_history": [],
            "latest_round_stats": None,
            "round_stats_history": [],
            "latest_parameter_details": None,
            "parameter_details_history": [],
            "baseline_log": None,
            "baseline_log_history": [],
            "early_stopped": False,
            "latest_status": "等待启动",
            "status_updated_at": None,
            "iteration_history": [],
            "best_metrics": None,
            "best_iteration": 0,
        }
        for index in range(num_workflows)
    }
    return {
        "status": status,
        "error": error,
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "config": {
            "idf_path": config.get("idf_path", "in.idf"),
            "idd_path": config.get("idd_path", DEFAULT_IDD_PATH),
            "api_key_path": config.get("api_key_path", "api_key.txt"),
            "epw_path": config.get("epw_path", DEFAULT_EPW_PATH),
            "log_dir": config.get("log_dir", DEFAULT_LOG_DIR),
            "max_iterations": max_iterations,
            "optimization_rounds": optimization_rounds,
            "max_iterations_cap": max_iterations_cap,
            "early_stop_target_total_saving_pct": early_stop_target_total_saving_pct,
            "num_workflows": num_workflows,
        },
        "global_summary_log": None,
        "workflows": workflows,
    }


def _write_snapshot(snapshot_path: str, snapshot: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    last_error = None

    for attempt in range(10):
        temp_path = f"{snapshot_path}.{os.getpid()}.{threading.get_ident()}.{attempt}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as file_obj:
                file_obj.write(payload)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, snapshot_path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            last_error = exc
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            time.sleep(0.03 * (attempt + 1))

    # 回退为非原子写，确保后台线程不因临时文件锁冲突崩溃。
    try:
        with open(snapshot_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(payload)
    except OSError:
        if last_error:
            raise last_error
        raise


def _read_snapshot(snapshot_path: str | None) -> dict[str, Any]:
    if not snapshot_path or not os.path.exists(snapshot_path):
        return _build_empty_snapshot()
    with open(snapshot_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _fallback_global_summary_log(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if snapshot.get("global_summary_log"):
        return snapshot.get("global_summary_log")

    # Do not read old log files when no run has been started yet.
    if str(snapshot.get("status", "idle") or "idle") == "idle":
        return None

    config = snapshot.get("config", {}) or {}
    log_dir = str(config.get("log_dir", DEFAULT_LOG_DIR) or DEFAULT_LOG_DIR)
    log_dir_path = os.path.join(CURRENT_DIR, log_dir)
    if not os.path.isdir(log_dir_path):
        return None

    candidates = [name for name in os.listdir(log_dir_path) if name.lower().endswith(".log")]
    if not candidates:
        return None

    latest_log = max(candidates, key=lambda name: os.path.getmtime(os.path.join(log_dir_path, name)))
    latest_log_path = os.path.join(log_dir_path, latest_log)

    try:
        with open(latest_log_path, "r", encoding="utf-8", errors="ignore") as log_file:
            lines = [line.rstrip("\n") for line in log_file]
    except OSError:
        return None

    if not lines:
        return None

    end_index = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if "【总】" in line and "【并行优化循环完成】" in line:
            end_index = index
            break

    if end_index is None:
        return None

    start_index = None
    for index in range(end_index, -1, -1):
        line = lines[index]
        if "【总】" in line and "【Token使用统计】" in line:
            start_index = index
            break

    if start_index is None:
        for index in range(end_index, -1, -1):
            line = lines[index]
            if "【总】" in line and "【全局耗时统计】" in line:
                start_index = index
                break

    if start_index is None:
        for index in range(end_index, -1, -1):
            line = lines[index]
            if "【总】" in line and ("并行流程总耗时" in line or "总Token消耗" in line):
                start_index = index
                break

    if start_index is None:
        return None

    block_lines = lines[start_index : end_index + 1]
    if not any("总Token消耗" in line for line in block_lines):
        return None

    updated_at = datetime.now().strftime("%H:%M:%S")
    for line in reversed(block_lines):
        timestamp_match = re.match(r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
        if timestamp_match:
            updated_at = timestamp_match.group(1).split(" ", 1)[1]
            break

    return {
        "updated_at": updated_at,
        "text": _normalize_timestamp_log_lines("\n".join(block_lines)),
    }


def _clean_summary_lines(lines: list[str]) -> str:
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        raw_for_check = _strip_log_prefix(stripped)
        if raw_for_check.startswith("=") or raw_for_check.startswith("-"):
            continue
        if raw_for_check == "【修改摘要】汇总 - 快速查看修改概览":
            continue
        if raw_for_check.startswith("[计划修改]") or raw_for_check.startswith("[文本替换"):
            continue
        if "警告" in raw_for_check or raw_for_check.startswith("⚠"):
            continue
        if not raw_for_check:
            if cleaned and not prev_blank:
                cleaned.append("")
                prev_blank = True
            continue

        if raw_for_check.startswith("▶") and cleaned and not prev_blank:
            cleaned.append("")

        cleaned.append(stripped)
        prev_blank = False

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned)


def _format_reasoning_lines(text: str) -> str:
    normalized_text = str(text or "")
    # Some responses contain escaped newlines as literal \n characters.
    normalized_text = normalized_text.replace("\\r\\n", "\n").replace("\\n", "\n")
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return ""

    marker_matches = list(re.finditer(r"(?<!\d)(\d+[\.、\)）])", normalized_text))
    if len(marker_matches) >= 2:
        segmented_items = []
        for index, match in enumerate(marker_matches):
            start = match.start()
            end = marker_matches[index + 1].start() if index + 1 < len(marker_matches) else len(normalized_text)
            segment = normalized_text[start:end].strip()
            segment = re.sub(r"\s+", " ", segment)
            if segment:
                segmented_items.append(segment)
        if segmented_items:
            return "\n".join(segmented_items)

    raw_lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    items = []
    for line in raw_lines:
        if re.match(r"^\d+[\.、\)）]\s*", line):
            line = re.sub(r"\s+", " ", line)
            items.append(line)
            continue
        content = re.sub(r"^\d+[\.、\)）]\s*", "", line).strip()
        if content:
            items.append(f"{len(items) + 1}. {content}")

    return "\n".join(items)


def _upsert_iteration_payload(history: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    iteration = int(payload.get("iteration", 0) or 0)
    for index, item in enumerate(history):
        if int(item.get("iteration", 0) or 0) == iteration:
            history[index] = payload
            return
    history.append(payload)
    history.sort(key=lambda item: int(item.get("iteration", 0) or 0))


def _short_status(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return "等待日志输出"
    first_line = text.splitlines()[0].strip()
    if len(first_line) > 72:
        return f"{first_line[:72]}..."
    return first_line


def _strip_log_prefix(line: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s-\s[A-Z]+\s-\s", "", str(line or "")).strip()


def _normalize_timestamp_log_lines(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = LOG_LINE_START_REGEX.sub(r"\n\1", normalized)
    return normalized.lstrip("\n")

def _split_timestamp_log_lines(text: str) -> list[str]:
    return _normalize_timestamp_log_lines(text).split("\n")


def _format_record_lines(record: logging.LogRecord, message: str) -> list[str]:
    ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
    level = str(record.levelname or "INFO").upper()
    lines = str(message or "").splitlines() or [str(message or "")]
    return [f"{ts} - {level} - {line}" if line else f"{ts} - {level} - " for line in lines]


def _clean_parameter_detail_lines(lines: list[str]) -> str:
    cleaned = []
    prev_blank = False
    for line in lines:
        raw = _strip_log_prefix(line)
        if not raw:
            if cleaned and not prev_blank:
                cleaned.append("")
                prev_blank = True
            continue
        if raw.startswith("【计时】") or raw.startswith("【最终汇总报告") or raw.startswith("全局最佳"):
            continue
        cleaned.append(line.strip())
        prev_blank = False

    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return _normalize_timestamp_log_lines("\n".join(cleaned))


def _prepare_parameter_details_for_view(text: str, keep_total_summary_block: bool) -> str:
    filtered: list[str] = []
    for line in _split_timestamp_log_lines(text):
        raw = _strip_log_prefix(line)
        if raw == "":
            filtered.append("")
            continue

        is_total_block_line = (
            raw.startswith("【总】")
            or raw.startswith("【各工作流最优能耗对比】")
            or raw.startswith("────────────────")
        )
        is_global_plain_line = (
            bool(re.match(r"^workflow_\d+\s+\d+\s+", raw))
            or "【🏆 全局最优方案】" in raw
            or raw.startswith("最优IDF文件:")
        )

        if keep_total_summary_block:
            # 汇总页保留【总】块，去掉未标记【总】的全局摘要噪声行。
            if is_global_plain_line and not raw.startswith("【总】"):
                continue
        else:
            # 工作流页去掉总览块，仅保留对象修改详情与最终曲线保存行。
            if is_total_block_line or is_global_plain_line:
                continue

        filtered.append(line)

    while filtered and filtered[-1] == "":
        filtered.pop()
    return _normalize_timestamp_log_lines("\n".join(filtered))


def _extract_parallel_curve_saved_line(summary_log_payload: dict[str, Any] | None) -> str | None:
    if not summary_log_payload or not summary_log_payload.get("text"):
        return None
    for line in reversed(_split_timestamp_log_lines(str(summary_log_payload.get("text", "")))):
        raw = _strip_log_prefix(line)
        if "【总】" in raw and "已保存并行汇总能耗曲线:" in raw:
            return line
    return None


def _replace_last_curve_saved_line(text: str, replacement_line: str | None) -> str:
    lines = _split_timestamp_log_lines(text)
    filtered = [line for line in lines if "已保存单工作流能耗曲线:" not in _strip_log_prefix(line)]
    while filtered and filtered[-1] == "":
        filtered.pop()
    if replacement_line and replacement_line.strip():
        filtered.append(replacement_line)
    return _normalize_timestamp_log_lines("\n".join(filtered))


@st.cache_data(show_spinner=False, max_entries=256)
def _build_timestamp_log_html(text: str) -> str:
    line_html = []
    for line in _split_timestamp_log_lines(text):
        if line == "":
            line_html.append("<div class='mono-log-line mono-log-line-blank'>&nbsp;</div>")
        else:
            line_html.append(f"<div class='mono-log-line'>{html.escape(line)}</div>")
    return "".join(line_html)


def _render_timestamp_log_block(text: str) -> None:
    body_html = _build_timestamp_log_html(str(text or ""))
    st.markdown(
        f"<div class='glass scroll-block mono-log-block'>{body_html}</div>",
        unsafe_allow_html=True,
    )


def _extract_int_after_label(text: str, label: str) -> int | None:
    pattern = re.escape(label) + r"\s*(\d+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))
    return _normalize_timestamp_log_lines("\n".join(cleaned))

def _parse_plan_metrics_line(message: str) -> dict[str, Any]:
    unique_fields = _extract_int_after_label(message, "字段类别")
    object_categories = _extract_int_after_label(message, "对象类别:")
    overlap_match = re.search(r"\((\d+)/(\d+)\)", message)
    overlap_count = None
    overlap_total = None
    if overlap_match:
        overlap_count = int(overlap_match.group(1))
        overlap_total = int(overlap_match.group(2))

    similarity_pct = None
    percent_match = PERCENT_REGEX.search(message)
    if percent_match:
        similarity_pct = float(percent_match.group(1))

    return {
        "unique_fields": unique_fields,
        "object_categories": object_categories,
        "similarity_pct": similarity_pct,
        "overlap_count": overlap_count,
        "overlap_total": overlap_total,
        "line": message.strip(),
    }


def _simplify_iteration_history(iteration_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in iteration_history or []:
        metrics = item.get("metrics", {}) or {}
        rows.append(
            {
                "iteration": _real_to_display_iteration(int(item.get("iteration", 0) or 0)),
                "metrics": {
                    "total_site_energy_kwh": float(metrics.get("total_site_energy_kwh", 0) or 0),
                    "eui_kwh_per_m2": float(metrics.get("eui_kwh_per_m2", 0) or 0),
                    "total_cooling_kwh": float(metrics.get("total_cooling_kwh", 0) or 0),
                    "total_heating_kwh": float(metrics.get("total_heating_kwh", 0) or 0),
                },
                "idf_path": item.get("idf_path"),
                "plan_description": item.get("plan_description"),
            }
        )
    return rows


def _compose_snapshot(
    optimizer: EnergyPlusOptimizer | None,
    capture_state: dict[str, Any],
    config: dict[str, Any],
    status: str,
    started_at: str | None,
    finished_at: str | None,
    error: str | None,
) -> dict[str, Any]:
    snapshot = _build_empty_snapshot(config=config, status=status, error=error)
    snapshot["started_at"] = started_at
    snapshot["finished_at"] = finished_at
    snapshot["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not optimizer:
        return snapshot

    global_capture = capture_state.get("__global__", _build_global_capture_state())
    global_payload = global_capture.get("summary_log")
    global_lines = global_capture.get("summary_log_lines", []) or []
    if global_payload and global_payload.get("text"):
        snapshot["global_summary_log"] = global_payload
    elif global_lines:
        snapshot["global_summary_log"] = {
            "updated_at": global_capture.get("updated_at") or datetime.now().strftime("%H:%M:%S"),
            "text": _normalize_timestamp_log_lines("\n".join(global_lines)),
        }

    for workflow_id in sorted(optimizer.workflows.keys()):
        workflow_data = optimizer.workflows.get(workflow_id, {}) or {}
        raw_history = workflow_data.get("iteration_history", []) or []
        history = _simplify_iteration_history(raw_history)
        capture = capture_state.get(workflow_id, _build_default_capture_state())
        current_iteration_real = int(capture.get("current_iteration", 0) or 0)
        if not current_iteration_real and raw_history:
            current_iteration_real = int(raw_history[-1].get("iteration", 0) or 0)
        max_iterations_real = int(capture.get("max_iterations", config.get("max_iterations", 5)) or config.get("max_iterations", 5))
        current_iteration = _real_to_display_iteration(current_iteration_real)
        max_iterations = _real_to_display_max_iterations(max_iterations_real)
        snapshot["workflows"][workflow_id] = {
            "current_iteration": current_iteration,
            "max_iterations": max_iterations,
            "progress": (current_iteration / max_iterations) if max_iterations else 0.0,
            "latest_reasoning": capture.get("latest_reasoning"),
            "latest_summary": capture.get("latest_summary"),
            "reasoning_history": capture.get("reasoning_history", []),
            "summary_history": capture.get("summary_history", []),
            "latest_plan_metrics": capture.get("latest_plan_metrics"),
            "plan_metrics_history": capture.get("plan_metrics_history", []),
            "latest_round_stats": capture.get("latest_round_stats"),
            "round_stats_history": capture.get("round_stats_history", []),
            "latest_parameter_details": capture.get("latest_parameter_details"),
            "parameter_details_history": capture.get("parameter_details_history", []),
            "baseline_log": capture.get("baseline_log"),
            "baseline_log_history": capture.get("baseline_log_history", []),
            "early_stopped": bool(capture.get("early_stopped", False)),
            "latest_status": capture.get("latest_status", "等待日志输出"),
            "status_updated_at": capture.get("status_updated_at"),
            "iteration_history": history,
            "best_metrics": workflow_data.get("best_metrics"),
            "best_iteration": _real_to_display_iteration(int(workflow_data.get("best_iteration", 0) or 0)),
        }

        if not snapshot["workflows"][workflow_id].get("latest_parameter_details"):
            pending_lines = capture.get("parameter_detail_lines", []) or []
            if pending_lines:
                pending_text = _clean_parameter_detail_lines(pending_lines)
                if pending_text:
                    snapshot["workflows"][workflow_id]["latest_parameter_details"] = {
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                        "iteration": current_iteration,
                        "text": pending_text,
                    }

        if not snapshot["workflows"][workflow_id].get("baseline_log"):
            baseline_lines = capture.get("baseline_log_lines", []) or []
            if baseline_lines:
                baseline_text = "\n".join(baseline_lines).strip()
                if baseline_text:
                    snapshot["workflows"][workflow_id]["baseline_log"] = {
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                        "iteration": 0,
                        "text": baseline_text,
                    }

    return snapshot


class WebCaptureHandler(logging.Handler):
    def __init__(self, capture_state: dict[str, Any], capture_lock: threading.Lock):
        super().__init__()
        self.capture_state = capture_state
        self.capture_lock = capture_lock

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return

        start_global_summary = ("【Token使用统计】" in message) or ("【全局耗时统计】" in message)
        with self.capture_lock:
            global_state = self.capture_state.setdefault("__global__", _build_global_capture_state())
            collecting_global_summary = bool(global_state.get("collecting_summary_log"))
            is_global_summary_line = collecting_global_summary or ("【总】" in message and start_global_summary)
            if start_global_summary and not collecting_global_summary:
                global_state["collecting_summary_log"] = True
                global_state["summary_log_lines"] = []
                is_global_summary_line = True

            if is_global_summary_line and "【总】" in message:
                global_state.setdefault("summary_log_lines", []).extend(_format_record_lines(record, message))
                global_state["updated_at"] = datetime.now().strftime("%H:%M:%S")
                global_state["summary_log"] = {
                    "updated_at": global_state["updated_at"],
                    "text": _normalize_timestamp_log_lines("\n".join(global_state.get("summary_log_lines", []))),
                }
                if "【并行优化循环完成】" in message:
                    global_state["collecting_summary_log"] = False
                return

        workflow_id = self._resolve_workflow_id(record, message)
        if not workflow_id:
            return

        with self.capture_lock:
            workflow_state = self.capture_state.setdefault(workflow_id, _build_default_capture_state())
            if "早停" in message:
                workflow_state["early_stopped"] = True
            workflow_state["latest_status"] = _short_status(message)
            workflow_state["status_updated_at"] = datetime.now().strftime("%H:%M:%S")
            formatted_lines = _format_record_lines(record, message)

            round_match = ROUND_REGEX.search(message)
            if round_match:
                workflow_state["current_iteration"] = int(round_match.group(1))
                workflow_state["max_iterations"] = int(round_match.group(2))

            if "【模拟】initial_baseline" in message:
                workflow_state["collecting_baseline_log"] = True
                workflow_state["baseline_log_lines"] = []

            if workflow_state.get("collecting_baseline_log"):
                stop_match = ROUND_REGEX.search(message)
                if stop_match and int(stop_match.group(1)) >= 2:
                    baseline_text = "\n".join(workflow_state.get("baseline_log_lines", [])).strip()
                    if baseline_text:
                        payload = {
                            "updated_at": datetime.now().strftime("%H:%M:%S"),
                            "iteration": 0,
                            "text": baseline_text,
                        }
                        workflow_state["baseline_log"] = payload
                        _upsert_iteration_payload(workflow_state.setdefault("baseline_log_history", []), payload)
                    workflow_state["collecting_baseline_log"] = False
                    workflow_state["baseline_log_lines"] = []
                else:
                    workflow_state["baseline_log_lines"].extend(formatted_lines)

            reasoning_match = REASONING_HEADER_REGEX.search(message)
            if reasoning_match and "最终采用方案" in reasoning_match.group(1):
                lines = [line for line in message.splitlines()[1:] if line.strip()]
                reasoning_text = "\n".join(lines).strip()
                if reasoning_text:
                    reasoning_payload = {
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                        "iteration": _real_to_display_iteration(workflow_state.get("current_iteration", 0)),
                        "text": reasoning_text,
                    }
                    workflow_state["latest_reasoning"] = reasoning_payload
                    _upsert_iteration_payload(workflow_state.setdefault("reasoning_history", []), reasoning_payload)

            if "【建议指标】" in message:
                metrics_payload = {
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                    "iteration": _real_to_display_iteration(workflow_state.get("current_iteration", 0)),
                    **_parse_plan_metrics_line(message),
                }
                workflow_state["latest_plan_metrics"] = metrics_payload
                _upsert_iteration_payload(workflow_state.setdefault("plan_metrics_history", []), metrics_payload)

            if "【本轮修改统计】" in message:
                stats_payload = {
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                    "iteration": _real_to_display_iteration(workflow_state.get("current_iteration", 0)),
                    "modified_objects": None,
                    "modified_fields": None,
                }
                workflow_state["latest_round_stats"] = stats_payload
                _upsert_iteration_payload(workflow_state.setdefault("round_stats_history", []), stats_payload)

            if "修改对象总数:" in message or "修改字段总数:" in message:
                stats_payload = dict(workflow_state.get("latest_round_stats") or {})
                stats_payload["updated_at"] = datetime.now().strftime("%H:%M:%S")
                stats_payload["iteration"] = _real_to_display_iteration(workflow_state.get("current_iteration", 0))
                if "修改对象总数:" in message:
                    stats_payload["modified_objects"] = _extract_int_after_label(message, "修改对象总数:")
                if "修改字段总数:" in message:
                    stats_payload["modified_fields"] = _extract_int_after_label(message, "修改字段总数:")
                workflow_state["latest_round_stats"] = stats_payload
                _upsert_iteration_payload(workflow_state.setdefault("round_stats_history", []), stats_payload)

            if "【修改摘要】" in message:
                workflow_state["collecting_summary"] = True
                workflow_state["summary_lines"] = []
                return

            if workflow_state.get("collecting_summary"):
                if "【计划修改】" in message:
                    summary_text = _clean_summary_lines(workflow_state.get("summary_lines", []))
                    if summary_text:
                        summary_payload = {
                            "updated_at": datetime.now().strftime("%H:%M:%S"),
                            "iteration": _real_to_display_iteration(workflow_state.get("current_iteration", 0)),
                            "text": summary_text,
                        }
                        workflow_state["latest_summary"] = summary_payload
                        _upsert_iteration_payload(workflow_state.setdefault("summary_history", []), summary_payload)
                    workflow_state["collecting_summary"] = False
                    workflow_state["summary_lines"] = []
                else:
                    workflow_state["summary_lines"].extend(formatted_lines)

            if "【参数修改详情】" in message:
                workflow_state["collecting_parameter_details"] = True
                workflow_state["parameter_detail_lines"] = []

            if workflow_state.get("collecting_parameter_details"):
                workflow_state["parameter_detail_lines"].extend(formatted_lines)
                raw_message = _strip_log_prefix(message)
                should_finalize = (
                    "已保存单工作流能耗曲线" in raw_message
                    or raw_message.startswith("启动")
                )
                if should_finalize:
                    detail_text = _clean_parameter_detail_lines(workflow_state.get("parameter_detail_lines", []))
                    if detail_text:
                        payload = {
                            "updated_at": datetime.now().strftime("%H:%M:%S"),
                            "iteration": _real_to_display_iteration(workflow_state.get("current_iteration", 0)),
                            "text": detail_text,
                        }
                        workflow_state["latest_parameter_details"] = payload
                        _upsert_iteration_payload(workflow_state.setdefault("parameter_details_history", []), payload)
                    workflow_state["collecting_parameter_details"] = False
                    workflow_state["parameter_detail_lines"] = []

    def _resolve_workflow_id(self, record: logging.LogRecord, message: str) -> str | None:
        logger_name = str(getattr(record, "name", "") or "")
        match = WORKFLOW_REGEX.search(logger_name)
        if match:
            return f"workflow_{match.group(1)}"

        workflow_prefix = str(getattr(record, "workflow_prefix", "") or "")
        match = re.search(r"工作流(\d+)", workflow_prefix)
        if match:
            return f"workflow_{match.group(1)}"

        match = WORKFLOW_REGEX.search(message)
        if match:
            return f"workflow_{match.group(1)}"

        return None


def _attach_capture_handler(optimizer: EnergyPlusOptimizer, capture_state: dict[str, Any], capture_lock: threading.Lock) -> logging.Handler:
    handler = WebCaptureHandler(capture_state, capture_lock)
    handler.setLevel(logging.INFO)
    optimizer.logger.addHandler(handler)
    for workflow_data in optimizer.workflows.values():
        workflow_logger = workflow_data.get("logger")
        if workflow_logger:
            workflow_logger.addHandler(handler)
    return handler


def _detach_capture_handler(optimizer: EnergyPlusOptimizer, handler: logging.Handler) -> None:
    try:
        optimizer.logger.removeHandler(handler)
    except Exception:
        pass
    for workflow_data in optimizer.workflows.values():
        workflow_logger = workflow_data.get("logger")
        if not workflow_logger:
            continue
        try:
            workflow_logger.removeHandler(handler)
        except Exception:
            pass


def _suppress_console_logs(logger_obj: logging.Logger) -> None:
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.CRITICAL + 1)


def _snapshot_pump(
    stop_event: threading.Event,
    optimizer: EnergyPlusOptimizer,
    capture_state: dict[str, Any],
    capture_lock: threading.Lock,
    snapshot_path: str,
    config: dict[str, Any],
    runtime_state: dict[str, Any],
) -> None:
    while not stop_event.wait(0.8):
        with capture_lock:
            capture_copy = json.loads(json.dumps(capture_state, ensure_ascii=False))
        snapshot = _compose_snapshot(
            optimizer=optimizer,
            capture_state=capture_copy,
            config=config,
            status=runtime_state.get("status", "running"),
            started_at=runtime_state.get("started_at"),
            finished_at=runtime_state.get("finished_at"),
            error=runtime_state.get("error"),
        )
        try:
            _write_snapshot(snapshot_path, snapshot)
        except Exception as exc:
            runtime_state["last_snapshot_error"] = str(exc)


def _run_optimizer_in_thread(snapshot_path: str, config: dict[str, Any], runtime_state: dict[str, Any]) -> None:
    capture_state: dict[str, Any] = {
        f"workflow_{index + 1}": _build_default_capture_state()
        for index in range(int(config.get("num_workflows", 2) or 2))
    }
    capture_state["__global__"] = _build_global_capture_state()
    capture_lock = threading.Lock()
    stop_event = threading.Event()
    optimizer = None
    capture_handler = None
    snapshot_thread = None

    runtime_state["status"] = "running"
    runtime_state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    runtime_state["finished_at"] = None
    runtime_state["error"] = None
    _write_snapshot(snapshot_path, _build_empty_snapshot(config=config, status="running"))

    try:
        optimizer = EnergyPlusOptimizer(
            idf_path=config["idf_path"],
            idd_path=config["idd_path"],
            api_key_path=config["api_key_path"],
            epw_path=config["epw_path"],
            log_dir=config["log_dir"],
            num_workflows=int(config["num_workflows"]),
        )
        _suppress_console_logs(optimizer.logger)
        capture_handler = _attach_capture_handler(optimizer, capture_state, capture_lock)

        snapshot_thread = threading.Thread(
            target=_snapshot_pump,
            args=(stop_event, optimizer, capture_state, capture_lock, snapshot_path, config, runtime_state),
            daemon=True,
        )
        snapshot_thread.start()

        optimizer.max_iterations_cap = int(config.get("max_iterations_cap", config.get("optimization_rounds", DEFAULT_MAX_ITERATIONS)) or DEFAULT_MAX_ITERATIONS)
        optimizer.early_stop_target_total_saving_pct = float(
            config.get("early_stop_target_total_saving_pct", DEFAULT_EARLY_STOP_TARGET_SAVING_PCT)
            or DEFAULT_EARLY_STOP_TARGET_SAVING_PCT
        )
        optimizer.run_optimization_loop(max_iterations=int(config["max_iterations"]))
        runtime_state["status"] = "finished"
        runtime_state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        runtime_state["status"] = "error"
        runtime_state["error"] = str(exc)
        runtime_state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finally:
        stop_event.set()
        if snapshot_thread:
            snapshot_thread.join(timeout=1.5)
        if optimizer and capture_handler:
            _detach_capture_handler(optimizer, capture_handler)
        with capture_lock:
            capture_copy = json.loads(json.dumps(capture_state, ensure_ascii=False))
        final_snapshot = _compose_snapshot(
            optimizer=optimizer,
            capture_state=capture_copy,
            config=config,
            status=runtime_state.get("status", "finished"),
            started_at=runtime_state.get("started_at"),
            finished_at=runtime_state.get("finished_at"),
            error=runtime_state.get("error"),
        )
        _write_snapshot(snapshot_path, final_snapshot)


def _build_metrics_dataframe(iteration_history: list[dict[str, Any]]) -> pd.DataFrame:
    if not iteration_history:
        return pd.DataFrame()

    rows = []
    baseline = iteration_history[0].get("metrics", {}) or {}
    base_total = float(baseline.get("total_site_energy_kwh", 0) or 0)
    base_eui = float(baseline.get("eui_kwh_per_m2", 0) or 0)
    base_cooling = float(baseline.get("total_cooling_kwh", 0) or 0)
    base_heating = float(baseline.get("total_heating_kwh", 0) or 0)

    def _pct(delta: float, base: float) -> float:
        if not base:
            return 0.0
        return (delta / base) * 100.0

    for item in iteration_history:
        metrics = item.get("metrics", {}) or {}
        total = float(metrics.get("total_site_energy_kwh", 0) or 0)
        eui = float(metrics.get("eui_kwh_per_m2", 0) or 0)
        cooling = float(metrics.get("total_cooling_kwh", 0) or 0)
        heating = float(metrics.get("total_heating_kwh", 0) or 0)

        total_delta = base_total - total
        eui_delta = base_eui - eui
        cooling_delta = base_cooling - cooling
        heating_delta = base_heating - heating

        rows.append(
            {
                "轮次": int(item.get("iteration", 0) or 0),
                "总建筑能耗(kWh)": total,
                "EUI(kWh/m²)": eui,
                "制冷能耗(kWh/m²)": cooling,
                "供暖能耗(kWh/m²)": heating,
                "总建筑能耗节能幅度(%)": _pct(total_delta, base_total),
                "EUI节能幅度(%)": _pct(eui_delta, base_eui),
                "制冷节能幅度(%)": _pct(cooling_delta, base_cooling),
                "供暖节能幅度(%)": _pct(heating_delta, base_heating),
            }
        )
    return pd.DataFrame(rows)


def _build_progressive_curve(dataframe: pd.DataFrame, workflow_id: str):
    if dataframe.empty:
        return None

    melted = pd.DataFrame(
        {
            "轮次": list(dataframe["轮次"]) + list(dataframe["轮次"]),
            "数值": list(dataframe["制冷能耗(kWh/m²)"]) + list(dataframe["供暖能耗(kWh/m²)"]),
            "指标": ["制冷能耗"] * len(dataframe) + ["供暖能耗"] * len(dataframe),
        }
    )

    rounds = sorted({int(value) for value in dataframe["轮次"].tolist()})
    return (
        alt.Chart(melted)
        .mark_line(point=alt.OverlayMarkDef(size=120, filled=True), strokeWidth=3.2)
        .encode(
            x=alt.X(
                "轮次:O",
                title="迭代轮次",
                sort=rounds,
                axis=alt.Axis(labelAngle=0, labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=16, titleFontWeight="bold"),
            ),
            y=alt.Y(
                "数值:Q",
                title="能耗 (kWh/m²)",
                axis=alt.Axis(labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=14, tickCount=6, titleFontWeight="bold"),
            ),
            color=alt.Color(
                "指标:N",
                scale=alt.Scale(domain=["制冷能耗", "供暖能耗"], range=["#1F77B4", "#E67E22"]),
                legend=None,
            ),
            strokeDash=alt.StrokeDash(
                "指标:N",
                scale=alt.Scale(domain=["制冷能耗", "供暖能耗"], range=[[1, 0], [8, 4]]),
            ),
            tooltip=["轮次:O", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
        )
        .properties(
            width=1080,
            height=460,
            padding={"left": 24, "right": 20, "top": 8, "bottom": 20},
        )
        .configure(background="#FCFCFA")
        .configure_view(strokeWidth=1, stroke="#C8D3DE", fill="#FFFFFF")
        .configure_axis(
            grid=True,
            gridColor="#D2DCE6",
            domainColor="#516575",
            tickColor="#516575",
            labelColor="#253946",
            titleColor="#1A2E3A",
        )
    )


def _build_total_energy_curve(dataframe: pd.DataFrame, workflow_id: str):
    if dataframe.empty:
        return None

    rounds = sorted({int(value) for value in dataframe["轮次"].tolist()})
    curve_df = pd.DataFrame(
        {
            "轮次": list(dataframe["轮次"]),
            "总建筑能耗(kWh)": list(dataframe["总建筑能耗(kWh)"]),
        }
    )
    return (
        alt.Chart(curve_df)
        .mark_line(point=alt.OverlayMarkDef(size=120, filled=True), strokeWidth=3.2, color="#2A9D8F")
        .encode(
            x=alt.X(
                "轮次:O",
                title="迭代轮次",
                sort=rounds,
                axis=alt.Axis(labelAngle=0, labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=16, titleFontWeight="bold"),
            ),
            y=alt.Y(
                "总建筑能耗(kWh):Q",
                title="总建筑能耗 (kWh)",
                axis=alt.Axis(labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=14, tickCount=6, titleFontWeight="bold"),
            ),
            tooltip=["轮次:O", alt.Tooltip("总建筑能耗(kWh):Q", format=".4f")],
        )
        .properties(
            width=1080,
            height=420,
            padding={"left": 24, "right": 20, "top": 8, "bottom": 20},
        )
        .configure(background="#FCFCFA")
        .configure_view(strokeWidth=1, stroke="#C8D3DE", fill="#FFFFFF")
        .configure_axis(
            grid=True,
            gridColor="#D2DCE6",
            domainColor="#516575",
            tickColor="#516575",
            labelColor="#253946",
            titleColor="#1A2E3A",
        )
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _build_all_workflows_progressive_curve(workflows: dict[str, dict[str, Any]]):
    rows = []
    for workflow_id, workflow_snapshot in sorted((workflows or {}).items()):
        dataframe = _build_metrics_dataframe(workflow_snapshot.get("iteration_history", []) or [])
        if dataframe.empty:
            continue
        for _, row in dataframe.iterrows():
            rows.append({"工作流": workflow_id, "轮次": int(row["轮次"]), "指标": "制冷能耗", "数值": float(row["制冷能耗(kWh/m²)"])})
            rows.append({"工作流": workflow_id, "轮次": int(row["轮次"]), "指标": "供暖能耗", "数值": float(row["供暖能耗(kWh/m²)"])})

    if not rows:
        return None

    curve_df = pd.DataFrame(rows)
    rounds = sorted({int(value) for value in curve_df["轮次"].tolist()})
    workflow_ids = sorted({str(value) for value in curve_df["工作流"].tolist()})
    color_range = [WORKFLOW_PALETTE[index % len(WORKFLOW_PALETTE)] for index in range(len(workflow_ids))]

    return (
        alt.Chart(curve_df)
        .mark_line(point=alt.OverlayMarkDef(size=95, filled=True), strokeWidth=2.8)
        .encode(
            x=alt.X(
                "轮次:O",
                title="迭代轮次",
                sort=rounds,
                axis=alt.Axis(labelAngle=0, labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=16, titleFontWeight="bold"),
            ),
            y=alt.Y(
                "数值:Q",
                title="能耗 (kWh/m²)",
                axis=alt.Axis(labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=14, tickCount=6, titleFontWeight="bold"),
            ),
            color=alt.Color(
                "工作流:N",
                scale=alt.Scale(domain=workflow_ids, range=color_range),
                legend=None,
            ),
            strokeDash=alt.StrokeDash(
                "指标:N",
                scale=alt.Scale(domain=["制冷能耗", "供暖能耗"], range=[[1, 0], [8, 4]]),
                legend=None,
            ),
            tooltip=["工作流:N", "轮次:O", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
        )
        .properties(
            width=1080,
            height=460,
            padding={"left": 24, "right": 20, "top": 8, "bottom": 20},
        )
        .configure(background="#FCFCFA")
        .configure_view(strokeWidth=1, stroke="#C8D3DE", fill="#FFFFFF")
        .configure_axis(
            grid=True,
            gridColor="#D2DCE6",
            domainColor="#516575",
            tickColor="#516575",
            labelColor="#253946",
            titleColor="#1A2E3A",
        )
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _build_all_workflows_total_energy_curve(workflows: dict[str, dict[str, Any]]):
    rows = []
    for workflow_id, workflow_snapshot in sorted((workflows or {}).items()):
        dataframe = _build_metrics_dataframe(workflow_snapshot.get("iteration_history", []) or [])
        if dataframe.empty:
            continue
        for _, row in dataframe.iterrows():
            rows.append(
                {
                    "工作流": workflow_id,
                    "轮次": int(row["轮次"]),
                    "总建筑能耗(kWh)": float(row["总建筑能耗(kWh)"]),
                }
            )

    if not rows:
        return None

    curve_df = pd.DataFrame(rows)
    rounds = sorted({int(value) for value in curve_df["轮次"].tolist()})
    workflow_ids = sorted({str(value) for value in curve_df["工作流"].tolist()})
    color_range = [WORKFLOW_PALETTE[index % len(WORKFLOW_PALETTE)] for index in range(len(workflow_ids))]

    return (
        alt.Chart(curve_df)
        .mark_line(point=alt.OverlayMarkDef(size=95, filled=True), strokeWidth=2.8)
        .encode(
            x=alt.X(
                "轮次:O",
                title="迭代轮次",
                sort=rounds,
                axis=alt.Axis(labelAngle=0, labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=16, titleFontWeight="bold"),
            ),
            y=alt.Y(
                "总建筑能耗(kWh):Q",
                title="总建筑能耗 (kWh)",
                axis=alt.Axis(labelFontSize=17, titleFontSize=22, labelPadding=8, titlePadding=14, tickCount=6, titleFontWeight="bold"),
            ),
            color=alt.Color(
                "工作流:N",
                scale=alt.Scale(domain=workflow_ids, range=color_range),
                legend=None,
            ),
            tooltip=["工作流:N", "轮次:O", alt.Tooltip("总建筑能耗(kWh):Q", format=".4f")],
        )
        .properties(
            width=1080,
            height=420,
            padding={"left": 24, "right": 20, "top": 8, "bottom": 20},
        )
        .configure(background="#FCFCFA")
        .configure_view(strokeWidth=1, stroke="#C8D3DE", fill="#FFFFFF")
        .configure_axis(
            grid=True,
            gridColor="#D2DCE6",
            domainColor="#516575",
            tickColor="#516575",
            labelColor="#253946",
            titleColor="#1A2E3A",
        )
    )


def _render_summary_curve_header(workflow_ids: list[str]) -> None:
    workflow_items = []
    for index, workflow_id in enumerate(workflow_ids):
        color = WORKFLOW_PALETTE[index % len(WORKFLOW_PALETTE)]
        workflow_items.append(
            f"<span class='chart-shell-legend-item'><span class='chart-shell-legend-swatch' style='background:{color}'></span>{html.escape(workflow_id)}</span>"
        )

    st.markdown(
        f"""
        <div class='chart-shell-header'>
            <div></div>
            <div class='chart-shell-title'>并行工作流冷暖能耗迭代曲线</div>
            <div class='chart-shell-legend chart-shell-legend-stack'>
                <div class='chart-shell-legend-row'>
                    {''.join(workflow_items)}
                </div>
                <div class='chart-shell-legend-row'>
                    <span class='chart-shell-legend-item'><span class='chart-shell-legend-line chart-shell-legend-cooling'></span>制冷能耗（实线）</span>
                    <span class='chart-shell-legend-item'><span class='chart-shell-legend-line chart-shell-legend-heating'></span>供暖能耗（虚线）</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_total_energy_curve_header(workflow_ids: list[str]) -> None:
    workflow_items = []
    for index, workflow_id in enumerate(workflow_ids):
        color = WORKFLOW_PALETTE[index % len(WORKFLOW_PALETTE)]
        workflow_items.append(
            f"<span class='chart-shell-legend-item'><span class='chart-shell-legend-swatch' style='background:{color}'></span>{html.escape(workflow_id)}</span>"
        )

    st.markdown(
        f"""
        <div class='chart-shell-header'>
            <div></div>
            <div class='chart-shell-title'>并行工作流总建筑能耗迭代曲线</div>
            <div class='chart-shell-legend'>
                {''.join(workflow_items)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _table_column_config(dataframe: pd.DataFrame) -> dict[str, Any]:
    config = {}
    for column in dataframe.columns:
        if column in {"轮次", "最优轮次", "工作流", "全局最佳标识", "最优迭代标识"}:
            width = "small"
        elif "路径" in column:
            width = "large"
        else:
            width = "medium"
        config[column] = st.column_config.Column(column, width=width)
    return config


def _stringify_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    rendered = dataframe.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].apply(lambda value: "" if value is None else str(value))
    return rendered


def _left_align_dataframe(dataframe: pd.DataFrame):
    return dataframe.style.hide(axis="index").set_properties(**{"text-align": "left"}).set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("font-size", "20px"), ("font-weight", "800"), ("color", "#1F2F36")]},
            {"selector": "td", "props": [("text-align", "left"), ("font-size", "19px"), ("font-weight", "800"), ("color", "#1F2F36")]},
        ]
    )


def _render_html_table(dataframe: pd.DataFrame) -> None:
    html_table = dataframe.to_html(index=False, escape=False, classes="custom-html-table")
    st.markdown(f"<div class='table-wrap'>{html_table}</div>", unsafe_allow_html=True)


def _render_notice(message: str, kind: str = "info") -> None:
    css_class = "notice-info" if kind == "info" else "notice-error"
    st.markdown(f"<div class='notice-banner {css_class}'>{html.escape(str(message))}</div>", unsafe_allow_html=True)


def _render_metric_card(title: str, value: str, delta: str | None = None) -> None:
    delta_class = ""
    if delta:
        delta_text = str(delta).strip()
        if delta_text.startswith("+"):
            delta_class = " pos"
        elif delta_text.startswith("-"):
            delta_class = " neg"
        else:
            delta_class = " neutral"
    delta_html = f"<div class='metric-card-delta{delta_class}'>{html.escape(delta)}</div>" if delta else ""
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-title'>{html.escape(title)}</div>
            <div class='metric-card-value'>{html.escape(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _visible_workflow_ids(snapshot: dict[str, Any]) -> list[str]:
    workflows = snapshot.get("workflows") or {}
    snapshot_status = str(snapshot.get("status", "idle") or "idle")
    session_count = int(st.session_state.get("cfg_num_workflows", 0) or 0)
    fallback_count = len(workflows) if workflows else DEFAULT_NUM_WORKFLOWS
    configured_count = session_count if session_count > 0 else int(snapshot.get("config", {}).get("num_workflows", fallback_count) or fallback_count)
    if configured_count <= 0:
        return sorted(workflows.keys())

    # 在未启动时，直接按当前配置显示对应数量的工作流模块。
    if snapshot_status == "idle":
        return [f"workflow_{index + 1}" for index in range(configured_count)]

    if not workflows:
        return []

    preferred = [f"workflow_{index + 1}" for index in range(configured_count) if f"workflow_{index + 1}" in workflows]
    if preferred:
        return preferred
    return sorted(workflows.keys())[:configured_count]


def _pick_best_iteration(workflow_snapshot: dict[str, Any]) -> int:
    best_iteration = int(workflow_snapshot.get("best_iteration", 0) or 0)
    if best_iteration > 0:
        return best_iteration

    history = workflow_snapshot.get("iteration_history", []) or []
    if not history:
        return 0

    best_item = min(
        history,
        key=lambda item: float((item.get("metrics") or {}).get("total_site_energy_kwh", float("inf")) or float("inf")),
    )
    return int(best_item.get("iteration", 0) or 0)


def _best_metrics_from_history(workflow_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    best_metrics = workflow_snapshot.get("best_metrics")
    if isinstance(best_metrics, dict) and best_metrics:
        return best_metrics

    best_iteration = _pick_best_iteration(workflow_snapshot)
    history = workflow_snapshot.get("iteration_history", []) or []
    for item in history:
        if int(item.get("iteration", 0) or 0) == best_iteration:
            return (item.get("metrics") or {})
    return None


def _best_idf_from_history(workflow_snapshot: dict[str, Any], best_iteration: int) -> str | None:
    history = workflow_snapshot.get("iteration_history", []) or []
    for item in history:
        if int(item.get("iteration", 0) or 0) == best_iteration:
            return item.get("idf_path")
    return None


def _render_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');
        .stApp {
            background:
                radial-gradient(920px 380px at 0% 0%, rgba(127, 203, 190, 0.24), rgba(127, 203, 190, 0) 60%),
                radial-gradient(820px 420px at 100% 0%, rgba(255, 183, 128, 0.22), rgba(255, 183, 128, 0) 60%),
                linear-gradient(180deg, #F7F5F0 0%, #EFE8DB 100%);
            color: #2C3A3F;
            font-family: 'Times New Roman', 'Noto Sans SC', serif;
            line-height: 1.55;
        }
        /* Keep rerun updates visually stable: disable Streamlit stale-element dimming. */
        .stale-element,
        [data-testid="staleElement"] {
            opacity: 1 !important;
            filter: none !important;
        }
        .stApp [data-stale="true"],
        .stApp [class*="stale"],
        .stApp [class*="Stale"] {
            opacity: 1 !important;
            filter: none !important;
        }
        /* Disable mount/fade transitions that become obvious on heavy sections. */
        .stApp [data-testid="stVerticalBlock"],
        .stApp [data-testid="element-container"],
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stVegaLiteChart"],
        .stApp [data-testid="stTable"] {
            transition: none !important;
            animation: none !important;
        }
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
        header[data-testid="stHeader"] {
            display: none !important;
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        /* Keep helper iframe from intercepting clicks/wheel in any viewport region. */
        [data-testid="stIFrame"] iframe,
        iframe[title="st.iframe"] {
            pointer-events: none !important;
        }
        div[data-testid="stToolbar"] {
            background: transparent !important;
        }
        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] a,
        [data-testid="stToolbar"] [role="button"],
        [data-testid="stStatusWidget"] button,
        [data-testid="stStatusWidget"] [role="button"] {
            color: #000000 !important;
            font-size: 15px !important;
            font-weight: 700 !important;
        }
        [data-testid="stToolbar"] button span,
        [data-testid="stToolbar"] a span,
        [data-testid="stToolbar"] [role="button"] span,
        [data-testid="stStatusWidget"] button span,
        [data-testid="stStatusWidget"] [role="button"] span {
            color: #000000 !important;
            font-size: 15px !important;
            font-weight: 700 !important;
        }
        [data-testid="stStatusWidget"] svg,
        [data-testid="stStatusWidget"] img {
            filter: brightness(0.42) contrast(1.12);
        }
        .stApp :lang(zh),
        .stApp [lang="zh"],
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            font-family: 'Noto Sans SC', 'Times New Roman', sans-serif;
        }
        .stMarkdown p, .stCaption, label, .stRadio label, .stSelectbox label, .stFileUploader label {
            color: #2C3A3F !important;
        }
        .stCaption {
            font-size: 16px !important;
            font-weight: 700 !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #2C3A3F !important;
        }
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(52, 103, 115, 0.16);
            border-radius: 12px;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: #1F3942 !important;
            background: transparent !important;
            font-size: 21px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stExpanderDetails"] {
            background: rgba(255, 255, 255, 0.92) !important;
            color: #2C3A3F !important;
        }
        div[data-testid="stFileUploader"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            font-size: 18px !important;
            font-weight: 700 !important;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
        [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #1F3942 !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: #F6F1E4 !important;
            border: 1px dashed #A38A5A !important;
        }
        div[data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] label {
            color: #2F2413 !important;
        }
        div[data-testid="stFileUploaderDropzone"] svg {
            fill: #2F2413 !important;
            color: #2F2413 !important;
        }
        div[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 12px;
        }
        /* NumberInput: suppress sticky focus ring on +/- steppers after click. */
        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInput"] button:focus,
        div[data-testid="stNumberInput"] button:focus-visible,
        div[data-testid="stNumberInput"] button:active,
        div[data-testid="stNumberInput"] [role="button"],
        div[data-testid="stNumberInput"] [role="button"]:focus,
        div[data-testid="stNumberInput"] [role="button"]:focus-visible,
        div[data-testid="stNumberInput"] [role="button"]:active {
            outline: none !important;
            box-shadow: none !important;
            border-color: #C5B79A !important;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus-visible,
        div[data-testid="stNumberInput"] input:active {
            outline: none !important;
            box-shadow: none !important;
            border-color: #C5B79A !important;
        }
        div[data-testid="stNumberInput"] [data-baseweb="input"],
        div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
        div[data-testid="stNumberInput"] div:focus-within {
            outline: none !important;
            box-shadow: none !important;
            border-color: #C5B79A !important;
        }
        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploader"] [role="button"] {
            background: #F2E8D5 !important;
            border: 1px solid #9C7A40 !important;
            color: #2F2413 !important;
        }
        div[data-testid="stFileUploader"] button:hover,
        div[data-testid="stFileUploader"] [role="button"]:hover {
            background: #EAD8B8 !important;
            border-color: #7D5E2C !important;
            color: #20170C !important;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] * {
            color: #111111 !important;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] svg {
            fill: #111111 !important;
            color: #111111 !important;
        }
        .stButton button {
            background: #F6F3EC;
            border: 1px solid #C5B79A;
            color: #2C3A3F;
            border-radius: 12px;
            font-weight: 700;
            font-size: 17px;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stButton"] button *,
        [data-testid^="stBaseButton-"] {
            font-size: 17px !important;
            font-weight: 700 !important;
        }
        .stButton button:hover {
            border-color: #6A8D92;
            color: #1E3D47;
        }
        .stButton button:disabled,
        .stButton button:disabled *,
        div[data-testid="stButton"] button:disabled,
        div[data-testid="stButton"] button:disabled * {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            opacity: 1 !important;
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #2D6A8E, #3E8CB5);
            color: #FFFFFF;
            border-color: #2D6A8E;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(52, 103, 115, 0.18);
            border-radius: 12px;
            padding: 10px 12px;
            min-height: 118px;
        }
        .metric-card-title {
            font-size: 21px;
            font-weight: 800;
            color: #1E3D47;
            line-height: 1.25;
        }
        .metric-card-value {
            font-size: 33px;
            font-weight: 800;
            color: #2C3A3F;
            margin-top: 2px;
            line-height: 1.15;
        }
        .metric-card-delta {
            font-size: 20px;
            font-weight: 800;
            margin-top: 3px;
            color: #1F3942;
        }
        .metric-card-delta.pos {
            color: #1F8A4C;
            border-left: 4px solid #1F8A4C;
            padding-left: 8px;
        }
        .metric-card-delta.neg {
            color: #BE3D37;
            border-left: 4px solid #BE3D37;
            padding-left: 8px;
        }
        .metric-card-delta.neutral {
            color: #1F3942;
            border-left: 4px solid #6F8792;
            padding-left: 8px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(52, 103, 115, 0.15);
            border-radius: 12px;
            padding: 10px 12px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #2C3A3F !important;
        }
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] label {
            font-size: 21px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] div,
        div[data-testid="stMetricLabel"] span {
            font-size: 21px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 33px !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 20px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] small,
        div[data-testid="stMetric"] p {
            font-size: 16px !important;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', 'Noto Sans SC', sans-serif;
            letter-spacing: 0.4px;
            color: #1E3D47;
        }
        .glass {
            border: 1px solid rgba(52, 103, 115, 0.18);
            background: rgba(255, 255, 255, 0.84);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(2px);
        }
        .flow-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .flow-item {
            border-radius: 12px;
            padding: 10px;
            min-height: 72px;
            background: linear-gradient(135deg, #E9D8A6, #FAEDCD);
            border: 1px solid rgba(202, 103, 2, 0.18);
            font-size: 14px;
            font-weight: 700;
        }
        .flow-item strong {
            display: block;
            margin-bottom: 6px;
            font-size: 17px;
        }
        .workflow-chip-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 10px;
        }
        .workflow-chip {
            border-radius: 14px;
            padding: 14px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(0, 95, 115, 0.14);
            font-size: 17px;
            font-weight: 700;
            line-height: 1.45;
        }
        .workflow-chip strong {
            display: block;
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .workflow-status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            color: #ffffff;
            margin-left: 8px;
        }
        .workflow-status-badge.badge-stopped { background: #ff7a45; }
        .workflow-status-badge.badge-running { background: #0b84ff; }
        .workflow-status-badge.badge-finished { background: #2ea44f; }
        .workflow-status-badge.badge-idle { background: #95a5a6; }
        .mono-block {
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.55;
            margin: 0;
        }
        .mono-log-block {
            font-size: 13px;
            line-height: 1.55;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        .mono-log-line {
            white-space: pre-wrap;
            margin: 0;
            padding: 0;
        }
        .mono-log-line-blank {
            min-height: 0.95em;
        }
        .metrics-bold-block {
            font-size: 22px !important;
            font-weight: 800 !important;
            line-height: 1.7 !important;
            color: #1F2F36 !important;
        }
        .scroll-block {
            max-height: 560px;
            overflow-y: scroll !important;
            padding-right: 6px;
        }
        .chart-shell-header {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid #C8D3DE;
            border-bottom: none;
            border-radius: 16px 16px 0 0;
            padding: 10px 16px 6px 16px;
            margin-bottom: 0;
        }
        .chart-shell-title {
            grid-column: 2;
            justify-self: center;
            font-size: 26px;
            font-weight: 700;
            color: #163342;
            text-align: center;
            line-height: 1.2;
        }
        .chart-shell-legend {
            grid-column: 3;
            justify-self: end;
            display: flex;
            gap: 14px;
            align-items: center;
            font-size: 16px;
            font-weight: 700;
            color: #203746;
        }
        .chart-shell-legend-stack {
            flex-direction: column;
            align-items: flex-end;
            gap: 6px;
        }
        .chart-shell-legend-row {
            display: flex;
            gap: 14px;
            align-items: center;
            justify-content: flex-end;
            flex-wrap: wrap;
        }
        .chart-shell-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            white-space: nowrap;
        }
        .chart-shell-legend-swatch {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid rgba(22, 51, 66, 0.18);
        }
        .chart-shell-legend-line {
            display: inline-block;
            width: 18px;
            height: 0;
            border-top-width: 3px;
        }
        .chart-shell-legend-cooling {
            border-top: 3px solid #1F77B4;
        }
        .chart-shell-legend-heating {
            border-top: 3px dashed #E67E22;
        }
        div[data-testid="stVegaLiteChart"] {
            background: #FFFFFF;
            border: 1px solid #C8D3DE;
            border-top: none;
            border-radius: 0 0 16px 16px;
            padding-top: 2px;
            margin-top: 0 !important;
        }
        .scroll-block::-webkit-scrollbar,
        section[data-testid="stMain"]::-webkit-scrollbar {
            width: 11px;
            height: 11px;
        }
        .scroll-block::-webkit-scrollbar-thumb,
        section[data-testid="stMain"]::-webkit-scrollbar-thumb {
            background: #5E6A75;
            border-radius: 8px;
            border: 2px solid #DDE5EB;
        }
        .scroll-block::-webkit-scrollbar-track,
        section[data-testid="stMain"]::-webkit-scrollbar-track {
            background: #DDE5EB;
        }
        .scroll-block {
            scrollbar-width: auto;
            scrollbar-color: #5E6A75 #DDE5EB;
        }
        section[data-testid="stMain"] {
            scrollbar-width: auto;
            scrollbar-color: #5E6A75 #DDE5EB;
        }
        div[data-testid="stAlert"] {
            background: #EAF2F8 !important;
            border: 1px solid #8FB3CC !important;
            border-radius: 10px !important;
            color: #18435A !important;
        }
        div[data-testid="stAlert"] * {
            color: #18435A !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        .notice-banner {
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 8px;
            margin-bottom: 10px;
            font-weight: 700;
            font-size: 16px;
        }
        .notice-info {
            background: #EAF2F8;
            border: 1px solid #8FB3CC;
            color: #18435A;
        }
        .notice-error {
            background: #FCEDEC;
            border: 1px solid #D79A96;
            color: #8B2E28;
        }
        .metric-box {
            border-radius: 14px;
            border: 1px solid rgba(40, 86, 98, 0.18);
            background: rgba(255, 255, 255, 0.82);
            padding: 10px 12px;
            margin-bottom: 10px;
            font-size: 17px;
        }
        .metric-box-title {
            font-size: 22px;
            font-weight: 800;
            color: #1F3942;
            line-height: 1.25;
        }
        .metric-box-value {
            font-size: 32px;
            font-weight: 800;
            color: #1E3D47;
            line-height: 1.2;
            margin-top: 4px;
        }
        .config-card {
            margin-top: 8px;
            margin-bottom: 18px;
            font-size: 17px;
            font-weight: 700;
            line-height: 1.55;
        }
        .config-card strong {
            font-size: 19px;
            font-weight: 700;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            margin-right: 8px;
            font-size: 15px;
            font-weight: 700;
            border: 1px solid #98A9AD;
            background: #F8F6F2;
            color: #274047;
        }
        .status-pill.done {
            border-color: #3E8CB5;
            background: #E7F2F9;
            color: #1E5F86;
        }
        .status-pill.pending {
            border-color: #C65D57;
            background: #FCEDEC;
            color: #9E2F2A;
        }
        .best-workflow-banner {
            background: #EAF2F8;
            border: 1px solid #8FB3CC;
            color: #18435A;
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 8px;
            margin-bottom: 10px;
            font-weight: 700;
            font-size: 17px;
        }
        .workflow-best-summary {
            font-size: 17px;
            font-weight: 700;
            line-height: 1.6;
        }
        .glass strong {
            font-size: 19px;
        }
        .glass {
            font-size: 16px;
        }
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            justify-content: flex-start !important;
            text-align: left !important;
        }
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
            justify-content: flex-start !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"] > div,
        div[data-testid="stDataFrame"] [role="columnheader"] > div {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] *,
        div[data-testid="stDataFrame"] [role="columnheader"] * {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #1F2F36 !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"] {
            background: #EAF2F8 !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"] {
            background: #FFFFFF !important;
        }
        div[data-testid="stTable"] table,
        div[data-testid="stTable"] th,
        div[data-testid="stTable"] td,
        .stTable table,
        .stTable th,
        .stTable td {
            color: #1F2F36 !important;
            background: #FFFFFF !important;
            text-align: left !important;
            font-size: 17px !important;
            font-weight: 800 !important;
        }
        div[data-testid="stTable"] th,
        .stTable th {
            background: #EAF2F8 !important;
        }
        .progress-round-label {
            font-size: 20px;
            font-weight: 800;
            color: #1E3D47;
            margin-bottom: 6px;
        }
        .table-wrap {
            margin-top: 14px;
            overflow-x: auto;
        }
        .table-wrap .custom-html-table {
            width: 100%;
            border-collapse: collapse;
            background: #FFFFFF;
            border: 1px solid #D6E0E8;
            border-radius: 8px;
            overflow: hidden;
        }
        .table-wrap .custom-html-table th,
        .table-wrap .custom-html-table td {
            text-align: left;
            font-size: 22px;
            font-weight: 800;
            color: #1F2F36;
            padding: 10px 12px;
            border-bottom: 1px solid #E2E8EE;
            white-space: nowrap;
        }
        .table-wrap .custom-html-table th {
            background: #EAF2F8;
        }
        .table-wrap .custom-html-table tbody tr:nth-child(even) {
            background: #F8FBFD;
        }
        @media (max-width: 900px) {
            .flow-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_number_input_focus_fix() -> None:
    components.html(
        """
        <script>
        (function () {
            const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;
            if (!parentDoc || !parentDoc.body) {
                return;
            }

            const guardAttr = "data-numinput-autoblur";
            if (parentDoc.body.getAttribute(guardAttr) === "1") {
                return;
            }
            parentDoc.body.setAttribute(guardAttr, "1");

            const isNumberInputNode = (node) => {
                return !!(node && node.closest && node.closest('[data-testid="stNumberInput"]'));
            };

            const blurIfNeeded = () => {
                const active = parentDoc.activeElement;
                if (isNumberInputNode(active) && typeof active.blur === "function") {
                    active.blur();
                }
            };

            const onPointerRelease = (event) => {
                if (!isNumberInputNode(event.target)) {
                    return;
                }
                window.setTimeout(blurIfNeeded, 0);
            };

            const onKeyConfirm = (event) => {
                if (event.key !== "Enter" && event.key !== " ") {
                    return;
                }
                if (!isNumberInputNode(event.target)) {
                    return;
                }
                window.setTimeout(blurIfNeeded, 0);
            };

            parentDoc.addEventListener("pointerup", onPointerRelease, true);
            parentDoc.addEventListener("mouseup", onPointerRelease, true);
            parentDoc.addEventListener("touchend", onPointerRelease, true);
            parentDoc.addEventListener("keydown", onKeyConfirm, true);
        })();
        </script>
        """,
        height=0,
    )


def _render_workflow_overview(snapshot: dict[str, Any]) -> None:
    cards = []
    workflows = snapshot.get("workflows") or {}
    snapshot_status = str(snapshot.get("status", "idle") or "idle")
    is_running = snapshot_status in {"running", "starting"}
    complete_keywords = ("优化完成", "并行优化循环完成", "优化完成汇总")
    for workflow_id in _visible_workflow_ids(snapshot):
        workflow = workflows.get(workflow_id, {})
        current_iteration = int(workflow.get("current_iteration", 0) or 0)
        max_iterations = int(workflow.get("max_iterations", 0) or 0)
        progress_pct = int((float(workflow.get("progress", 0.0) or 0.0)) * 100)
        latest_status = str(workflow.get("latest_status", "") or "")
        early_stopped = bool(workflow.get("early_stopped", False))

        status_label = "未运行"
        badge_class = "badge-idle"

        if snapshot_status == "finished":
            status_label = "已完成"
            badge_class = "badge-finished"
        elif early_stopped or ("早停" in latest_status):
            status_label = "已早停"
            badge_class = "badge-stopped"
        elif any(keyword in latest_status for keyword in complete_keywords) or progress_pct >= 100:
            status_label = "已完成"
            badge_class = "badge-finished"
        elif is_running:
            status_label = "运行中"
            badge_class = "badge-running"

        cards.append(
            f"<div class='workflow-chip'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<strong>{workflow_id}</strong>"
            f"<span class='workflow-status-badge {badge_class}'>{html.escape(status_label)}</span>"
            f"</div>"
            f"<div>进度：第 {current_iteration}/{max_iterations} 轮</div>"
            f"<div>完成度：{progress_pct}%</div></div>"
        )
    if cards:
        st.markdown(f"<div class='workflow-chip-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_text_panel(
    title: str,
    payload: dict[str, Any] | None,
    placeholder: str,
    numbered: bool = False,
    normalize_log_lines: bool = False,
) -> None:
    st.markdown(f"### {title}")
    if not payload or not payload.get("text"):
        st.markdown(f"<div class='glass'>{placeholder}</div>", unsafe_allow_html=True)
        return

    if payload.get("iteration") is None:
        caption = f"更新时间 {payload.get('updated_at', '-')}"
    else:
        caption = f"更新时间 {payload.get('updated_at', '-')} | 第{payload.get('iteration', 0)}轮"
    st.caption(caption)
    text = str(payload.get("text", ""))
    if numbered:
        text = _format_reasoning_lines(text)
        _render_timestamp_log_block(text)
        return
    if normalize_log_lines:
        _render_timestamp_log_block(text)
        return
    content = html.escape(text)
    st.markdown(f"<div class='glass scroll-block'><pre class='mono-block'>{content}</pre></div>", unsafe_allow_html=True)


def _render_workflow_best_section(
    workflow_id: str,
    workflow_snapshot: dict[str, Any],
    keep_total_summary_block: bool = False,
    details_title: str = "本工作流最终参数修改详情",
    summary_log_payload: dict[str, Any] | None = None,
) -> None:
    details_payload = workflow_snapshot.get("latest_parameter_details")
    has_details = bool(details_payload and details_payload.get("text"))
    if not has_details:
        details_history = workflow_snapshot.get("parameter_details_history", []) or []
        for item in reversed(details_history):
            if item and item.get("text"):
                details_payload = item
                has_details = True
                break

    best_iteration = _pick_best_iteration(workflow_snapshot)
    if best_iteration <= 0 and not has_details:
        st.markdown("<div class='glass'>当前工作流尚未产生可比较的最优轮次。</div>", unsafe_allow_html=True)
        return

    best_metrics = _best_metrics_from_history(workflow_snapshot) or {}
    best_idf = _best_idf_from_history(workflow_snapshot, best_iteration) if best_iteration > 0 else None
    history = workflow_snapshot.get("iteration_history", []) or []
    baseline_metrics = (history[0].get("metrics", {}) if history else {})

    def _pct_str(best_value: float, baseline_value: float) -> str:
        baseline_num = float(baseline_value or 0)
        best_num = float(best_value or 0)
        pct = ((baseline_num - best_num) / baseline_num * 100.0) if baseline_num else 0.0
        return f"{pct:+.2f}%"

    st.markdown("### 当前工作流最优结果")
    if best_iteration > 0:
        st.markdown(
            f"""
            <div class='glass workflow-best-summary'>
                <strong>{workflow_id} 最优轮次：</strong> 第 {best_iteration} 轮<br/>
                <strong>最优IDF路径：</strong> {html.escape(str(best_idf or '-'))}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='glass workflow-best-summary'>
                <strong>{workflow_id} 最优轮次：</strong> 计算中<br/>
                <strong>最优IDF路径：</strong> -
            </div>
            """,
            unsafe_allow_html=True,
        )

    if best_metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _render_metric_card(
                "最优总建筑能耗(kWh)",
                f"{float(best_metrics.get('total_site_energy_kwh', 0) or 0):.2f}",
                _pct_str(best_metrics.get('total_site_energy_kwh', 0), baseline_metrics.get('total_site_energy_kwh', 0)),
            )
        with c2:
            _render_metric_card(
                "最优EUI(kWh/m²)",
                f"{float(best_metrics.get('eui_kwh_per_m2', 0) or 0):.2f}",
                _pct_str(best_metrics.get('eui_kwh_per_m2', 0), baseline_metrics.get('eui_kwh_per_m2', 0)),
            )
        with c3:
            _render_metric_card(
                "最优制冷能耗(kWh/m²)",
                f"{float(best_metrics.get('total_cooling_kwh', 0) or 0):.4f}",
                _pct_str(best_metrics.get('total_cooling_kwh', 0), baseline_metrics.get('total_cooling_kwh', 0)),
            )
        with c4:
            _render_metric_card(
                "最优供暖能耗(kWh/m²)",
                f"{float(best_metrics.get('total_heating_kwh', 0) or 0):.4f}",
                _pct_str(best_metrics.get('total_heating_kwh', 0), baseline_metrics.get('total_heating_kwh', 0)),
            )

    details_for_view = details_payload
    if details_payload and details_payload.get("text"):
        details_for_view = dict(details_payload)
        prepared_text = _prepare_parameter_details_for_view(
            str(details_payload.get("text", "")),
            keep_total_summary_block=keep_total_summary_block,
        )
        if keep_total_summary_block:
            prepared_text = _replace_last_curve_saved_line(
                prepared_text,
                _extract_parallel_curve_saved_line(summary_log_payload),
            )
        details_for_view["text"] = prepared_text

    _render_text_panel(
        details_title,
        details_for_view,
        "当前还没有抓取到本工作流最终参数修改详情。",
        normalize_log_lines=True,
    )


def _render_summary_page(snapshot: dict[str, Any], running: bool = False) -> None:
    st.subheader("当前查看：并行工作流汇总")
    all_workflows = snapshot.get("workflows") or {}
    visible_ids = _visible_workflow_ids(snapshot)
    workflows = {workflow_id: all_workflows.get(workflow_id, {}) for workflow_id in visible_ids}
    if not workflows:
        _render_notice("暂无可汇总数据。")
        return

    # 运行中时汇总页保持初始化静态视图，避免与单工作流页面内容交替渲染。
    if running:
        st.markdown("### 并行工作流汇总可视化曲线")
        _render_notice("运行中：汇总页已锁定为初始化视图，待全部工作流完成后统一更新。")
        _render_text_panel(
            "系统配置消耗量",
            None,
            "运行中：汇总结果区域暂不更新。",
            normalize_log_lines=True,
        )
        st.markdown("### 各工作流最优轮次对比")
        _render_notice("运行中：该区域暂不更新。")
        st.markdown("### 并行工作流最佳优化参数详情")
        st.markdown(
            "<div class='glass'>运行中：该区域暂不更新，全部工作流完成后统一显示最终结果。</div>",
            unsafe_allow_html=True,
        )
        return

    # Fast path for initial/empty state: render only template placeholders and
    # skip all chart/table preparation to keep the first summary-tab click instant.
    has_any_history = any((wf.get("iteration_history") or []) for wf in workflows.values())
    if not has_any_history:
        st.markdown("### 并行工作流汇总可视化曲线")
        _render_notice("尚未产生可汇总的迭代数据。启动后将自动显示曲线。")
        _render_text_panel(
            "系统配置消耗量",
            None,
            "等待并行汇总日志输出。",
            normalize_log_lines=True,
        )
        st.markdown("### 各工作流最优轮次对比")
        _render_notice("尚无可对比的最优轮次结果。")
        return

    # Only pass minimal numeric fields needed by the chart builder.
    # Strips plan_description / idf_path / other large strings so the
    # cache-key hash is near-instant on every refresh.
    curve_input = {
        wf_id: {"iteration_history": [
            {
                "iteration": item.get("iteration", 0),
                "metrics": {
                    "total_cooling_kwh": float((item.get("metrics") or {}).get("total_cooling_kwh", 0) or 0),
                    "total_heating_kwh": float((item.get("metrics") or {}).get("total_heating_kwh", 0) or 0),
                },
            }
            for item in wf.get("iteration_history", [])
        ]}
        for wf_id, wf in workflows.items()
    }
    summary_curve = _build_all_workflows_progressive_curve(curve_input)
    if summary_curve is not None:
        st.markdown("### 并行工作流汇总可视化曲线")
        _render_summary_curve_header(sorted(workflows.keys()))
        st.altair_chart(summary_curve, width='stretch')
    else:
        _render_notice("正在准备并行汇总曲线数据，生成后会自动显示完整图像。")

    summary_total_energy_curve = _build_all_workflows_total_energy_curve(workflows)
    if summary_total_energy_curve is not None:
        st.markdown("### 并行工作流总建筑能耗可视化曲线")
        _render_summary_total_energy_curve_header(sorted(workflows.keys()))
        st.altair_chart(summary_total_energy_curve, width='stretch')
    else:
        _render_notice("正在准备并行总建筑能耗曲线数据，生成后会自动显示完整图像。")

    summary_log_payload = _fallback_global_summary_log(snapshot)
    _render_text_panel(
        "系统配置消耗量",
        summary_log_payload,
        "等待并行汇总日志输出。",
        normalize_log_lines=True,
    )

    rows = []
    for workflow_id, workflow_snapshot in sorted(workflows.items()):
        best_iteration = _pick_best_iteration(workflow_snapshot)
        best_metrics = _best_metrics_from_history(workflow_snapshot) or {}
        history = workflow_snapshot.get("iteration_history", []) or []
        baseline_metrics = (history[0].get("metrics", {}) if history else {})

        def _fmt_with_pct(value: float, baseline: float) -> str:
            value_num = float(value or 0)
            baseline_num = float(baseline or 0)
            pct = ((baseline_num - value_num) / baseline_num * 100.0) if baseline_num else 0.0
            return f"{value_num:.4f} ({pct:+.2f}%)"

        rows.append(
            {
                "工作流": workflow_id,
                "最优轮次": best_iteration,
                "总建筑能耗(kWh)": _fmt_with_pct(best_metrics.get("total_site_energy_kwh", 0), baseline_metrics.get("total_site_energy_kwh", 0)),
                "EUI(kWh/m²)": _fmt_with_pct(best_metrics.get("eui_kwh_per_m2", 0), baseline_metrics.get("eui_kwh_per_m2", 0)),
                "制冷能耗(kWh/m²)": _fmt_with_pct(best_metrics.get("total_cooling_kwh", 0), baseline_metrics.get("total_cooling_kwh", 0)),
                "供暖能耗(kWh/m²)": _fmt_with_pct(best_metrics.get("total_heating_kwh", 0), baseline_metrics.get("total_heating_kwh", 0)),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        _render_notice("暂无可汇总数据。")
        return

    valid_rows = []
    for workflow_id, workflow_snapshot in sorted(workflows.items()):
        best_metrics = _best_metrics_from_history(workflow_snapshot) or {}
        total = float(best_metrics.get("total_site_energy_kwh", 0) or 0)
        if total > 0:
            valid_rows.append((workflow_id, total))

    global_best_workflow = None
    if valid_rows:
        global_best_workflow = min(valid_rows, key=lambda item: item[1])[0]

    if global_best_workflow:
        summary_df["全局最佳标识"] = summary_df["工作流"].apply(lambda wf: "✅" if wf == global_best_workflow else "")
    else:
        summary_df["全局最佳标识"] = ""

    summary_df = summary_df[
        [
            "工作流",
            "最优轮次",
            "总建筑能耗(kWh)",
            "EUI(kWh/m²)",
            "制冷能耗(kWh/m²)",
            "供暖能耗(kWh/m²)",
            "全局最佳标识",
        ]
    ]

    st.markdown("### 各工作流最优轮次对比")
    summary_display_df = _stringify_dataframe(summary_df)
    _render_html_table(summary_display_df)

    if global_best_workflow:
        st.markdown(f"<div class='best-workflow-banner'>全局最佳工作流：{global_best_workflow}</div>", unsafe_allow_html=True)
        workflow_snapshot = workflows.get(global_best_workflow, {})
        # Only show detailed best-workflow section when finished; during running
        # it shows the same heavy content as individual workflow pages which is
        # both confusing and slow to render on every auto-refresh.
        if not running:
            _render_workflow_best_section(
                global_best_workflow,
                workflow_snapshot,
                keep_total_summary_block=True,
                details_title="并行工作流最佳优化参数详情",
                summary_log_payload=summary_log_payload,
            )
    else:
        _render_notice("尚无法判断全局最佳工作流（可能仍在运行初期）。")


def _render_iteration_buttons(selected_workflow: str, workflow_snapshot: dict[str, Any], selected_iteration: int | None) -> int | None:
    max_iterations = int(workflow_snapshot.get("max_iterations", 0) or 0)
    current_iteration = int(workflow_snapshot.get("current_iteration", 0) or 0)
    if max_iterations < 0:
        return selected_iteration

    st.markdown("### 迭代轮次切换")
    total_buttons = max_iterations + 1  # 包含第0轮基准
    cols = st.columns(total_buttons)
    result = selected_iteration
    for i in range(0, total_buttons):
        done = i <= current_iteration
        label = f"{'🔵' if done else '🔴'} {i}"
        with cols[i]:
            clicked = st.button(
                label,
                key=f"iter_btn_{selected_workflow}_{i}",
                disabled=(not done),
                width="stretch",
            )
            if clicked:
                result = i
                st.session_state[f"iter_pick_{selected_workflow}"] = i
                web_state = st.session_state.get("web_ui_state")
                if isinstance(web_state, dict):
                    web_state["last_ui_action_ts"] = time.time()

    st.markdown(
        "<div style='margin-top:6px'><span class='status-pill done'>🔵 已产生该轮数据（第0轮为基准模拟）</span><span class='status-pill pending'>🔴 尚未到该轮</span></div>",
        unsafe_allow_html=True,
    )
    return result


def _render_plan_metrics_panel(payload: dict[str, Any] | None) -> None:
    st.markdown("### 建议指标")
    if not payload:
        st.markdown("<div class='glass'>等待本轮输出建议指标（丰富度/对象类别/相似度）。</div>", unsafe_allow_html=True)
        return

    lines = []
    if payload.get("unique_fields") is not None:
        lines.append(f"字段类别（对象.字段去重）：{payload['unique_fields']} 项")
    if payload.get("object_categories") is not None:
        lines.append(f"对象类别：{payload['object_categories']} 个")
    if payload.get("similarity_pct") is not None:
        lines.append(f"与上一轮字段重复率：{payload['similarity_pct']:.1f}%")
    if payload.get("overlap_count") is not None and payload.get("overlap_total") is not None:
        lines.append(f"重复字段计数：{payload['overlap_count']}/{payload['overlap_total']}")
    if not lines and payload.get("line"):
        lines.append(str(payload["line"]))

    content = "<br/>".join(html.escape(line) for line in lines)
    st.caption(f"更新时间 {payload.get('updated_at', '-')} | 第{payload.get('iteration', 0)}轮")
    st.markdown(f"<div class='glass metrics-bold-block'>{content}</div>", unsafe_allow_html=True)


def _render_round_stats_panel(payload: dict[str, Any] | None) -> None:
    st.markdown("### 本轮修改统计")
    if not payload:
        st.markdown("<div class='glass'>等待本轮修改统计（修改对象总数/修改字段总数）。</div>", unsafe_allow_html=True)
        return

    obj_count = payload.get("modified_objects")
    field_count = payload.get("modified_fields")
    st.caption(f"更新时间 {payload.get('updated_at', '-')} | 第{payload.get('iteration', 0)}轮")
    st.markdown(
        """
        <div class='glass'>
            <div class='metric-box'><div class='metric-box-title'>修改对象总数</div><div class='metric-box-value'>{obj}</div></div>
            <div class='metric-box'><div class='metric-box-title'>修改字段总数</div><div class='metric-box-value'>{field}</div></div>
        </div>
        """.format(
            obj=(obj_count if obj_count is not None else "-"),
            field=(field_count if field_count is not None else "-"),
        ),
        unsafe_allow_html=True,
    )


def _render_baseline_panel(workflow_snapshot: dict[str, Any]) -> None:
    st.markdown("### 第0轮（基准模拟）")
    baseline_payload = workflow_snapshot.get("baseline_log")
    if not baseline_payload or not baseline_payload.get("text"):
        st.markdown("<div class='glass'>等待基准模拟日志输出（initial_baseline）。</div>", unsafe_allow_html=True)
        return

    st.caption(f"更新时间 {baseline_payload.get('updated_at', '-')} | 第0轮")
    baseline_text = str(baseline_payload.get("text", ""))
    baseline_lines = _split_timestamp_log_lines(baseline_text)
    if baseline_lines and _strip_log_prefix(baseline_lines[0]) == "":
        baseline_text = _normalize_timestamp_log_lines("\n".join(baseline_lines[1:]))
    _render_timestamp_log_block(baseline_text)


def _pick_iteration_payload(history: list[dict[str, Any]], iteration: int) -> dict[str, Any] | None:
    for item in history:
        if int(item.get("iteration", 0) or 0) == int(iteration):
            return item
    return None


def _save_uploaded_file(uploaded_file, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, uploaded_file.name)
    with open(target_path, "wb") as out_file:
        out_file.write(uploaded_file.getbuffer())
    return target_path


def _render_runtime_hint(status: str, workflow_snapshot: dict[str, Any]) -> None:
    if status == "idle":
        _render_notice("等待启动：请先上传 IDF 与 API 文件，然后点击启动按钮。")
        return
    if status == "starting":
        _render_notice("准备中：正在初始化并行工作流与日志捕获。")
        return
    if status == "error":
        _render_notice("运行中断：可查看上方错误信息并重新启动。", kind="error")
        return

    current_iteration = int(workflow_snapshot.get("current_iteration", 0) or 0)
    max_iterations = int(workflow_snapshot.get("max_iterations", DEFAULT_MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS)
    latest_status = str(workflow_snapshot.get("latest_status", "正在处理中...") or "正在处理中...")
    status_time = workflow_snapshot.get("status_updated_at")
    prefix = f"第 {current_iteration} 轮（最大 {max_iterations} 轮，早停自动结束）"
    if status_time:
        _render_notice(f"{prefix} | {latest_status} | 更新时间 {status_time}")
    else:
        _render_notice(f"{prefix} | {latest_status}")


def _init_page_state() -> None:
    if "web_ui_state" not in st.session_state:
        st.session_state["web_ui_state"] = {
            "snapshot_path": None,
            "runner_thread": None,
            "selected_workflow": "workflow_1",
            "selected_view": "workflow_1",
            "runtime_state": None,
            "uploaded_idf": None,
            "uploaded_api": None,
            "last_ui_action_ts": 0.0,
            "auto_summary_marker": None,
            "summary_live_marker": None,
            "summary_last_rerun_ts": 0.0,
        }


def _on_view_nav_click(view_name: str, default_workflow: str) -> None:
    page_state = st.session_state.get("web_ui_state")
    if not isinstance(page_state, dict):
        return
    page_state["selected_view"] = view_name
    if view_name != "summary":
        page_state["selected_workflow"] = view_name
    else:
        page_state["selected_workflow"] = page_state.get("selected_workflow", default_workflow)
    page_state["last_ui_action_ts"] = time.time()


def _install_summary_finish_watcher(snapshot_path: str | None, selected_view: str, running: bool) -> None:
    """When user stays on summary, refresh only when snapshot changes.

    - While running/starting: rerun only if snapshot updated_at changed, so top overview
      can keep progressing without forcing constant reruns.
    - When finished/error: trigger one final rerun to render complete summary content.
    """
    if not snapshot_path or selected_view != "summary" or not running:
        return

    fragment_api = getattr(st, "fragment", None)
    if not callable(fragment_api):
        return

    interval = max(float(AUTO_REFRESH_SECONDS), 0.5)

    @fragment_api(run_every=f"{interval}s")
    def _summary_finish_poller() -> None:
        latest_snapshot = _read_snapshot(snapshot_path)
        latest_status = str(latest_snapshot.get("status", "idle") or "idle")
        state = st.session_state.get("web_ui_state")
        if not isinstance(state, dict):
            return

        if latest_status in {"running", "starting"}:
            live_marker = latest_snapshot.get("updated_at") or latest_status
            if state.get("summary_live_marker") != live_marker:
                state["summary_live_marker"] = live_marker
                now_ts = time.time()
                last_rerun_ts = state.get("summary_last_rerun_ts", 0.0) or 0.0
                if now_ts - last_rerun_ts >= 2.0:
                    state["summary_last_rerun_ts"] = now_ts
                    st.rerun()
                    return
            st.markdown("<div style='display:none'></div>", unsafe_allow_html=True)
            return

        if latest_status in {"finished", "error"}:
            finish_marker = latest_snapshot.get("finished_at") or latest_status
            if state.get("auto_summary_marker") == finish_marker:
                return

            state["selected_view"] = "summary"
            state["auto_summary_marker"] = finish_marker
            st.rerun()

    _summary_finish_poller()


def main() -> None:
    st.set_page_config(page_title="EnergyPlus 并行工作流可视化", layout="wide")
    _init_page_state()
    _render_style()
    _render_number_input_focus_fix()

    page_state = st.session_state["web_ui_state"]
    snapshot = _read_snapshot(page_state.get("snapshot_path"))
    status = snapshot.get("status", "idle")
    running = status in {"running", "starting"}
    _install_summary_finish_watcher(
        snapshot_path=page_state.get("snapshot_path"),
        selected_view=str(page_state.get("selected_view", "workflow_1")),
        running=running,
    )

    st.title("EnergyPlus 并行工作流可视化Web界面")
    st.caption("Version 1.0 | Powered by Streamlit | By OpenAI")

    st.markdown(
        """
        <div class='glass'>
            <strong>工作流运行逻辑：</strong>
            <div class='flow-grid'>
                <div class='flow-item'><strong>1. 基准模拟</strong>以初始 IDF 运行 EnergyPlus，得到第一轮基线结果。</div>
                <div class='flow-item'><strong>2. 指标提取</strong>读取模拟输出，提取总能耗、EUI、制冷、供暖等关键指标。</div>
                <div class='flow-item'><strong>3. LLM优化分析</strong>调用原有 LLM 推理逻辑，生成本轮优化建议。</div>
                <div class='flow-item'><strong>4. 应用优化修改</strong>按原始规则修改 IDF，只在网页展示修改摘要。</div>
                <div class='flow-item'><strong>5. 迭代验证</strong>用新 IDF 继续模拟，逐轮更新曲线和能耗数值。</div>
                <div class='flow-item'><strong>6. 多工作流汇总</strong>并行对比不同工作流，支持 1/2/3... 切换查看。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("运行配置与说明", expanded=(status == "idle")):
        _render_notice("第0轮为基准模拟（不做优化）；运行采用早停机制，实际优化轮次会根据目标节能率和收敛情况自动决定。")
        col_left, col_right = st.columns(2)
        with col_left:
            uploaded_idf = st.file_uploader("上传 IDF 模型文件", type=["idf"], key="idf_uploader", help="必填。上传后将自动复制到运行目录，不改动原始算法逻辑。")
        with col_right:
            uploaded_api = st.file_uploader("上传 API Key 文本文件", type=["txt"], key="api_uploader", help="必填。内容应为可用密钥，等价于原来的 api_key.txt。")

        col_cfg_1, col_cfg_2, col_cfg_3 = st.columns(3)
        with col_cfg_1:
            num_workflows = st.number_input(
                "并行工作流数量",
                min_value=1,
                value=int(snapshot.get("config", {}).get("num_workflows", DEFAULT_NUM_WORKFLOWS) or DEFAULT_NUM_WORKFLOWS),
                step=1,
                key="cfg_num_workflows",
            )
        with col_cfg_2:
            max_iterations_cap = st.number_input(
                "最大优化迭代轮次",
                min_value=1,
                value=int(snapshot.get("config", {}).get("max_iterations_cap", snapshot.get("config", {}).get("optimization_rounds", DEFAULT_MAX_ITERATIONS)) or DEFAULT_MAX_ITERATIONS),
                step=1,
                key="cfg_max_iterations_cap",
            )
        with col_cfg_3:
            early_stop_target_total_saving_pct = st.number_input(
                "最大节能幅度目标(%)",
                min_value=0.1,
                max_value=99.9,
                value=float(snapshot.get("config", {}).get("early_stop_target_total_saving_pct", DEFAULT_EARLY_STOP_TARGET_SAVING_PCT) or DEFAULT_EARLY_STOP_TARGET_SAVING_PCT),
                step=0.5,
                format="%.1f",
                key="cfg_early_stop_target_total_saving_pct",
            )

        st.markdown(
            f"""
            <div class='glass config-card'>
                <strong>固定配置</strong><br/>
                IDD: {DEFAULT_IDD_PATH}<br/>
                EPW: {DEFAULT_EPW_PATH}<br/>
                日志目录: {DEFAULT_LOG_DIR}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        start_disabled = running or (page_state.get("runner_thread") is not None and page_state["runner_thread"].is_alive())
        if st.button("启动并行优化并实时可视化", type="primary", disabled=start_disabled):
            if not uploaded_idf or not uploaded_api:
                _render_notice("请先上传 IDF 文件和 API Key 文件。", kind="error")
                return

            os.makedirs(RUNTIME_DIR, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(RUNTIME_DIR, f"dashboard_snapshot_{run_id}.json")
            upload_dir = os.path.join(RUNTIME_DIR, "uploads", run_id)
            idf_path = _save_uploaded_file(uploaded_idf, upload_dir)
            api_key_path = _save_uploaded_file(uploaded_api, upload_dir)
            config = {
                "idf_path": idf_path,
                "idd_path": DEFAULT_IDD_PATH,
                "api_key_path": api_key_path,
                "epw_path": DEFAULT_EPW_PATH,
                "log_dir": DEFAULT_LOG_DIR,
                "max_iterations": int(max_iterations_cap) + 1,
                "optimization_rounds": int(max_iterations_cap),
                "max_iterations_cap": int(max_iterations_cap),
                "early_stop_target_total_saving_pct": float(early_stop_target_total_saving_pct),
                "num_workflows": int(num_workflows),
            }
            runtime_state = {"status": "starting", "started_at": None, "finished_at": None, "error": None}
            _write_snapshot(snapshot_path, _build_empty_snapshot(config=config, status="starting"))
            runner_thread = threading.Thread(
                target=_run_optimizer_in_thread,
                args=(snapshot_path, config, runtime_state),
                daemon=True,
            )
            page_state["snapshot_path"] = snapshot_path
            page_state["runner_thread"] = runner_thread
            page_state["runtime_state"] = runtime_state
            page_state["selected_workflow"] = "workflow_1"
            runner_thread.start()
            st.rerun()

    top1, top2, top3, top4 = st.columns(4)
    status_map = {"idle": "未运行", "starting": "准备中", "running": "运行中", "finished": "已完成", "error": "异常中断"}
    with top1:
        _render_metric_card("当前状态", status_map.get(status, status))
    with top2:
        _render_metric_card("开始时间", str(snapshot.get("started_at") or "-"))
    with top3:
        _render_metric_card("结束时间", str(snapshot.get("finished_at") or "-"))
    with top4:
        _render_metric_card("当前工作流数", str(snapshot.get("config", {}).get("num_workflows", 0)))

    if snapshot.get("error"):
        _render_notice(f"运行失败：{snapshot['error']}", kind="error")

    st.markdown("### 工作流总览")
    _render_workflow_overview(snapshot)

    workflow_keys = _visible_workflow_ids(snapshot)
    if not workflow_keys:
        _render_notice("点击上方按钮后，页面会实时显示每条工作流的推理过程、修改摘要、进度和能耗曲线。")
        return

    view_options = list(workflow_keys) + ["summary"]

    if status in {"finished", "error"}:
        finish_marker = snapshot.get("finished_at") or status
        if page_state.get("auto_summary_marker") != finish_marker:
            page_state["selected_view"] = "summary"
            page_state["auto_summary_marker"] = finish_marker

    if page_state.get("selected_view") not in view_options:
        page_state["selected_view"] = workflow_keys[0]

    nav_cols = st.columns(len(view_options))
    for idx, view_name in enumerate(view_options):
        label = "汇总" if view_name == "summary" else f"工作流{view_name.split('_')[-1]}"
        is_active = page_state.get("selected_view") == view_name
        button_label = f"▶ {label}" if is_active else label
        with nav_cols[idx]:
            st.button(
                button_label,
                key=f"view_nav_{view_name}",
                width="stretch",
                on_click=_on_view_nav_click,
                args=(view_name, workflow_keys[0]),
            )

    # ── 缓存预热 ────────────────────────────────────────────────────────────
    # 每次刷新都以精简格式预先计算汇总图（仅数值字段，哈希极快），
    # 确保用户首次点击 "汇总" 标签时直接命中缓存，零延迟。
    _wf_map_prewarm = snapshot.get("workflows") or {}
    has_prewarm_data = any((_wf_map_prewarm.get(_wf_id, {}).get("iteration_history") or []) for _wf_id in workflow_keys)
    if has_prewarm_data:
        _prewarm_curve_input = {
            _wf_id: {"iteration_history": [
                {
                    "iteration": _it.get("iteration", 0),
                    "metrics": {
                        "total_cooling_kwh": float((_it.get("metrics") or {}).get("total_cooling_kwh", 0) or 0),
                        "total_heating_kwh": float((_it.get("metrics") or {}).get("total_heating_kwh", 0) or 0),
                    },
                }
                for _it in _wf_map_prewarm.get(_wf_id, {}).get("iteration_history", [])
            ]}
            for _wf_id in workflow_keys
        }
        _build_all_workflows_progressive_curve(_prewarm_curve_input)
    # ────────────────────────────────────────────────────────────────────────

    if page_state.get("selected_view") == "summary":
        _render_summary_page(snapshot, running=running)
        return

    # 仅当明确不是汇总页面时，才进入工作流页面逻辑
    if page_state.get("selected_view") == "summary":
        # 这是一个双重检查，防止任何情况下工作流代码被执行
        return

    selected_workflow = str(page_state.get("selected_view"))
    if selected_workflow not in workflow_keys:
        selected_workflow = workflow_keys[0]
        page_state["selected_view"] = selected_workflow
    workflow_snapshot = (snapshot.get("workflows") or {}).get(selected_workflow, {}) or {}

    st.subheader(f"当前查看：{selected_workflow}")
    round_text = f"第 {workflow_snapshot.get('current_iteration', 0)} 轮（最大 {workflow_snapshot.get('max_iterations', 0)} 轮）"
    st.markdown(f"<div class='progress-round-label'>{html.escape(round_text)}</div>", unsafe_allow_html=True)
    st.progress(min(max(float(workflow_snapshot.get("progress", 0.0) or 0.0), 0.0), 1.0))

    _render_runtime_hint(status, workflow_snapshot)

    reasoning_history = workflow_snapshot.get("reasoning_history", []) or []
    summary_history = workflow_snapshot.get("summary_history", []) or []
    plan_metrics_history = workflow_snapshot.get("plan_metrics_history", []) or []
    round_stats_history = workflow_snapshot.get("round_stats_history", []) or []
    parameter_details_history = workflow_snapshot.get("parameter_details_history", []) or []
    baseline_log_history = workflow_snapshot.get("baseline_log_history", []) or []
    history_iterations = sorted(
        {
            int(item.get("iteration", 0) or 0)
            for item in reasoning_history + summary_history + plan_metrics_history + round_stats_history + parameter_details_history + baseline_log_history
            if int(item.get("iteration", 0) or 0) >= 0
        }
    )

    if not history_iterations:
        history_iterations = sorted({int(item.get("iteration", 0) or 0) for item in workflow_snapshot.get("iteration_history", []) if int(item.get("iteration", 0) or 0) >= 0})

    selected_iteration = 0
    if history_iterations:
        iteration_key = f"iter_pick_{selected_workflow}"
        if iteration_key not in st.session_state or int(st.session_state[iteration_key]) not in history_iterations:
            st.session_state[iteration_key] = history_iterations[-1]

        selected_iteration = _render_iteration_buttons(
            selected_workflow,
            workflow_snapshot,
            int(st.session_state.get(iteration_key, history_iterations[-1])),
        )
        if selected_iteration not in history_iterations:
            selected_iteration = history_iterations[-1]
            st.session_state[iteration_key] = selected_iteration

        selected_reasoning = _pick_iteration_payload(reasoning_history, int(selected_iteration))
        selected_summary = _pick_iteration_payload(summary_history, int(selected_iteration))
        selected_plan_metrics = _pick_iteration_payload(plan_metrics_history, int(selected_iteration))
        selected_round_stats = _pick_iteration_payload(round_stats_history, int(selected_iteration))
        selected_parameter_details = _pick_iteration_payload(parameter_details_history, int(selected_iteration))
        selected_baseline_log = _pick_iteration_payload(baseline_log_history, int(selected_iteration))
    else:
        selected_reasoning = workflow_snapshot.get("latest_reasoning")
        selected_summary = workflow_snapshot.get("latest_summary")
        selected_plan_metrics = workflow_snapshot.get("latest_plan_metrics")
        selected_round_stats = workflow_snapshot.get("latest_round_stats")
        selected_parameter_details = workflow_snapshot.get("latest_parameter_details")
        selected_baseline_log = workflow_snapshot.get("baseline_log")

    if int(selected_iteration or 0) == 0:
        _render_baseline_panel({"baseline_log": selected_baseline_log})
    else:
        _render_text_panel("LLM推理过程", selected_reasoning, "等待当前工作流产出最终采用的 LLM 推理过程。", numbered=True)
        _render_text_panel(
            "修改摘要",
            selected_summary,
            "等待当前工作流产出成功执行后的修改摘要。",
            normalize_log_lines=True,
        )
        metrics_left, metrics_right = st.columns(2)
        with metrics_left:
            _render_plan_metrics_panel(selected_plan_metrics)
        with metrics_right:
            _render_round_stats_panel(selected_round_stats)

    iteration_history = workflow_snapshot.get("iteration_history", []) or []
    dataframe = _build_metrics_dataframe(iteration_history)

    st.markdown("### 实时可视化曲线")
    progressive_chart = _build_progressive_curve(dataframe, selected_workflow)
    if progressive_chart is not None:
        st.markdown(
            f"""
            <div class='chart-shell-header'>
                <div></div>
                <div class='chart-shell-title'>{html.escape(selected_workflow)} 冷暖能耗迭代曲线</div>
                <div class='chart-shell-legend'>
                    <span class='chart-shell-legend-item'><span class='chart-shell-legend-line chart-shell-legend-cooling'></span>制冷能耗</span>
                    <span class='chart-shell-legend-item'><span class='chart-shell-legend-line chart-shell-legend-heating'></span>供暖能耗</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(progressive_chart, width='stretch')
    elif running:
        _render_notice("正在准备曲线图数据，生成后会自动显示完整图像。")
    else:
        _render_notice("尚未开始运行，启动后将在此显示实时曲线。")

    st.markdown("### 总建筑能耗变化曲线")
    total_energy_chart = _build_total_energy_curve(dataframe, selected_workflow)
    if total_energy_chart is not None:
        st.markdown(
            f"""
            <div class='chart-shell-header'>
                <div></div>
                <div class='chart-shell-title'>{html.escape(selected_workflow)} 总建筑能耗迭代曲线</div>
                <div class='chart-shell-legend'>
                    <span class='chart-shell-legend-item'><span class='chart-shell-legend-line' style='border-top: 3px solid #2A9D8F;'></span>总建筑能耗</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(total_energy_chart, width='stretch')
    elif running:
        _render_notice("正在准备总建筑能耗曲线数据，生成后会自动显示完整图像。")
    else:
        _render_notice("尚未开始运行，启动后将在此显示总建筑能耗曲线。")

    st.markdown("### 各项能耗指标与节能变化")
    if dataframe.empty:
        _render_notice("第一轮模拟完成后，这里会自动出现四个核心指标数据。")
    else:
        selected_rows = dataframe[dataframe["轮次"] == int(selected_iteration or 0)]
        current_row = selected_rows.iloc[-1] if not selected_rows.empty else dataframe.iloc[-1]
        best_iteration = _pick_best_iteration(workflow_snapshot)
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        with metric_1:
            _render_metric_card("总建筑能耗(kWh)", f"{current_row['总建筑能耗(kWh)']:.2f}", f"{current_row['总建筑能耗节能幅度(%)']:+.2f}%")
        with metric_2:
            _render_metric_card("EUI(kWh/m²)", f"{current_row['EUI(kWh/m²)']:.2f}", f"{current_row['EUI节能幅度(%)']:+.2f}%")
        with metric_3:
            _render_metric_card("制冷能耗(kWh/m²)", f"{current_row['制冷能耗(kWh/m²)']:.2f}", f"{current_row['制冷节能幅度(%)']:+.2f}%")
        with metric_4:
            _render_metric_card("供暖能耗(kWh/m²)", f"{current_row['供暖能耗(kWh/m²)']:.2f}", f"{current_row['供暖节能幅度(%)']:+.2f}%")

        display_df = dataframe.copy()
        display_df["总建筑能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['总建筑能耗(kWh)']:.2f} ({row['总建筑能耗节能幅度(%)']:+.2f}%)", axis=1)
        display_df["EUI(含节能幅度)"] = display_df.apply(lambda row: f"{row['EUI(kWh/m²)']:.2f} ({row['EUI节能幅度(%)']:+.2f}%)", axis=1)
        display_df["制冷能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['制冷能耗(kWh/m²)']:.2f} ({row['制冷节能幅度(%)']:+.2f}%)", axis=1)
        display_df["供暖能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['供暖能耗(kWh/m²)']:.2f} ({row['供暖节能幅度(%)']:+.2f}%)", axis=1)
        display_df["最优迭代标识"] = display_df["轮次"].apply(lambda value: "✅" if int(value) == int(best_iteration) else "")
        display_df = display_df[
            [
                "轮次",
                "总建筑能耗(含节能幅度)",
                "EUI(含节能幅度)",
                "制冷能耗(含节能幅度)",
                "供暖能耗(含节能幅度)",
                "最优迭代标识",
            ]
        ]
        number_columns = [column for column in display_df.columns if column != "轮次"]
        if number_columns:
            for column in number_columns:
                if pd.api.types.is_numeric_dtype(display_df[column]):
                    display_df[column] = display_df[column].round(4)
        display_df = _stringify_dataframe(display_df)
        _render_html_table(display_df)

    _render_workflow_best_section(selected_workflow, workflow_snapshot, keep_total_summary_block=False)

    if running:
        time.sleep(AUTO_REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        has_context = get_script_run_ctx() is not None
    except Exception:
        has_context = False

    if not has_context:
        print("检测到当前不是 Streamlit 页面运行方式。")
        print("请在项目目录中使用以下命令启动：")
        print("  streamlit run web_visualization.py")
        sys.exit(0)

    main()