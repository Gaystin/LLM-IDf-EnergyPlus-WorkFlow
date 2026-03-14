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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import EnergyPlusOptimizer


RUNTIME_DIR = os.path.join(CURRENT_DIR, ".web_runtime")
WORKFLOW_REGEX = re.compile(r"workflow_(\d+)", flags=re.IGNORECASE)
ROUND_REGEX = re.compile(r"【第(\d+)轮/(\d+)】")
REASONING_HEADER_REGEX = re.compile(r"【LLM推理过程\s*-\s*(.*?)】")
PERCENT_REGEX = re.compile(r"([+-]?\d+(?:\.\d+)?)%")

DEFAULT_IDD_PATH = "Energy+.idd"
DEFAULT_EPW_PATH = "weather.epw"
DEFAULT_LOG_DIR = "optimization_logs_并行2_web"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_NUM_WORKFLOWS = 2


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
        "latest_status": "等待启动",
        "status_updated_at": None,
        "collecting_summary": False,
        "summary_lines": [],
    }


def _build_empty_snapshot(config: dict[str, Any] | None = None, status: str = "idle", error: str | None = None) -> dict[str, Any]:
    config = config or {}
    max_iterations = int(config.get("max_iterations", 5) or 5)
    num_workflows = int(config.get("num_workflows", 2) or 2)
    workflows = {
        f"workflow_{index + 1}": {
            "current_iteration": 0,
            "max_iterations": max_iterations,
            "progress": 0.0,
            "latest_reasoning": None,
            "latest_summary": None,
            "reasoning_history": [],
            "summary_history": [],
            "latest_plan_metrics": None,
            "plan_metrics_history": [],
            "latest_round_stats": None,
            "round_stats_history": [],
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
            "num_workflows": num_workflows,
        },
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


def _clean_summary_lines(lines: list[str]) -> str:
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("=") or stripped.startswith("-"):
            continue
        if stripped == "【修改摘要】汇总 - 快速查看修改概览":
            continue
        if stripped.startswith("[计划修改]") or stripped.startswith("[文本替换"):
            continue
        if "警告" in stripped or stripped.startswith("⚠"):
            continue
        if not stripped:
            if cleaned and not prev_blank:
                cleaned.append("")
                prev_blank = True
            continue

        if stripped.startswith("▶") and cleaned and not prev_blank:
            cleaned.append("")

        cleaned.append(stripped)
        prev_blank = False

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned)


def _format_reasoning_lines(text: str) -> str:
    raw_lines = [line.strip() for line in str(text or "").splitlines()]
    items = []
    for line in raw_lines:
        if not line:
            continue
        normalized = re.sub(r"^\d+[\.|、|\)]\s*", "", line)
        items.append(normalized)
    if not items:
        return ""
    return "\n".join([f"{index}. {line}" for index, line in enumerate(items, start=1)])


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


def _extract_int_after_label(text: str, label: str) -> int | None:
    pattern = re.escape(label) + r"\s*(\d+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


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
                "iteration": int(item.get("iteration", 0) or 0),
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

    for workflow_id in sorted(optimizer.workflows.keys()):
        workflow_data = optimizer.workflows.get(workflow_id, {}) or {}
        history = _simplify_iteration_history(workflow_data.get("iteration_history", []) or [])
        capture = capture_state.get(workflow_id, _build_default_capture_state())
        current_iteration = int(capture.get("current_iteration", 0) or 0)
        if not current_iteration and history:
            current_iteration = int(history[-1].get("iteration", 0) or 0)
        max_iterations = int(capture.get("max_iterations", config.get("max_iterations", 5)) or config.get("max_iterations", 5))
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
            "latest_status": capture.get("latest_status", "等待日志输出"),
            "status_updated_at": capture.get("status_updated_at"),
            "iteration_history": history,
            "best_metrics": workflow_data.get("best_metrics"),
            "best_iteration": int(workflow_data.get("best_iteration", 0) or 0),
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

        workflow_id = self._resolve_workflow_id(record, message)
        if not workflow_id:
            return

        with self.capture_lock:
            workflow_state = self.capture_state.setdefault(workflow_id, _build_default_capture_state())
            workflow_state["latest_status"] = _short_status(message)
            workflow_state["status_updated_at"] = datetime.now().strftime("%H:%M:%S")

            round_match = ROUND_REGEX.search(message)
            if round_match:
                workflow_state["current_iteration"] = int(round_match.group(1))
                workflow_state["max_iterations"] = int(round_match.group(2))

            reasoning_match = REASONING_HEADER_REGEX.search(message)
            if reasoning_match and "最终采用方案" in reasoning_match.group(1):
                lines = [line for line in message.splitlines()[1:] if line.strip()]
                reasoning_text = "\n".join(lines).strip()
                if reasoning_text:
                    reasoning_payload = {
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                        "iteration": workflow_state.get("current_iteration", 0),
                        "text": reasoning_text,
                    }
                    workflow_state["latest_reasoning"] = reasoning_payload
                    _upsert_iteration_payload(workflow_state.setdefault("reasoning_history", []), reasoning_payload)

            if "【建议指标】" in message:
                metrics_payload = {
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                    "iteration": workflow_state.get("current_iteration", 0),
                    **_parse_plan_metrics_line(message),
                }
                workflow_state["latest_plan_metrics"] = metrics_payload
                _upsert_iteration_payload(workflow_state.setdefault("plan_metrics_history", []), metrics_payload)

            if "【本轮修改统计】" in message:
                stats_payload = {
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                    "iteration": workflow_state.get("current_iteration", 0),
                    "modified_objects": None,
                    "modified_fields": None,
                }
                workflow_state["latest_round_stats"] = stats_payload
                _upsert_iteration_payload(workflow_state.setdefault("round_stats_history", []), stats_payload)

            if "修改对象总数:" in message or "修改字段总数:" in message:
                stats_payload = dict(workflow_state.get("latest_round_stats") or {})
                stats_payload["updated_at"] = datetime.now().strftime("%H:%M:%S")
                stats_payload["iteration"] = workflow_state.get("current_iteration", 0)
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
                            "iteration": workflow_state.get("current_iteration", 0),
                            "text": summary_text,
                        }
                        workflow_state["latest_summary"] = summary_payload
                        _upsert_iteration_payload(workflow_state.setdefault("summary_history", []), summary_payload)
                    workflow_state["collecting_summary"] = False
                    workflow_state["summary_lines"] = []
                else:
                    workflow_state["summary_lines"].extend(message.splitlines())

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
        .mark_line(point=alt.OverlayMarkDef(size=88, filled=True), strokeWidth=2.4)
        .encode(
            x=alt.X("轮次:O", title="迭代轮次", sort=rounds, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("数值:Q", title="能耗 (kWh/m²)"),
            color=alt.Color(
                "指标:N",
                scale=alt.Scale(domain=["制冷能耗", "供暖能耗"], range=["#2F80ED", "#F2994A"]),
            ),
            strokeDash=alt.StrokeDash(
                "指标:N",
                scale=alt.Scale(domain=["制冷能耗", "供暖能耗"], range=[[1, 0], [6, 4]]),
            ),
            tooltip=["轮次:O", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
        )
        .properties(title=f"{workflow_id} 冷暖能耗迭代曲线", width=520, height=220)
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=15, color="#204051")
    )


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
            font-family: 'Noto Sans SC', 'Space Grotesk', sans-serif;
        }
        .stMarkdown p, .stCaption, label, .stRadio label, .stSelectbox label, .stFileUploader label {
            color: #2C3A3F !important;
        }
        .stButton button {
            background: #F6F3EC;
            border: 1px solid #C5B79A;
            color: #2C3A3F;
            border-radius: 10px;
            font-weight: 600;
        }
        .stButton button:hover {
            border-color: #6A8D92;
            color: #1E3D47;
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #2D6A8E, #3E8CB5);
            color: #FFFFFF;
            border-color: #2D6A8E;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(52, 103, 115, 0.15);
            border-radius: 12px;
            padding: 8px 10px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #2C3A3F !important;
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
            font-size: 13px;
        }
        .flow-item strong {
            display: block;
            margin-bottom: 6px;
        }
        .workflow-chip-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 10px;
        }
        .workflow-chip {
            border-radius: 14px;
            padding: 12px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(0, 95, 115, 0.14);
        }
        .mono-block {
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.55;
            margin: 0;
        }
        .scroll-block {
            max-height: 340px;
            overflow-y: auto;
            padding-right: 6px;
        }
        .metric-box {
            border-radius: 14px;
            border: 1px solid rgba(40, 86, 98, 0.18);
            background: rgba(255, 255, 255, 0.82);
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            margin-right: 8px;
            font-size: 12px;
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
        @media (max-width: 900px) {
            .flow-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow_overview(snapshot: dict[str, Any]) -> None:
    cards = []
    for workflow_id, workflow in sorted((snapshot.get("workflows") or {}).items()):
        current_iteration = int(workflow.get("current_iteration", 0) or 0)
        max_iterations = int(workflow.get("max_iterations", 0) or 0)
        progress_pct = int((float(workflow.get("progress", 0.0) or 0.0)) * 100)
        cards.append(
            f"<div class='workflow-chip'><strong>{workflow_id}</strong>"
            f"<div>进度：第 {current_iteration}/{max_iterations} 轮</div>"
            f"<div>完成度：{progress_pct}%</div></div>"
        )
    if cards:
        st.markdown(f"<div class='workflow-chip-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_text_panel(title: str, payload: dict[str, Any] | None, placeholder: str, numbered: bool = False) -> None:
    st.markdown(f"### {title}")
    if not payload or not payload.get("text"):
        st.markdown(f"<div class='glass'>{placeholder}</div>", unsafe_allow_html=True)
        return

    caption = f"更新时间 {payload.get('updated_at', '-')} | 第{payload.get('iteration', 0)}轮"
    st.caption(caption)
    text = str(payload.get("text", ""))
    if numbered:
        text = _format_reasoning_lines(text)
    content = html.escape(text)
    st.markdown(f"<div class='glass scroll-block'><pre class='mono-block'>{content}</pre></div>", unsafe_allow_html=True)


def _render_workflow_best_section(workflow_id: str, workflow_snapshot: dict[str, Any]) -> None:
    best_iteration = _pick_best_iteration(workflow_snapshot)
    if best_iteration <= 0:
        st.markdown("<div class='glass'>当前工作流尚未产生可比较的最优轮次。</div>", unsafe_allow_html=True)
        return

    best_metrics = _best_metrics_from_history(workflow_snapshot) or {}
    best_idf = _best_idf_from_history(workflow_snapshot, best_iteration)
    summary_payload = _pick_iteration_payload(workflow_snapshot.get("summary_history", []) or [], best_iteration)

    st.markdown("### 当前工作流最优结果")
    st.markdown(
        f"""
        <div class='glass'>
            <strong>{workflow_id} 最优轮次：</strong> 第 {best_iteration} 轮<br/>
            <strong>最优IDF路径：</strong> {html.escape(str(best_idf or '-'))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if best_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最优总建筑能耗(kWh)", f"{float(best_metrics.get('total_site_energy_kwh', 0) or 0):.2f}")
        c2.metric("最优EUI(kWh/m²)", f"{float(best_metrics.get('eui_kwh_per_m2', 0) or 0):.2f}")
        c3.metric("最优制冷能耗(kWh/m²)", f"{float(best_metrics.get('total_cooling_kwh', 0) or 0):.4f}")
        c4.metric("最优供暖能耗(kWh/m²)", f"{float(best_metrics.get('total_heating_kwh', 0) or 0):.4f}")

    _render_text_panel("最优轮次对应修改摘要", summary_payload, "当前还没有抓取到最优轮次的修改摘要。")


def _render_summary_page(snapshot: dict[str, Any]) -> None:
    st.subheader("汇总")
    workflows = snapshot.get("workflows") or {}
    if not workflows:
        st.info("暂无可汇总数据。")
        return

    rows = []
    for workflow_id, workflow_snapshot in sorted(workflows.items()):
        best_iteration = _pick_best_iteration(workflow_snapshot)
        best_metrics = _best_metrics_from_history(workflow_snapshot) or {}
        rows.append(
            {
                "工作流": workflow_id,
                "最优轮次": best_iteration,
                "总建筑能耗(kWh)": float(best_metrics.get("total_site_energy_kwh", 0) or 0),
                "EUI(kWh/m²)": float(best_metrics.get("eui_kwh_per_m2", 0) or 0),
                "制冷能耗(kWh/m²)": float(best_metrics.get("total_cooling_kwh", 0) or 0),
                "供暖能耗(kWh/m²)": float(best_metrics.get("total_heating_kwh", 0) or 0),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        st.info("暂无可汇总数据。")
        return

    valid_df = summary_df[summary_df["总建筑能耗(kWh)"] > 0]
    global_best_workflow = None
    if not valid_df.empty:
        global_best_workflow = valid_df.loc[valid_df["总建筑能耗(kWh)"].idxmin(), "工作流"]

    st.markdown("### 各工作流最优轮次对比")
    st.dataframe(summary_df.round(4), width="stretch", hide_index=True)

    if global_best_workflow:
        st.success(f"全局最佳工作流：{global_best_workflow}")
        workflow_snapshot = workflows.get(global_best_workflow, {})
        _render_workflow_best_section(global_best_workflow, workflow_snapshot)
    else:
        st.info("尚无法判断全局最佳工作流（可能仍在运行初期）。")


def _render_iteration_buttons(selected_workflow: str, workflow_snapshot: dict[str, Any], selected_iteration: int | None) -> int | None:
    max_iterations = int(workflow_snapshot.get("max_iterations", 0) or 0)
    current_iteration = int(workflow_snapshot.get("current_iteration", 0) or 0)
    if max_iterations <= 0:
        return selected_iteration

    st.markdown("### 迭代轮次切换")
    cols = st.columns(max_iterations)
    result = selected_iteration
    for i in range(1, max_iterations + 1):
        done = i <= current_iteration
        label = f"{'🔵' if done else '🔴'} {i}"
        with cols[i - 1]:
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
        "<div style='margin-top:6px'><span class='status-pill done'>🔵 已产生该轮数据</span><span class='status-pill pending'>🔴 尚未到该轮</span></div>",
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

    content = html.escape("\n".join(lines))
    st.caption(f"更新时间 {payload.get('updated_at', '-')} | 第{payload.get('iteration', 0)}轮")
    st.markdown(f"<div class='glass'><pre class='mono-block'>{content}</pre></div>", unsafe_allow_html=True)


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
            <div class='metric-box'><strong>修改对象总数</strong><br/>{obj}</div>
            <div class='metric-box'><strong>修改字段总数</strong><br/>{field}</div>
        </div>
        """.format(
            obj=(obj_count if obj_count is not None else "-"),
            field=(field_count if field_count is not None else "-"),
        ),
        unsafe_allow_html=True,
    )


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
        st.info("等待启动：请先上传 IDF 与 API 文件，然后点击启动按钮。")
        return
    if status == "starting":
        st.info("准备中：正在初始化并行工作流与日志捕获。")
        return
    if status == "error":
        st.error("运行中断：可查看上方错误信息并重新启动。")
        return

    current_iteration = int(workflow_snapshot.get("current_iteration", 0) or 0)
    max_iterations = int(workflow_snapshot.get("max_iterations", DEFAULT_MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS)
    latest_status = str(workflow_snapshot.get("latest_status", "正在处理中...") or "正在处理中...")
    status_time = workflow_snapshot.get("status_updated_at")
    prefix = f"第 {current_iteration}/{max_iterations} 轮"
    if status_time:
        st.info(f"{prefix} | {latest_status} | 更新时间 {status_time}")
    else:
        st.info(f"{prefix} | {latest_status}")


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
        }


def main() -> None:
    st.set_page_config(page_title="EnergyPlus 并行工作流可视化", layout="wide")
    _init_page_state()
    _render_style()

    page_state = st.session_state["web_ui_state"]
    snapshot = _read_snapshot(page_state.get("snapshot_path"))
    status = snapshot.get("status", "idle")
    running = status in {"running", "starting"}

    st.title("EnergyPlus 并行工作流可视化驾驶舱")
    st.caption("不改动原有优化逻辑，只把并行工作流的全过程稳定地展示在网页上。")

    st.markdown(
        """
        <div class='glass'>
            <strong>这套页面会完整呈现原始工作流：</strong>
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
        st.info("上传 IDF 和 API 文件后，可自行设置工作流数量与迭代次数；其余配置保持默认逻辑。")
        col_left, col_right = st.columns(2)
        with col_left:
            uploaded_idf = st.file_uploader("上传 IDF 模型文件", type=["idf"], key="idf_uploader", help="必填。上传后将自动复制到运行目录，不改动原始算法逻辑。")
        with col_right:
            uploaded_api = st.file_uploader("上传 API Key 文本文件", type=["txt"], key="api_uploader", help="必填。内容应为可用密钥，等价于原来的 api_key.txt。")

        col_cfg_1, col_cfg_2 = st.columns(2)
        with col_cfg_1:
            num_workflows = st.number_input(
                "并行工作流数量",
                min_value=1,
                max_value=10,
                value=int(snapshot.get("config", {}).get("num_workflows", DEFAULT_NUM_WORKFLOWS) or DEFAULT_NUM_WORKFLOWS),
                step=1,
            )
        with col_cfg_2:
            max_iterations = st.number_input(
                "最大迭代轮次",
                min_value=2,
                max_value=30,
                value=int(snapshot.get("config", {}).get("max_iterations", DEFAULT_MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS),
                step=1,
            )

        st.markdown(
            f"""
            <div class='glass'>
                <strong>固定配置（保持默认逻辑）</strong><br/>
                IDD: {DEFAULT_IDD_PATH}<br/>
                EPW: {DEFAULT_EPW_PATH}<br/>
                日志目录: {DEFAULT_LOG_DIR}
            </div>
            """,
            unsafe_allow_html=True,
        )

        start_disabled = running or (page_state.get("runner_thread") is not None and page_state["runner_thread"].is_alive())
        if st.button("启动并行优化并实时可视化", type="primary", disabled=start_disabled):
            if not uploaded_idf or not uploaded_api:
                st.error("请先上传 IDF 文件和 API Key 文件。")
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
                "max_iterations": int(max_iterations),
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
    top1.metric("当前状态", status_map.get(status, status))
    top2.metric("开始时间", snapshot.get("started_at") or "-")
    top3.metric("结束时间", snapshot.get("finished_at") or "-")
    top4.metric("当前工作流数", str(snapshot.get("config", {}).get("num_workflows", 0)))

    if snapshot.get("error"):
        st.error(f"运行失败：{snapshot['error']}")

    st.markdown("### 工作流总览")
    _render_workflow_overview(snapshot)

    workflow_keys = sorted((snapshot.get("workflows") or {}).keys())
    if not workflow_keys:
        st.info("点击上方按钮后，页面会实时显示每条工作流的推理过程、修改摘要、进度和能耗曲线。")
        return

    view_options = list(workflow_keys)
    if status in {"finished", "error"}:
        view_options.append("summary")
    if page_state.get("selected_view") not in view_options:
        page_state["selected_view"] = workflow_keys[0]

    nav_cols = st.columns(len(view_options))
    for idx, view_name in enumerate(view_options):
        label = "汇总" if view_name == "summary" else f"工作流{view_name.split('_')[-1]}"
        is_active = page_state.get("selected_view") == view_name
        button_label = f"▶ {label}" if is_active else label
        with nav_cols[idx]:
            if st.button(button_label, key=f"view_nav_{view_name}", width="stretch"):
                page_state["selected_view"] = view_name
                page_state["selected_workflow"] = view_name if view_name != "summary" else page_state.get("selected_workflow", workflow_keys[0])
                page_state["last_ui_action_ts"] = time.time()
                st.rerun()

    if page_state.get("selected_view") == "summary":
        _render_summary_page(snapshot)
        if running and (time.time() - float(page_state.get("last_ui_action_ts", 0.0) or 0.0) > 1.2):
            time.sleep(0.8)
            st.rerun()
        return

    selected_workflow = str(page_state.get("selected_view"))
    if selected_workflow not in workflow_keys:
        selected_workflow = workflow_keys[0]
        page_state["selected_view"] = selected_workflow
    workflow_snapshot = (snapshot.get("workflows") or {}).get(selected_workflow, {}) or {}

    st.subheader(f"当前查看：{selected_workflow}")
    st.progress(min(max(float(workflow_snapshot.get("progress", 0.0) or 0.0), 0.0), 1.0), text=f"第 {workflow_snapshot.get('current_iteration', 0)}/{workflow_snapshot.get('max_iterations', 0)} 轮")

    _render_runtime_hint(status, workflow_snapshot)

    reasoning_history = workflow_snapshot.get("reasoning_history", []) or []
    summary_history = workflow_snapshot.get("summary_history", []) or []
    plan_metrics_history = workflow_snapshot.get("plan_metrics_history", []) or []
    round_stats_history = workflow_snapshot.get("round_stats_history", []) or []
    history_iterations = sorted(
        {
            int(item.get("iteration", 0) or 0)
            for item in reasoning_history + summary_history + plan_metrics_history + round_stats_history
            if int(item.get("iteration", 0) or 0) > 0
        }
    )

    if not history_iterations:
        history_iterations = sorted({int(item.get("iteration", 0) or 0) for item in workflow_snapshot.get("iteration_history", []) if int(item.get("iteration", 0) or 0) > 0})

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
    else:
        selected_reasoning = workflow_snapshot.get("latest_reasoning")
        selected_summary = workflow_snapshot.get("latest_summary")
        selected_plan_metrics = workflow_snapshot.get("latest_plan_metrics")
        selected_round_stats = workflow_snapshot.get("latest_round_stats")

    _render_text_panel("LLM推理过程", selected_reasoning, "等待当前工作流产出最终采用的 LLM 推理过程。", numbered=True)
    _render_text_panel("修改摘要", selected_summary, "等待当前工作流产出成功执行后的修改摘要。")
    metrics_left, metrics_right = st.columns(2)
    with metrics_left:
        _render_plan_metrics_panel(selected_plan_metrics)
    with metrics_right:
        _render_round_stats_panel(selected_round_stats)

    iteration_history = workflow_snapshot.get("iteration_history", []) or []
    dataframe = _build_metrics_dataframe(iteration_history)

    st.markdown("### 实时可视化曲线（仅供冷/供暖）")
    progressive_chart = _build_progressive_curve(dataframe, selected_workflow)
    if progressive_chart is not None:
        chart_col, _ = st.columns([5, 5])
        with chart_col:
            st.altair_chart(progressive_chart, width='content')
    else:
        st.info("当前还没有可绘制的制冷/供暖迭代数据。")

    st.markdown("### 各项能耗指标与节能变化")
    if dataframe.empty:
        st.info("第一轮模拟完成后，这里会自动出现四个核心指标数据。")
    else:
        latest = dataframe.iloc[-1]
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("总建筑能耗(kWh)", f"{latest['总建筑能耗(kWh)']:.2f}", f"{latest['总建筑能耗节能幅度(%)']:+.2f}%")
        metric_2.metric("EUI(kWh/m²)", f"{latest['EUI(kWh/m²)']:.2f}", f"{latest['EUI节能幅度(%)']:+.2f}%")
        metric_3.metric("制冷能耗(kWh/m²)", f"{latest['制冷能耗(kWh/m²)']:.2f}", f"{latest['制冷节能幅度(%)']:+.2f}%")
        metric_4.metric("供暖能耗(kWh/m²)", f"{latest['供暖能耗(kWh/m²)']:.2f}", f"{latest['供暖节能幅度(%)']:+.2f}%")

        display_df = dataframe.copy()
        display_df["总建筑能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['总建筑能耗(kWh)']:.2f} ({row['总建筑能耗节能幅度(%)']:+.2f}%)", axis=1)
        display_df["EUI(含节能幅度)"] = display_df.apply(lambda row: f"{row['EUI(kWh/m²)']:.2f} ({row['EUI节能幅度(%)']:+.2f}%)", axis=1)
        display_df["制冷能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['制冷能耗(kWh/m²)']:.2f} ({row['制冷节能幅度(%)']:+.2f}%)", axis=1)
        display_df["供暖能耗(含节能幅度)"] = display_df.apply(lambda row: f"{row['供暖能耗(kWh/m²)']:.2f} ({row['供暖节能幅度(%)']:+.2f}%)", axis=1)
        display_df = display_df[
            [
                "轮次",
                "总建筑能耗(含节能幅度)",
                "EUI(含节能幅度)",
                "制冷能耗(含节能幅度)",
                "供暖能耗(含节能幅度)",
            ]
        ]
        number_columns = [column for column in display_df.columns if column != "轮次"]
        if number_columns:
            for column in number_columns:
                if pd.api.types.is_numeric_dtype(display_df[column]):
                    display_df[column] = display_df[column].round(4)
        st.dataframe(display_df, width='stretch', hide_index=True)

    _render_workflow_best_section(selected_workflow, workflow_snapshot)

    if running and (time.time() - float(page_state.get("last_ui_action_ts", 0.0) or 0.0) > 1.2):
        time.sleep(0.8)
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