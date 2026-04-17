"""
Task 批量优化入口：遍历 Task 下不同复杂度 IDF，按配置范围逐个运行工作流。

设计原则：
1. 不改动原工作流能力，直接复用 main.py 中的 EnergyPlusOptimizer。
2. 每个 IDF 仅运行 1 次，城市固定 Beijing。
3. 输出目录固定为 Task-result/<Task-Tier>/<idf_stem>/。
4. 支持按复杂度文件夹与 IDF 序号区间可配置遍历，便于 debug。
"""

import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from main import (
    EnergyPlusOptimizer,
    _collect_run_export_rows,
    _export_city_xlsx,
)


# =========================
# 可配置区（按需修改）
# =========================
IDD_PATH = "Energy+v24.2.idd"
API_KEY_PATH = "api_key.txt"
WEATHER_PATH = os.path.join("weather", "Beijing.epw")

# 复用主流程模式："target" 或 "convergence"
OPTIMIZATION_MODE = "convergence"

# 并行工作流数量（保持和原主程序一致风格，默认 1）
NUM_WORKFLOWS = 1

# 轮次上限：None 表示沿用 EnergyPlusOptimizer 内部默认上限
MAX_ITERATIONS: Optional[int] = None

# Task 根目录与输出根目录
TASK_ROOT = "Task_new"
OUTPUT_ROOT = "Task-result"

# 要遍历的复杂度文件夹（目录名需与 Task 下实际目录一致）
TARGET_TIERS: List[str] = ["Task-High", "Task-Medium", "Task-Low"]

# 可配置遍历范围（1-based 且含两端）
# 例如："Task-High": (3, 10) 表示仅跑该目录下第3到第10个IDF
# 设为 (None, None) 表示全量
RANGE_BY_TIER: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "Task-High": (None, None),
    "Task-Medium": (None, None),
    "Task-Low": (None, None),
}

# 是否只跑指定文件名（不含后缀）；空列表表示不启用该过滤
# 支持字符串或整数写法，都会自动转为字符串比较
# 示例: ["10125", "10751"] 或 [10125, 10751]
ONLY_IDF_STEMS: List[str] = [10610, 10470, 10465, 10784, 10493, 10808, 10597, 10626]


def _idf_sort_key(path_obj: Path):
    """IDF 文件排序键：数字文件名优先按数值排序，其余按字符串排序。"""
    stem = path_obj.stem
    if re.fullmatch(r"\d+", stem):
        return (0, int(stem))
    return (1, stem.lower())


def _slice_by_1based_range(
    items: List[Path],
    start_idx: Optional[int],
    end_idx: Optional[int],
) -> List[Path]:
    """按 1-based 且含两端的区间切片。"""
    if not items:
        return []

    s = 1 if start_idx is None else max(1, int(start_idx))
    e = len(items) if end_idx is None else min(len(items), int(end_idx))

    if s > e:
        return []

    return items[s - 1 : e]


def _build_empty_city_rows() -> Dict[str, list]:
    return {
        "run_summary": [],
        "round_details": [],
        "early_stop": [],
        "object_freq": [],
        "field_freq": [],
        "iteration_object_freq": [],
        "iteration_field_freq": [],
        "run_frequency_combined": [],
    }


def _validate_common_inputs():
    missing = []
    if not os.path.exists(IDD_PATH):
        missing.append(IDD_PATH)
    if not os.path.exists(API_KEY_PATH):
        missing.append(API_KEY_PATH)
    if not os.path.exists(WEATHER_PATH):
        missing.append(WEATHER_PATH)
    if not os.path.exists(TASK_ROOT):
        missing.append(TASK_ROOT)

    if missing:
        raise FileNotFoundError("缺少必要输入文件/目录: " + ", ".join(missing))


def main():
    _validate_common_inputs()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_selected = 0
    total_done = 0

    for tier in TARGET_TIERS:
        tier_input_dir = Path(TASK_ROOT) / tier
        if not tier_input_dir.exists():
            print(f"⚠ 跳过 {tier}: 目录不存在 -> {tier_input_dir}")
            continue

        all_idf_files = sorted(tier_input_dir.glob("*.idf"), key=_idf_sort_key)
        if ONLY_IDF_STEMS:
            allow = {str(x).strip() for x in ONLY_IDF_STEMS if str(x).strip()}
            all_idf_files = [p for p in all_idf_files if p.stem in allow]

        range_cfg = RANGE_BY_TIER.get(tier, (None, None))
        start_idx, end_idx = range_cfg
        selected_files = _slice_by_1based_range(all_idf_files, start_idx, end_idx)

        print("\n" + "=" * 100)
        print(f"复杂度目录: {tier}")
        print(f"IDF总数: {len(all_idf_files)} | 本次选中: {len(selected_files)} | 范围: {range_cfg}")
        print("=" * 100)

        total_selected += len(selected_files)

        for idf_path in selected_files:
            idf_name = idf_path.stem
            idf_output_root = Path(OUTPUT_ROOT) / tier / idf_name
            idf_output_root.mkdir(parents=True, exist_ok=True)

            log_dir = str(idf_output_root / "optimization_logs_并行_Beijing")
            optimization_dir = str(idf_output_root / "optimization_results_并行_Beijing")
            plot_dir = str(idf_output_root / "optimization_plot_并行_Beijing")

            print("\n" + "-" * 100)
            print(f"开始运行: {idf_path}")
            print(f"输出目录: {idf_output_root}")
            print("-" * 100)

            city_rows = _build_empty_city_rows()
            optimizer = None

            try:
                optimizer = EnergyPlusOptimizer(
                    idf_path=str(idf_path),
                    idd_path=IDD_PATH,
                    api_key_path=API_KEY_PATH,
                    epw_path=WEATHER_PATH,
                    city_name="Beijing",
                    optimization_mode=OPTIMIZATION_MODE,
                    log_dir=log_dir,
                    optimization_dir=optimization_dir,
                    plot_dir=plot_dir,
                    num_workflows=NUM_WORKFLOWS,
                )

                if MAX_ITERATIONS is None:
                    optimizer.run_optimization_loop()
                else:
                    optimizer.run_optimization_loop(max_iterations=int(MAX_ITERATIONS))

                run_rows = _collect_run_export_rows(
                    city="Beijing",
                    run_tag="run_01",
                    optimizer=optimizer,
                    run_status="success",
                    error_message="",
                )
                for k in city_rows.keys():
                    city_rows[k].extend(run_rows.get(k, []))

                summary_xlsx = _export_city_xlsx(str(idf_output_root), "Beijing", city_rows)
                print(f"✓ 完成: {idf_path}")
                print(f"✓ 统计表: {summary_xlsx}")

            except Exception as e:
                run_rows = _collect_run_export_rows(
                    city="Beijing",
                    run_tag="run_01",
                    optimizer=optimizer,
                    run_status="failed",
                    error_message=str(e),
                )
                for k in city_rows.keys():
                    city_rows[k].extend(run_rows.get(k, []))

                try:
                    summary_xlsx = _export_city_xlsx(str(idf_output_root), "Beijing", city_rows)
                    print(f"✗ 失败: {idf_path}")
                    print(f"✗ 错误: {e}")
                    print(f"✓ 已导出失败统计表: {summary_xlsx}")
                except Exception as export_e:
                    print(f"✗ 失败: {idf_path}")
                    print(f"✗ 错误: {e}")
                    print(f"✗ 导出失败统计表也失败: {export_e}")

                traceback.print_exc()

            total_done += 1
            print(f"当前进度: {total_done}/{total_selected}")

    print("\n" + "=" * 100)
    print(f"任务结束: 已完成 {total_done}/{total_selected} 个 IDF")
    print(f"结果根目录: {OUTPUT_ROOT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
