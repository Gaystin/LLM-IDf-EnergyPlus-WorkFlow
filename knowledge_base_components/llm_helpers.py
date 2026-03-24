from typing import Dict, Optional, Set, Tuple


def extract_plan_field_keys(plan: Dict) -> Set[str]:
    """从LLM计划中提取标准化字段键集合 OBJECT.FIELD。"""
    keys = set()
    if not isinstance(plan, dict):
        return keys
    modifications = plan.get("modifications", [])
    if not isinstance(modifications, list):
        return keys

    for mod in modifications:
        if not isinstance(mod, dict):
            continue
        obj = str(mod.get("object_type", "")).upper().strip()
        fields = mod.get("fields", {})
        if not obj or not isinstance(fields, dict):
            continue
        for field in fields.keys():
            keys.add(f"{obj}.{str(field).upper().strip()}")
    return keys


def build_novelty_directive(
    last_round_fields: Set[str],
    field_modification_history: Dict[str, int],
    rejected_plan_keys: Set[str] = None,
    max_items: int = 10,
) -> str:
    """构建给LLM的低频/未出现字段约束。"""
    lines = ["【低频与新颖性硬约束（必须遵守）】"]

    last_round = sorted(list(last_round_fields or set()))
    if last_round:
        lines.append("1) 以下为上一轮已使用字段，除非必要请避免复用：")
        for key in last_round[:max_items]:
            lines.append(f"   - {key}")

    if field_modification_history:
        sorted_freq = sorted(field_modification_history.items(), key=lambda x: x[1], reverse=True)
        high_freq = [str(k).upper() for k, _ in sorted_freq[:max_items]]
        if high_freq:
            lines.append("2) 以下为历史高频字段，本轮应尽量避免：")
            for key in high_freq:
                lines.append(f"   - {key}")

    rejected_keys = sorted(list(rejected_plan_keys or set()))
    if rejected_keys:
        lines.append("3) 以下为当前被判定不合格方案涉及字段，重生成时应优先避免原样重复（可保留少量已验证字段）：")
        for key in rejected_keys[:max_items]:
            lines.append(f"   - {key}")

    lines.append("4) 请优先从候选中选择历史低频或未出现字段，并尽量与上一轮至少50%不同。")
    lines.append("5) 若可行字段不足，可保留少量高价值历史字段，同时扩大对象类型覆盖，避免机械重复。")
    return "\n".join(lines)


def build_retry_anti_repeat_directive(
    last_round_fields: Set[str],
    rejected_plan_keys: Set[str] = None,
    max_items: int = 12,
) -> str:
    """构建重试反重复约束。"""
    lines = ["【重试反重复约束（必须遵守）】"]

    previous_fields = sorted(list(last_round_fields or set()))
    if previous_fields:
        lines.append("1) 以下是上一轮已实际修改字段，本轮与其重合比例必须<=50%，禁止原样照搬：")
        for key in previous_fields[:max_items]:
            lines.append(f"   - {key}")

    rejected_fields = sorted(list(rejected_plan_keys or set()))
    if rejected_fields:
        lines.append("2) 以下是刚刚被判失败的字段集合，下一次重试不得与其完全相同：")
        for key in rejected_fields[:max_items]:
            lines.append(f"   - {key}")

    lines.append("3) 若仍选择上述部分字段，必须同时加入足够多的新字段，使总重合比例降到50%以下。")
    lines.append("4) 不要只改措辞；reasoning 和 modifications 都必须体现新的字段组合。")
    return "\n".join(lines)


def get_field_usage_summary(
    field_modification_history: Dict[str, int],
    max_high_freq: int = 10,
    max_low_freq: int = 10,
) -> str:
    """生成字段使用频率摘要，用于LLM prompt。"""
    if not field_modification_history:
        return "暂无历史修改记录，这是首轮优化。"

    sorted_fields = sorted(
        field_modification_history.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    high_freq_fields = sorted_fields[:max_high_freq]
    low_freq_fields = sorted_fields[-max_low_freq:] if len(sorted_fields) > max_low_freq else []

    summary = "【历史字段修改频率统计】\n"

    if high_freq_fields:
        summary += "⚠️ 高频修改字段（已被过度使用，本轮应避免或减少使用）：\n"
        for field, count in high_freq_fields:
            summary += f"  - {field}: 已使用 {count} 次\n"

    if low_freq_fields:
        summary += "\n✅ 低频修改字段（推荐优先使用，增加优化多样性）：\n"
        for field, count in low_freq_fields:
            summary += f"  - {field}: 仅使用 {count} 次\n"

    object_types_count = {}
    for field_key in field_modification_history.keys():
        obj_type = field_key.split('.')[0]
        object_types_count[obj_type] = object_types_count.get(obj_type, 0) + 1

    summary += f"\n已修改的对象类型数: {len(object_types_count)}\n"
    summary += "对象类型覆盖情况：\n"
    for obj_type, count in sorted(object_types_count.items(), key=lambda x: x[1], reverse=True):
        summary += f"  - {obj_type}: {count} 个字段被修改过\n"

    return summary


def contains_threshold_intent(text: str) -> bool:
    """判断文本是否包含阈值/上下限语义。"""
    text_lower = str(text or "").lower()
    threshold_keywords = [
        "阈值", "上下限", "上限", "下限", "限制", "最大", "最小",
        "threshold", "limit", "upper", "lower",
    ]
    return any(keyword in text_lower for keyword in threshold_keywords)


def get_temperature_mapping_issue(plan: Dict, user_request: str) -> Optional[str]:
    """校验温度建议是否符合“设计温度优先、阈值按语义启用”的规则。"""
    if not isinstance(plan, dict):
        return "计划格式无效"

    reasoning = str(plan.get("reasoning", ""))
    text_for_intent = f"{user_request}\n{reasoning}"
    has_threshold_intent = contains_threshold_intent(text_for_intent)

    has_ideal_threshold_fields = False
    has_sizing_design_fields = False

    for mod in plan.get("modifications", []):
        object_type = str(mod.get("object_type", "")).upper()
        fields = mod.get("fields", {})
        if not isinstance(fields, dict):
            continue

        field_names_upper = {str(field).upper() for field in fields.keys()}

        if object_type == "ZONEHVAC:IDEALLOADSAIRSYSTEM":
            if (
                "MINIMUM_COOLING_SUPPLY_AIR_TEMPERATURE" in field_names_upper
                or "MAXIMUM_HEATING_SUPPLY_AIR_TEMPERATURE" in field_names_upper
            ):
                has_ideal_threshold_fields = True

        if object_type == "SIZING:ZONE":
            if (
                "ZONE_COOLING_DESIGN_SUPPLY_AIR_TEMPERATURE" in field_names_upper
                or "ZONE_HEATING_DESIGN_SUPPLY_AIR_TEMPERATURE" in field_names_upper
            ):
                has_sizing_design_fields = True

    if not has_threshold_intent and has_ideal_threshold_fields and not has_sizing_design_fields:
        return "未检测到阈值语义，但方案使用了IdealLoads阈值字段且缺少Sizing:Zone设计温度字段"

    return None


def check_field_diversity(
    plan: Dict,
    last_round_fields: Set[str],
    field_modification_history: Dict[str, int],
) -> Tuple[bool, Optional[str]]:
    """检查优化方案的字段多样性。"""
    if not isinstance(plan, dict) or 'modifications' not in plan:
        return True, None

    modifications = plan.get('modifications', [])
    if not modifications:
        return True, None

    current_fields = set()
    for mod in modifications:
        obj_type = str(mod.get('object_type', '')).upper()
        fields = mod.get('fields', {})
        for field_name in fields.keys():
            field_key = f"{obj_type}.{str(field_name).upper()}"
            current_fields.add(field_key)

    if last_round_fields and len(current_fields) > 0:
        overlap_fields = current_fields & set(last_round_fields)
        overlap_ratio = len(overlap_fields) / len(current_fields)
        if overlap_ratio > 0.5:
            return False, (
                f"本轮方案与上一轮重复率过高（{overlap_ratio:.1%}），"
                f"{len(overlap_fields)}/{len(current_fields)}个字段与上一轮相同。"
                "要求不同轮次至少50%字段不同，请选择新的优化方向。"
            )

    if field_modification_history:
        sorted_fields = sorted(
            field_modification_history.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_5_high_freq = {field.upper() for field, _ in sorted_fields[:5]}

        overlap_count = 0
        for field in current_fields:
            if field.upper() in top_5_high_freq:
                overlap_count += 1

        if len(current_fields) > 0 and overlap_count / len(current_fields) >= 0.8:
            return True, None

    return True, None
