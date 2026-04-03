def is_special_value(value) -> bool:
    """检测字段值是否为EnergyPlus特殊关键字（不应修改）。"""
    if not isinstance(value, str):
        return False

    value_lower = value.lower().strip()
    if value_lower == '':
        return True

    special_keywords = [
        ('autosize', 'exact'),
        ('autocalculate', 'exact'),
        ('limitflowrateandcapacity', 'exact'),
        ('limitcapacity', 'exact'),
        ('limitflowrate', 'exact'),
        ('nocontrol', 'exact'),
        ('on', 'exact'),
        ('off', 'exact'),
        ('default', 'exact'),
        ('differencedays', 'exact'),
        ('differencescheduled', 'exact'),
        ('yes', 'exact'),
        ('no', 'exact'),
    ]

    for keyword, match_type in special_keywords:
        if match_type == 'exact' and value_lower == keyword:
            return True
        if match_type == 'contains' and keyword in value_lower:
            return True

    return False


def is_numeric_value(value) -> bool:
    """判断字段值是否为可计算的数值。"""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return False
        try:
            float(text)
            return True
        except (ValueError, TypeError):
            return False
    return False


def build_balanced_expression_for_field(object_type: str, field_name: str) -> str:
    """为指定字段构造保守平衡的表达式（用于多样性补充）。"""
    field_upper = str(field_name).upper()

    if "MULTIPLIER" in field_upper:
        return "existing_value"

    increase_patterns = ["EFFECTIVENESS", "EFFICIENCY", "COP", "EER", "HEATRECOVERY"]
    for pattern in increase_patterns:
        if pattern in field_upper:
            return "existing_value * 1.02"

    decrease_patterns = ["WATTS", "POWER", "DENSITY", "INFILTRATION", "CONDUCTIVITY", "FLOW", "PEOPLE", "SOLAR", "SHGC"]
    for pattern in decrease_patterns:
        if pattern in field_upper:
            return "existing_value * 0.97"

    if "TEMPERATURE" in field_upper:
        if "COOLING" in field_upper or "COOL" in field_upper:
            return "existing_value + 1.0"
        if "HEATING" in field_upper or "HEAT" in field_upper:
            return "existing_value - 1.0"

    return "existing_value * 0.96"
