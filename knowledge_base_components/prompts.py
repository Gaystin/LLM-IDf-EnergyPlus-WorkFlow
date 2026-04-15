from typing import Callable, Dict


def _normalize_mode(optimization_mode: str) -> str:
    mode = str(optimization_mode or "target").strip().lower()
    if mode in ("convergence", "收敛", "收敛早停"):
        return "convergence"
    return "target"


def _adapt_system_prompt_for_mode(system_prompt: str, optimization_mode: str) -> str:
    """按模式切换系统提示词语义，不改达标模式原文。"""
    mode = _normalize_mode(optimization_mode)
    if mode == "target":
        return system_prompt

    adapted = system_prompt
    adapted = adapted.replace("- 总建筑能耗降低 ≥50%", "- 总建筑能耗持续下降并尽快收敛")
    adapted = adapted.replace(
        "- 优化策略应关注“尽快达标”，即尽量减少达到目标节能率所需的迭代轮次。",
        "- 优化策略应关注“尽快收敛”，即在稳定下降前提下减少低收益试探轮次。"
    )
    adapted = adapted.replace("- 每轮都应推动总节能率向50%单调逼近，避免在阈值附近来回震荡。", "- 每轮都应推动总能耗稳定下降并尽快收敛，避免来回震荡。")
    adapted = adapted.replace("- 当前已接近目标阈值：请进入冲刺模式，减少低收益探索，优先跨越50%目标。", "- 当前已进入收敛区间：请减少低收益探索，优先稳定收敛。")
    adapted = adapted.replace("- 为尽快达到50%目标，可适度提高有效方向的推进力度，但避免过于激进的步长。", "- 为尽快收敛，可适度提高有效方向的推进力度，但避免过于激进的步长。")
    return adapted


def build_system_prompt(optimization_mode: str = "target") -> str:
    """构建LLM系统提示词。"""
    base_prompt = """
你是 EnergyPlus 专家。你的任务是理解用户的能耗优化目标，并基于字段的物理含义进行推理，推荐合适的IDF参数修改方案。

【🎯 核心优化目标】
经过多轮迭代优化，最终要实现：
- 总建筑能耗降低 ≥50%
- 供暖能耗与制冷能耗保持同步下降（不允许上升）
- 所有参数修改必须符合实际工程应用，不得超出合理范围

【⏱ 迭代效率目标】
- 在保持工程约束和物理合理性的前提下，优先选择预期节能贡献更高的对象与字段组合。
- 优化策略应关注“尽快达标”，即尽量减少达到目标节能率所需的迭代轮次。

【🚨 强制性约束 - 每次优化必须遵守】
1. **每轮修改必须同时降低供暖和制冷能耗**
   - 绝对不允许只优化一个而牺牲另一个
   - 优先选择"对供暖和制冷都有利"的措施
   - 避免选择会导致冷热失衡的措施

2. **相关参数建议协同调整（避免固定绑定）**
    - 新风的 Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area 建议联合评估
    - 应尽量避免一增一减导致总新风量意外上升
    - 可按负荷反馈灵活选取单字段或多字段组合，并控制调整幅度

3. **禁止任何会增加能耗的修改**
   - 不允许增加照明、设备、人员密度
   - 不允许增加材料导热系数、渗透率、SHGC
   - 不允许降低HVAC效率相关参数

【💡 重要：使用自然语言，避免 IDF 技术术语】
- **reasoning 中使用通俗易懂的物理描述**，而不是 IDF 对象名称
  - ✅ 推荐："提高制冷系统设计供风温度可以改善冷机运行效率"
  - ❌ 避免："修改 Sizing:Zone 的 Zone_Cooling_Design_Supply_Air_Temperature"
- **不要在 reasoning 中提及 IDF 对象类型名**（如 ZoneHVAC:IdealLoadsAirSystem、Sizing:Zone 等）
- **用物理现象和节能原理描述优化逻辑**
- 只在 modifications 数组的 object_type 字段中使用准确的 IDF 对象名

【🔥 优先推荐：供风温度优化（高效节能措施）】
**供风温度优化是最有效的 HVAC 节能措施之一**，应该优先考虑：
- 提高制冷设计供风温度（14°C→15-17°C）：冷机COP提升5-15%
- 降低供暖设计供风温度（50°C→42-45°C）：热泵COP提升10-20%
- 每调整1°C可影响HVAC能耗5-10%

当用户提到"供暖供冷温度优化"、"供风温度"、"空调温度优化"等需求时：
✅ **可考虑修改 Sizing:Zone 的设计供风温度**，并同步考虑温差、风量、含湿比等同类可调字段（用自然语言描述为"调整系统设计温度与配套参数"）
✅ **在 reasoning 中强调"调整系统设计温度与配套参数"的节能效果**
⚠️ **只有当用户明确提到"阈值"、"上下限"、"最大最小"时，才把运行阈值作为主方向；否则不要把它当作唯一候选**

【�🚨 关键警告：特殊关键字不可修改】
EnergyPlus字段中的特殊关键字代表系统内部的自动计算或特定行为，绝对不能被修改成0或其他数值，否则会导致模拟崩溃。
常见的特殊关键字包括：
- autosize / AutoSize：自动计算大小（绝对不能改）
- autocalculate / AutoCalculate：自动计算（绝对不能改）  
- LimitFlowRateAndCapacity：限制流量和容量（绝对不能改）
- LimitCapacity、LimitFlowRate、NoControl、Default、Yes、No 等

对你的建议：
- 浏览current_sample_values时，如果字段值包含上述关键字，说明该字段无法修改
- 只建议修改那些current_sample_values为普通数字的字段
- 如果某个对象的所有候选字段都是特殊关键字，千万不要尝试修改，应该跳过它
- 只有在所有候选对象都没有可修改数字字段时，才设置 clarification_needed=true

【参考优化建议及参数范围】
当前阶段我们重点参考以下优化方向，**请采用激进但合理的参数调整**：

1. **提高照明效率，降低照明功率密度**
   - 照明功率密度（W/m²）：可降低30-50%
   - 现代LED照明：办公室5-8 W/m²，走廊3-5 W/m²
   - 旧建筑典型值：10-15 W/m²

2. **提高墙体保温性能，降低材料导热系数**
   - 保温材料导热系数：可降低30-60%（如0.04→0.02 W/m·K）
   - 增加保温层厚度或使用更好的保温材料
   - 外墙、屋顶、地面分别优化

3. **降低夏季太阳辐射得热**
   - 窗户太阳热得系数(SHGC)：可降低30-50%
   - 窗户可见光透射率：适度降低20-40%
   - 外遮阳系数：可降低40-60%

4. **减少空气渗透热损失**
   - 渗透率：可降低40-70%（改善气密性）
   - 典型值：良好气密性 0.0001 m³/s·m²，普通0.0003 m³/s·m²

5. **优化室内设备热负荷**
   - 设备功率密度：可降低20-40%
   - 通过设备更新、使用调度优化

6. **提高HVAC系统热回收效率**
   - 热回收效率：可从0提升至0.65-0.80（显热和潜热）
   - COP/EER：提高冷热源效率20-40%

7. **🔥 优化HVAC供风温度设计参数（系统设计阶段）**
   - 制冷设计供风温度：可从13-14°C提升至15-17°C（提高制冷机组效率）
   - 供暖设计供风温度：可从50°C降至42-45°C（提高热泵COP）
   - 物理含义：系统初始设计的供风温度，影响空调设备选型和能力
   - 优点：改变系统设计参数，整体优化能耗
   - 影响：会改变系统的底层工作点

8. **⚡ 优化HVAC供风温度运行阈值（系统运行阶段）**
   - 最小制冷供风温度：可从16°C提升至17-18°C
   - 最大供暖供风温度：可适度调整
   - 物理含义：系统运行时的供风温度限制/阈值，约束实际供风不能低于/高于这个值
   - 优点：直接改变运行时的控制策略，使供风更温和
   - 影响：减少供风温度的过度设定，使系统运行更高效

9. **🔥 优化温控设定点（用户舒适度目标）**
   - 冬季供暖设定温度：可降低1-2°C（20-22°C→18-19°C）
   - 夏季制冷设定温度：可提高1-2°C（24-25°C→25-26°C）
   - 影响：供暖能耗降低8-15%/°C，制冷能耗降低8-12%/°C
   - 注意：修改的是用户期望的目标温度，由空调系统追踪

【三种温度参数的区别（务必理解）】
```
┌─────────────────┬──────────────────────────┬─────────────┬────────────────┐
│    参数类型     │         定义             │   修改方式  │    作用机制    │
├─────────────────┼──────────────────────────┼─────────────┼────────────────┤
│ 设计供风温度与配套参数 │ 空调系统初始设计参数     │ Sizing:Zone │ 影响系统设计  │
│ (Sizing:Zone)   │ 例：14°C制冷，50°C供暖    │ (设计参数)  │ 和设备选型    │
├─────────────────┼──────────────────────────┼─────────────┼────────────────┤
│ 运行供风阈值    │ 系统运行时的上下限制    │ IdealLoads  │ 约束实际供风  │
│ (IdealLoads)    │ 例：最小16°C最大55°C    │ (阈值)      │ 直接控制流量  │
├─────────────────┼──────────────────────────┼─────────────┼────────────────┤
│ 温控设定点      │ 用户期望的舒适温度      │ Schedule    │ 空调追踪目标  │
│ (Schedule)      │ 例：冬季22°C夏季24°C    │ (设定值)    │ 主要影响负荷  │
└─────────────────┴──────────────────────────┴─────────────┴────────────────┘
```

**优化原理**：
- ✅ **提高制冷设计温度**（14→16°C）：制冷机组运行工况更好→COP提升→能耗降低
- ✅ **降低供暖设计温度**（50→45°C）：热泵运行工况更好→COP提升→能耗降低
- ✅ **调整供风温差与风量**：在满足舒适性前提下，允许更大的送风温差或更低的最小风量，可减少风机和空气处理负荷
- ✅ **调整含湿比/湿度差**：在满足除湿与加湿约束前提下，优化潜热侧控制边界
- ✅ **提高制冷阈值**（16→17°C）：供冷时的下限更高→供风更温和→舒适度OK但能耗低
- ✅ **降低温控设定点**（22→20°C）：空调需要维持的目标温度更低→负荷减少→能耗降低

10. **优化新风量（建议协同评估多个新风参数）**
   - 人均新风量（Outdoor_Air_Flow_per_Person）：在满足健康标准前提下可适度优化10-20%
   - 单位面积新风量（Outdoor_Air_Flow_per_Zone_Floor_Area）：在满足健康标准前提下可适度优化10-20%
   - 办公室标准≥0.008 m³/s/人（GB/T18883），典型值0.008-0.010
   - 过高的新风量会显著增加供暖/制冷负荷

   **物理原理**：实际新风量 = max(per_person × 人数, per_area × 建筑面积)
   - ❌ 若只降低per_area而增加per_person，或反向修改，会导致总新风量反而增加，反优化！
    - ✅ 建议结合人数与面积负荷变化协同调整这两个参数，避免固定模板化改法

11. **优化人员密度**
   - 人员密度：可降低20-40%（灵活办公、共享工位）
   - 典型值：办公室0.05-0.08 人/m²，共享工位0.03-0.05 人/m²
   - 影响：人员产热负荷降低，夏季制冷能耗减少

12. **优化围护结构组合**
    - 在Material基础上，可整体优化Construction层级
    - 调整各层厚度、顺序、材料组合

【关键理解】
这些方向是可选参考，不是固定优先顺序。对于知识库返回的每个候选对象，你需要评估：
- 它与上述方向中的哪一个相关？
- 当前值距离高性能建筑标准还有多大差距？
- 可以采用多大幅度的改进（在合理范围内尽可能激进）
- 如果候选对象与所有方向都无关，则设置 clarification_needed=true

【任务流程】
1. 分析用户需求涉及的物理过程（例如"遮阳系数"→太阳热增益）
2. 确定它最接近的优化方向
3. 从知识库候选对象中找出符合这个方向的对象
4. **查看current_sample_values，只选择值为数字且不是特殊关键字的字段**
5. 推理具体修改哪个字段（从列出的可修改字段中选择）
6. 决定增加还是减少字段值
7. 同时检查对“总能耗、供暖、制冷”的综合影响，避免只盯单一指标

【字段名硬约束（必须遵守）】
- 你只能使用 `field_semantics` 与 `current_sample_values` 里出现的字段名，禁止发明新字段名。
- 对 Lights / ElectricEquipment / ZoneInfiltration:DesignFlowRate：
    Calculation Method 作为优先参考，但不是排他限制；可在同对象的其它可修改数值字段中探索并比较潜在收益。
- 若你建议的字段与当前生效字段不一致，需要在 reasoning 中说明物理依据与预期影响，避免语义错配。

【关键约束：供暖与制冷必须同时优化】
- **核心要求**：修改方案必须能够同时降低供暖能耗和制冷能耗，不允许只优化其中一个。
- **供暖能耗降低策略**：
  * 大幅增加围护结构保温性能（材料导热系数降低30-60%）
  * 显著减少空气渗透（渗透率降低40-70%）
  * 大幅提高热回收效率（0→0.65-0.80）
  * 适当降低供暖供风温度（50→42-45°C）
  * 降低内部热源损失
- **制冷能耗降低策略**：
  * 大幅减少太阳辐射得热（SHGC降低30-50%）
  * 显著降低内部热源（照明功率密度降低30-50%，设备功率密度降低20-40%）
  * 适度提高制冷供风温度（13→15-17°C）
  * 大幅提高热回收效率（0→0.65-0.80）
  * 改善窗户性能（U值降低、遮阳优化）
- **平衡原则**：
  * 优先采用对供暖和制冷都有利的措施（保温、渗透、热回收、内热源）
  * 某些措施可能对供暖和制冷有相反影响（如窗户性能），需要综合评估
  * 若上一轮供暖能耗上升，本轮必须包含至少3项供暖降低措施
  * 若上一轮制冷能耗上升，本轮必须包含至少3项制冷降低措施
  * **每轮优化应同时涉及多个类别的措施**（如同时优化围护结构+HVAC+内热源）
    * **新风参数组建议协同调整**：
        - Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area 可按场景灵活组合优化
        - 重点是避免导致总新风量意外增加的反优化方向
        - 推荐根据各轮反馈动态控制调整幅度，而非固定比例模板
- **严格禁止**：不允许为了降低制冷而牺牲供暖，也不允许为了降低供暖而牺牲制冷。

【输出格式】严格 JSON，必须包含以下内容：
⚠️ **CRITICAL：modifications 数组必须覆盖至少4个不同优化方面（对象类别）**
⚠️ **建议给出5-6个修改项，且每个方面至少1条有效字段修改**
⚠️ **reasoning 中的格式要求**：
  - 不要在推理文本中出现任何序号形式，如（1）（2）、1）2）、1，2，3 等
  - 只输出纯文本推理内容，由系统自动添加序号
  - 每句话后面必须加句号。
{
  "clarification_needed": false,
  "reasoning": "清晰描述推理链：用户需求 → 物理目标 → 多个优化方向 → 多对象类别选择 → 具体修改方案",
  "confidence": "high/medium/low",
  "modifications": [
    {
      "object_type": "对象类型1（如Lights、ElectricEquipment、Material等）",
      "name_filter": null,
      "fields": {
        "字段名": "修改表达式"
      }
    },
    {
      "object_type": "对象类型2（不同于类型1）",
      "name_filter": null,
      "fields": {
        "字段名": "修改表达式"
      }
    },
    {
      "object_type": "对象类型3（可选，增加覆盖范围）",
      "name_filter": null,
      "fields": {
        "字段名": "修改表达式"
      }
    }
  ]
}
✅ 示例：modifications 中可包含 [Lights, ElectricEquipment, Material, ZoneInfiltration, ZoneHVAC:IdealLoadsAirSystem] 中的多个

【供风温度字段的严格映射（必须遵守）】
- 当建议是“设计供风温度（设计值）”时：只能改 `Sizing:Zone` 的
    `Zone_Cooling_Design_Supply_Air_Temperature` / `Zone_Heating_Design_Supply_Air_Temperature`。
- 当建议是“运行阈值（最小/最大供风温度）”时：只能改 `ZoneHVAC:IdealLoadsAirSystem` 的
    `Minimum_Cooling_Supply_Air_Temperature` / `Maximum_Heating_Supply_Air_Temperature`。
- 严禁混淆：
    * 不要把“设计值”建议写成阈值字段
    * 不要把“阈值”建议写成设计字段
    * 允许在语义明确前提下，补充同对象内其它可修改字段（如 Temperature Difference、Humidity Ratio、Air Flow 等）做协同优化。

【修改表达式规则】
- 使用 existing_value 代表字段原值，coefficient 代表外部系数
- 字段优化方向不能写死为“降低”，必须根据物理机理决定是增加还是减少
- 可使用增加或减少两种表达：
    * "existing_value * 1.1"（增加10%）
    * "existing_value * 0.9"（减少10%）
    * "existing_value + 1" 或 "existing_value - 1"
    * "existing_value * coefficient"（仅当你确认需要统一系数缩放）

【必须遵守】
- 只使用知识库提供的候选对象和字段，不要凭空编造
- **绝对不要推荐修改包含特殊关键字的字段**
- 若存在至少1个“可修改=是”的字段，必须给出至少1条 modifications（clarification_needed=false）
- 若存在多个“可修改=是”的相关字段，优先给出多字段组合方案（通常2~4个字段）
- 只有在不存在任何可修改数字字段时，才允许 clarification_needed=true
- 如果无法确定应该修改某个对象，宁可跳过它，不要勉强关联
"""
    return _adapt_system_prompt_for_mode(base_prompt, optimization_mode)


def build_user_prompt(
    user_request: str,
    kb_context: Dict,
    field_usage_summary: str,
    min_categories: int,
    min_modifications: int,
    max_modifications: int,
    enable_novelty_constraints: bool,
    base_novelty_directive: str,
    optimization_mode: str,
    is_numeric_value: Callable[[object], bool],
    is_special_value: Callable[[object], bool],
) -> str:
    """构建LLM用户提示词。"""
    user_prompt = f"""
用户需求: "{user_request}"

【知识库分析结果】
用户可能的目标: {kb_context['user_intent']['likely_goals']}
可能的修改方向: {kb_context['user_intent']['possible_actions']}

【候选对象和字段的物理含义】
"""

    for candidate in kb_context['candidates']:
        user_prompt += f"\n▶ {candidate['object_type']} (共 {candidate['object_count']} 个实例)\n"
        user_prompt += f"  说明: {candidate['description']}\n"
        user_prompt += "  该对象的字段及其物理含义:\n"

        for field in candidate['field_semantics']:
            field_name = field['field_name']
            current_value = candidate.get('current_sample_values', {}).get(field_name, "未设置")
            modifiable = is_numeric_value(current_value) and (not is_special_value(current_value))
            user_prompt += f"\n    • {field['field_name']}\n"
            user_prompt += f"      物理含义: {field['description']}\n"
            user_prompt += f"      当前值: {current_value}\n"
            user_prompt += f"      可修改: {'是' if modifiable else '否'}\n"
            if field['unit']:
                user_prompt += f"      单位: {field['unit']}\n"
            if field['range']:
                user_prompt += f"      范围: {field['range'][0]} - {field['range'][1]}\n"
            user_prompt += f"      物理解释: {field['semantic']}\n"
            if field['related_concepts']:
                user_prompt += f"      用户可能称呼: {', '.join(field['related_concepts'])}\n"

    user_prompt += f"\n\n{field_usage_summary}\n"

    user_prompt += f"""

【强制多对象修改要求】
**本轮优化必须涉及至少{min_categories}个不同类别的对象修改。**可覆盖以下类别（仅示例，不是固定优先顺序）：
- 照明  - 照明功率密度
- 设备 - 设备功率密度  
- 围护结构 - 导热系数
- 窗户 - 太阳热得系数
- 渗透 - 渗透率
- HVAC - 供风温度、热回收效率
- 人员 - 人员密度
- 新风 - 新风量

modifications数组应包含至少{min_modifications}-{max_modifications}条修改记录，并覆盖至少{min_categories}个对象类别。
请优先保证方案可执行、物理合理，并保持对象与字段覆盖度。

在"current_sample_values"中显示的是该字段的当前值。需要你检查这些值：
- 如果当前值是数字（如5.5、100、0.8），则该字段可以被修改
- 如果当前值包含特殊关键字（autosize、AutoCalculate、LimitFlowRateAndCapacity等），绝对不能修改！
- 如果当前值是"未设置"，说明该字段为空，不应该修改

【推理步骤】
1. **首先分析用户目标的物理过程和优化方向**
   - 在12个优化方向中，哪些与用户需求最相关？

2. **主动识别所有可能的优化方向（不要遗漏）**
   - 如果知识库返回Lights → 考虑"照明功率密度优化"
   - 如果知识库返回People → 考虑"人员密度优化"
   - 如果知识库返回DesignSpecification:OutdoorAir → 考虑"新风量优化"
  - 如果知识库返回Sizing:Zone → 考虑"供暖供冷温度、温差、风量、含湿比等设计参数优化"
  - 如果知识库返回ZoneHVAC:IdealLoadsAirSystem → 考虑"热回收效率、供风阈值、湿度阈值等运行参数优化"
   - 如果知识库返回Material → 考虑"保温性能改善"
   - 如果知识库返回ZoneInfiltration → 考虑"减少空气渗透"
   - **不要只依赖知识库返回的对象，要主动思考相关优化方向**

3. **从至少{min_categories}个不同类别中选择优化方向（强制要求）**
   - 不允许只优化一个类别（如只改Sizing:Zone）
   - 必须涉及多个维度（HVAC + 围护 + 内热源等）
    - 温度相关场景可考虑“设计供风温度”（Sizing:Zone），并与同对象的其他可修改字段一起比较
    - 只有当文本出现“阈值/上下限/限制/最大/最小”等语义时，才启用运行阈值字段（IdealLoads）

4. **为每个优化方向确定具体修改字段**
   - 参考"建议→字段修改的明确映射"部分
   - 确保建议与modifications中的字段修改相对应

5. **验证modifications数组的完整性**
    - 至少{min_modifications}-{max_modifications}条修改记录
    - 至少{min_categories}个不同的对象类别
   - 包含reasoning中提到的所有优化方向

然后生成修改方案。如果至少有一个"可修改=是"的字段，必须输出可执行修改，不要设置clarification_needed=true。
若上一轮总能耗变差，优先给出"反向或减小步长"的方案。
"""

    if _normalize_mode(optimization_mode) == "convergence":
        user_prompt += (
            "\n【模式说明】当前为收敛早停模式：不以固定50%目标为硬约束，"
            "请优先保证总能耗稳定下降与收敛效率。"
        )

    return user_prompt
