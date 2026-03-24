"""
EnergyPlus IDF 知识库模块
用于从 IDD 文件提取结构化知识，并提供语义匹配功能，
帮助 LLM 准确识别用户需求对应的 object 和 field。
"""
import json
import re
from typing import Dict, List, Optional, Set, Tuple

from knowledge_base_components.mappings import (
    analyze_user_intent,
    build_field_keyword_mapping,
    build_field_semantics,
    build_keyword_mapping,
)
from knowledge_base_components.llm_helpers import (
    build_novelty_directive,
    build_retry_anti_repeat_directive,
    check_field_diversity,
    contains_threshold_intent,
    extract_plan_field_keys,
    get_field_usage_summary,
    get_temperature_mapping_issue,
)
from knowledge_base_components.prompts import (
    build_system_prompt,
    build_user_prompt,
)
from knowledge_base_components.value_rules import (
    build_balanced_expression_for_field,
    is_numeric_value,
    is_special_value,
)


class EnergyPlusKnowledgeBase:
    """EnergyPlus 知识库：解析 IDD 并提供智能匹配"""
    
    def __init__(self, idd_path: str, idf_path: str = None):
        self.idd_path = idd_path
        self.idf_path = idf_path  # 添加IDF路径以检查实际值
        self.objects_metadata = {}  # {object_type: {fields: [...], description: ...}}
        self.keyword_mapping = self._build_keyword_mapping()
        self.field_semantics = self._build_field_semantics()  # 字段语义库
        self._parse_idd()
        
        # 缓存每个对象的可修改字段（通过检查IDF中的实际值）
        self._modifiable_fields_cache = {}
        if idf_path:
            self._analyze_modifiable_fields(idf_path)
    
    def _build_keyword_mapping(self) -> Dict[str, List[str]]:
        """
        构建关键词到对象类型的映射库
        这是一个语义知识库，将常见用户描述映射到标准 EnergyPlus 对象
        """
        return build_keyword_mapping()
    
    def _build_field_keyword_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """
        构建粗粒度的关键词到字段的映射（仅用于初步过滤，最终由LLM推理）
        格式: {object_type: {keyword: [field_names]}}
        注意：这不是硬编码的参数映射，而是帮助过滤候选字段的工具
        """
        return build_field_keyword_mapping()
    
    def _build_field_semantics(self) -> Dict[str, Dict[str, Dict]]:
        """
        构建字段语义库：为每个对象的字段提供物理含义、数据范围、修改建议等
        这样LLM可以理解字段的实际含义，而不是靠硬编码的映射
        格式: {object_type: {field_name: {description, unit, range, semantic}}}
        """
        return build_field_semantics()
    
    def _parse_idd(self):
        """
        解析 IDD 文件，提取对象类型和字段元数据
        """
        try:
            with open(self.idd_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.idd_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # 按对象分割（以 \n 开头且不是注释的行为新对象）
        current_object = None
        current_fields = []
        current_desc = ""
        
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行和注释行（以!开头的完整行）
            if not stripped or (stripped.startswith('!') and '\\' not in stripped):
                continue
            
            # 检测新对象（以大写字母或数字开头，包含冒号或逗号）
            if not line.startswith((' ', '\t')) and (',' in stripped or ';' in stripped):
                # 保存之前的对象
                if current_object:
                    self.objects_metadata[current_object] = {
                        'fields': current_fields,
                        'description': current_desc
                    }
                
                # 解析新对象名
                obj_match = re.match(r'^([A-Za-z0-9:_-]+)', stripped)
                if obj_match:
                    current_object = obj_match.group(1).rstrip(',;')
                    current_fields = []
                    current_desc = ""
            
            # 提取字段（以 \\field 或 A/N 开头）
            elif current_object:
                # 字段定义通常是 \field Name 或 A1, \field Name
                if '\\field' in line:
                    field_match = re.search(r'\\field\s+([^\n!]+)', line)
                    if field_match:
                        field_name = field_match.group(1).strip()
                        # 标准化字段名（用下划线替换空格）
                        field_name = field_name.replace(' ', '_')
                        current_fields.append(field_name)
                
                # 提取对象描述
                if '\\memo' in line and not current_desc:
                    desc_match = re.search(r'\\memo\s+([^\n]+)', line)
                    if desc_match:
                        current_desc = desc_match.group(1).strip()
        
        # 保存最后一个对象
        if current_object:
            self.objects_metadata[current_object] = {
                'fields': current_fields,
                'description': current_desc
            }
        
        print(f"[OK] 知识库已加载: {len(self.objects_metadata)} 个对象类型")
    
    def _analyze_modifiable_fields(self, idf_path: str):
        """
        分析IDF文件中各对象类型的哪些字段包含可修改的数值
        这解决了"提出建议但无法执行"的问题：某些对象虽然存在，但其字段都是引用或关键字
        
        关键改进：对于关键词映射中的所有对象类型，都进行分析，而不仅限于IDF中出现的对象
        """
        try:
            from eppy.modeleditor import IDF
            IDF.setiddname(self.idd_path)
            idf = IDF(idf_path)
            
            # 收集需要分析的所有对象类型：包括IDF中出现的以及关键词映射中引用的
            objects_to_analyze = set(idf.idfobjects.keys())
            
            # 添加关键词映射中的所有对象类型
            for obj_types in self.keyword_mapping.values():
                objects_to_analyze.update(obj_types)
            
            for obj_type in objects_to_analyze:
                if obj_type not in self._modifiable_fields_cache:
                    modifiable_fields = set()
                    
                    # 获取该类型的对象实例
                    objs = idf.idfobjects.get(obj_type, [])
                    if not objs:
                        # 如果IDF中没有该对象的实例，仍然分析其定义（从objects_metadata中）
                        # 这是关键：确保Construction等对象被识别为无可修改字段
                        obj_fields = self.objects_metadata.get(obj_type, {}).get('fields', [])
                        # 不分析字段内容，直接标记为0个可修改字段（保守做法）
                        # 除非以后需要更精确的分析
                        self._modifiable_fields_cache[obj_type] = set()
                        continue
                    
                    sample_obj = objs[0]
                    
                    # 逐个检查该对象类型的所有字段
                    for field_name in sample_obj.fieldnames:
                        try:
                            field_value = getattr(sample_obj, field_name, None)
                            
                            # 判断这个字段是否可修改：必须是数值类型
                            if field_value is None:
                                continue
                            
                            # 检查是否是数值
                            try:
                                float(str(field_value))
                                modifiable_fields.add(field_name)
                            except (ValueError, TypeError):
                                # 非数值字段（字符串、引用等）不应被修改
                                pass
                        except Exception:
                            pass
                    
                    self._modifiable_fields_cache[obj_type] = modifiable_fields
        except Exception as e:
            # IDF加载失败或其他错误，跳过分析
            pass
    
    def _get_modifiable_fields(self, object_type: str) -> set:
        """
        获取该对象类型已确认可修改的字段集合
        如果没有IDF分析数据，返回空集（安全做法）
        """
        return self._modifiable_fields_cache.get(object_type, set())
    
    def match_objects_by_keywords(self, user_request: str) -> List[str]:
        """
        根据用户请求中的关键词匹配对象类型
        返回匹配的对象类型列表（按优先级排序）
        
        ✅ 改进策略：
        - 不删除任何匹配的对象，保证匹配成功率
        - 根据关键词调整对象的优先级（排序）
        - Sizing:Zone 在无阈值意图时优先级更高
        """
        user_request_lower = user_request.lower()
        matched_objects = set()
        
        # 遍历关键词映射
        for keyword, obj_types in self.keyword_mapping.items():
            if keyword.lower() in user_request_lower:
                matched_objects.update(obj_types)
        
        # 如果没有匹配到，返回空列表
        if not matched_objects:
            return []
        
        # 验证对象是否在元数据中存在
        valid_objects = [obj for obj in matched_objects if obj in self.objects_metadata]
        
        # ✅ 根据用户意图调整优先级（排序）而不是删除
        threshold_keywords = ["阈值", "最大", "最小", "上限", "下限", "限制", "极值", "范围", "threshold", "limit", "max", "min"]
        has_threshold_intent = any(kw in user_request_lower for kw in threshold_keywords)
        
        # 如果用户提到了供风温度相关的词
        temperature_keywords = ["供风温度", "供暖", "供冷", "制冷", "空调", "hvac"]
        has_temperature_intent = any(kw in user_request_lower for kw in temperature_keywords)
        
        # ✅ 优先级排序：根据意图调整对象顺序
        if has_temperature_intent:
            if has_threshold_intent:
                # 有阈值意图：IdealLoads 优先，Sizing:Zone 次之
                priority_order = ["ZoneHVAC:IdealLoadsAirSystem", "Sizing:Zone"]
                print(f"  ✓ 检测到阈值关键词，优先推荐 ZoneHVAC:IdealLoadsAirSystem")
            else:
                # 无阈值意图：Sizing:Zone 优先，IdealLoads 次之
                priority_order = ["Sizing:Zone", "ZoneHVAC:IdealLoadsAirSystem"]
                print(f"  ✓ 未检测到阈值关键词，优先推荐 Sizing:Zone")
            
            # 按优先级排序（优先级高的对象排在前面）
            sorted_objects = []
            for obj in priority_order:
                if obj in valid_objects:
                    sorted_objects.append(obj)
            # 添加其他未排序的对象
            for obj in valid_objects:
                if obj not in sorted_objects:
                    sorted_objects.append(obj)
            valid_objects = sorted_objects
        
        return valid_objects
    
    def match_fields_by_keywords(self, object_type: str, user_request: str) -> List[str]:
        """
        根据用户请求为特定对象类型匹配字段
        返回匹配的字段列表
        
        关键改进：对于没有预定义字段映射的对象，只返回已确认可修改的字段
        这解决了"提出建议但无法修改"的问题（如Construction只有字符串引用字段）
        """
        user_request_lower = user_request.lower()
        field_mapping = self._build_field_keyword_mapping()
        
        if object_type not in field_mapping:
            # 关键修改：如果没有预定义的字段映射，只返回该对象中实际可修改的字段
            all_fields = self.objects_metadata.get(object_type, {}).get('fields', [])
            
            # 检查是否有IDF分析的可修改字段缓存数据
            if object_type in self._modifiable_fields_cache:
                # IDF已被分析，只返回实际可修改的数值字段
                modifiable = self._modifiable_fields_cache[object_type]
                return [f for f in all_fields if f in modifiable]
            else:
                # 降级策略：没有IDF分析数据，返回所有字段
                # （这会在系统应用修改时被过滤掉非数值字段）
                return all_fields
        
        matched_fields = set()
        for keyword, fields in field_mapping[object_type].items():
            if keyword.lower() in user_request_lower:
                matched_fields.update(fields)
        
        # 验证字段是否存在
        valid_fields_in_object = self.objects_metadata.get(object_type, {}).get('fields', [])
        matched_fields = [f for f in matched_fields if f in valid_fields_in_object]
        
        return matched_fields if matched_fields else valid_fields_in_object
    
    def get_object_info(self, object_type: str) -> Dict:
        """
        获取对象的详细信息
        """
        return self.objects_metadata.get(object_type, {})
    
    def get_enhanced_context(self, user_request: str) -> Dict:
        """
        为 LLM 生成增强的上下文信息，包含字段的物理含义
        返回：{
            'matched_objects': [{object_type, description, fields_with_semantics}],
            'user_intent': 用户真实意图分析
        }
        """
        matched_obj_types = self.match_objects_by_keywords(user_request)
        
        enhanced_context = {
            'matched_objects': [],
            'user_intent_analysis': self._analyze_user_intent(user_request)
        }
        
        for obj_type in matched_obj_types:
            obj_info = self.get_object_info(obj_type)
            candidate_fields = self.match_fields_by_keywords(obj_type, user_request)
            
            # 为每个字段添加语义信息
            fields_with_semantics = []
            obj_semantics = self.field_semantics.get(obj_type, {})
            
            for field in candidate_fields:
                field_info = {
                    'field_name': field,
                    'description': '未知字段'
                }
                
                # 如果有字段语义信息，添加详细描述
                if field in obj_semantics:
                    semantic = obj_semantics[field]
                    field_info.update({
                        'description': semantic.get('description', ''),
                        'unit': semantic.get('unit', ''),
                        'range': semantic.get('range', []),
                        'semantic': semantic.get('semantic', ''),
                        'related_concepts': semantic.get('related_concept', [])
                    })
                
                fields_with_semantics.append(field_info)
            
            enhanced_context['matched_objects'].append({
                'object_type': obj_type,
                'description': obj_info.get('description', ''),
                'all_fields': obj_info.get('fields', []),
                'candidate_fields': fields_with_semantics
            })
        
        return enhanced_context
    
    def _analyze_user_intent(self, user_request: str) -> Dict:
        """
        分析用户的真实意图（用户可能不了解IDF结构）
        比如"降低遮阳系数"可能意味着"增加反射率"或"减少透射率"
        """
        return analyze_user_intent(user_request)
    
    def format_for_llm(self, user_request: str) -> str:
        """
        为 LLM 格式化知识库信息
        关键：提供字段的物理含义，让LLM能够推理而不是查表
        """
        context = self.get_enhanced_context(user_request)
        
        if not context['matched_objects']:
            return "未找到匹配的对象类型，请使用完整的对象列表。"
        
        formatted = "【知识库分析】\n"
        formatted += f"用户需求: {user_request}\n\n"
        
        # 分析用户意图
        intent = context['user_intent_analysis']
        if intent.get('likely_goals'):
            formatted += f"用户目标: {', '.join(intent['likely_goals'])}\n"
            formatted += f"可能的修改方向: {', '.join(intent.get('possible_actions', []))}\n\n"
        
        formatted += "【推荐的对象类型和字段】\n"
        formatted += "=" * 60 + "\n\n"
        
        for idx, obj in enumerate(context['matched_objects'], 1):
            formatted += f"{idx}. {obj['object_type']}\n"
            if obj['description']:
                formatted += f"   说明: {obj['description']}\n\n"
            
            formatted += f"   该对象的可修改字段:\n"
            for field_info in obj['candidate_fields']:
                formatted += f"   • {field_info['field_name']}\n"
                formatted += f"     含义: {field_info.get('description', '未知')}\n"
                if field_info.get('unit'):
                    formatted += f"     单位: {field_info['unit']}\n"
                if field_info.get('range'):
                    formatted += f"     范围: {field_info['range'][0]} - {field_info['range'][1]}\n"
                if field_info.get('semantic'):
                    formatted += f"     物理意义: {field_info['semantic']}\n"
                formatted += "\n"
            
            formatted += "\n"
        
        formatted += "=" * 60 + "\n"
        formatted += "【重要提示】\n"
        formatted += "上面的字段列表是初步候选。请根据用户的真实意图和每个字段的物理含义，\n"
        formatted += "推理应该修改哪个字段，以及是增加还是减少该字段的值。\n"
        formatted += "注意：某些用户概念（如'遮阳系数'）在IDF中可能不是单一字段，\n"
        formatted += "而是多个字段（透射率、反射率）的组合，需要根据物理原理来判断。\n"
        
        return formatted

    def build_system_prompt(self) -> str:
        """构建LLM系统提示词。"""
        return build_system_prompt()

    def build_user_prompt(
        self,
        user_request: str,
        kb_context: Dict,
        field_usage_summary: str,
        min_categories: int,
        min_modifications: int,
        max_modifications: int,
        enable_novelty_constraints: bool,
        base_novelty_directive: str,
    ) -> str:
        """构建LLM用户提示词。"""
        return build_user_prompt(
            user_request=user_request,
            kb_context=kb_context,
            field_usage_summary=field_usage_summary,
            min_categories=min_categories,
            min_modifications=min_modifications,
            max_modifications=max_modifications,
            enable_novelty_constraints=enable_novelty_constraints,
            base_novelty_directive=base_novelty_directive,
            is_numeric_value=is_numeric_value,
            is_special_value=is_special_value,
        )

    def extract_plan_field_keys(self, plan: Dict) -> Set[str]:
        """从LLM计划中提取标准化字段键集合 OBJECT.FIELD。"""
        return extract_plan_field_keys(plan)

    def build_novelty_directive(
        self,
        last_round_fields: Set[str],
        field_modification_history: Dict[str, int],
        rejected_plan_keys: Set[str] = None,
        max_items: int = 10,
    ) -> str:
        """构建给LLM的低频/未出现字段约束。"""
        return build_novelty_directive(
            last_round_fields=last_round_fields,
            field_modification_history=field_modification_history,
            rejected_plan_keys=rejected_plan_keys,
            max_items=max_items,
        )

    def build_retry_anti_repeat_directive(
        self,
        last_round_fields: Set[str],
        rejected_plan_keys: Set[str] = None,
        max_items: int = 12,
    ) -> str:
        """构建重试反重复约束。"""
        return build_retry_anti_repeat_directive(
            last_round_fields=last_round_fields,
            rejected_plan_keys=rejected_plan_keys,
            max_items=max_items,
        )

    def get_field_usage_summary(
        self,
        field_modification_history: Dict[str, int],
        max_high_freq: int = 10,
        max_low_freq: int = 10,
    ) -> str:
        """生成字段使用频率摘要，用于LLM prompt。"""
        return get_field_usage_summary(
            field_modification_history=field_modification_history,
            max_high_freq=max_high_freq,
            max_low_freq=max_low_freq,
        )

    def contains_threshold_intent(self, text: str) -> bool:
        """判断文本是否包含阈值/上下限语义。"""
        return contains_threshold_intent(text)

    def get_temperature_mapping_issue(self, plan: Dict, user_request: str) -> Optional[str]:
        """校验温度建议是否符合“设计温度优先、阈值按语义启用”的规则。"""
        return get_temperature_mapping_issue(plan, user_request)

    def check_field_diversity(
        self,
        plan: Dict,
        last_round_fields: Set[str],
        field_modification_history: Dict[str, int],
    ) -> Tuple[bool, Optional[str]]:
        """检查优化方案的字段多样性。"""
        return check_field_diversity(
            plan=plan,
            last_round_fields=last_round_fields,
            field_modification_history=field_modification_history,
        )

    def is_special_value(self, value) -> bool:
        """检测字段值是否为EnergyPlus特殊关键字（不应修改）。"""
        return is_special_value(value)

    def is_numeric_value(self, value) -> bool:
        """判断字段值是否为可计算的数值。"""
        return is_numeric_value(value)

    def build_balanced_expression_for_field(self, object_type: str, field_name: str) -> str:
        """为指定字段构造保守平衡的表达式（用于多样性补充）。"""
        return build_balanced_expression_for_field(object_type, field_name)


# 测试代码
if __name__ == "__main__":
    kb = EnergyPlusKnowledgeBase("Energy+.idd")
    
    # 测试用例
    test_requests = [
        "提高照明效率，降低照明功率密度",
        "提高墙体保温性能，降低外墙材料的导热系数",
        "降低夏季太阳辐射得热，减少窗户的遮阳系数",
        "减少空气渗透热损失，降低渗透率",
        "优化室内设备热负荷，降低设备功率密度"
    ]
    
    for req in test_requests:
        print("=" * 60)
        print(f"用户需求: {req}")
        print("-" * 60)
        print(kb.format_for_llm(req))
