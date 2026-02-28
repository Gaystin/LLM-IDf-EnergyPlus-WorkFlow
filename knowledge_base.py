"""
EnergyPlus IDF 知识库模块
用于从 IDD 文件提取结构化知识，并提供语义匹配功能，
帮助 LLM 准确识别用户需求对应的 object 和 field。
"""
import json
import re
from typing import Dict, List, Optional, Tuple


class EnergyPlusKnowledgeBase:
    """EnergyPlus 知识库：解析 IDD 并提供智能匹配"""
    
    def __init__(self, idd_path: str):
        self.idd_path = idd_path
        self.objects_metadata = {}  # {object_type: {fields: [...], description: ...}}
        self.keyword_mapping = self._build_keyword_mapping()
        self.field_semantics = self._build_field_semantics()  # 字段语义库
        self._parse_idd()
    
    def _build_keyword_mapping(self) -> Dict[str, List[str]]:
        """
        构建关键词到对象类型的映射库
        这是一个语义知识库，将常见用户描述映射到标准 EnergyPlus 对象
        """
        return {
            # 照明相关
            "照明": ["Lights"],
            "照明功率": ["Lights"],
            "照明密度": ["Lights"],
            "灯光": ["Lights"],
            "lighting": ["Lights"],
            "light": ["Lights"],
            
            # 墙体材料相关
            "墙体": ["Material", "Material:NoMass", "Construction"],
            "外墙": ["Material", "Material:NoMass", "Construction"],
            "墙": ["Material", "Material:NoMass", "Construction"],
            "材料": ["Material", "Material:NoMass"],
            "导热系数": ["Material", "Material:NoMass"],
            "热阻": ["Material:NoMass"],
            "保温": ["Material", "Material:NoMass"],
            "wall": ["Material", "Material:NoMass", "Construction"],
            "material": ["Material", "Material:NoMass"],
            
            # 窗户相关
            "窗": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing", "FenestrationSurface:Detailed"],
            "窗户": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing", "FenestrationSurface:Detailed"],
            "玻璃": ["WindowMaterial:Glazing", "WindowMaterial:SimpleGlazingSystem"],
            "遮阳": ["WindowMaterial:Glazing", "Shading:Building:Detailed"],
            "遮阳系数": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
            "太阳辐射": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
            "SHGC": ["WindowMaterial:SimpleGlazingSystem"],
            "window": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
            "glazing": ["WindowMaterial:Glazing", "WindowMaterial:SimpleGlazingSystem"],
            
            # 渗透相关
            "渗透": ["ZoneInfiltration:DesignFlowRate"],
            "空气渗透": ["ZoneInfiltration:DesignFlowRate"],
            "渗透率": ["ZoneInfiltration:DesignFlowRate"],
            "infiltration": ["ZoneInfiltration:DesignFlowRate"],
            
            # 设备相关
            "设备": ["ElectricEquipment", "GasEquipment"],
            "电气设备": ["ElectricEquipment"],
            "设备功率": ["ElectricEquipment", "GasEquipment"],
            "设备密度": ["ElectricEquipment", "GasEquipment"],
            "equipment": ["ElectricEquipment", "GasEquipment"],
            
            # 人员相关
            "人员": ["People"],
            "人员密度": ["People"],
            "people": ["People"],
            
            # HVAC相关
            "空调": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
            "暖通": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
            "供暖": ["Heating:DesignDay", "ZoneHVAC:IdealLoadsAirSystem"],
            "制冷": ["Cooling:DesignDay", "ZoneHVAC:IdealLoadsAirSystem"],
            "hvac": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
        }
    
    def _build_field_keyword_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """
        构建粗粒度的关键词到字段的映射（仅用于初步过滤，最终由LLM推理）
        格式: {object_type: {keyword: [field_names]}}
        注意：这不是硬编码的参数映射，而是帮助过滤候选字段的工具
        """
        return {
            "Lights": {
                "功率": ["Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
                "密度": ["Watts_per_Floor_Area", "Watts_per_Person"],
                "瓦特": ["Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
                "watts": ["Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
                "power": ["Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
            },
            "Material": {
                "导热": ["Conductivity"],
                "导热系数": ["Conductivity"],
                "热导率": ["Conductivity"],
                "conductivity": ["Conductivity"],
                "厚度": ["Thickness"],
                "thickness": ["Thickness"],
                "密度": ["Density"],
                "density": ["Density"],
            },
            "Material:NoMass": {
                "热阻": ["Thermal_Resistance"],
                "resistance": ["Thermal_Resistance"],
            },
            "WindowMaterial:SimpleGlazingSystem": {
                "U值": ["UFactor"],
                "传热系数": ["UFactor"],
                "ufactor": ["UFactor"],
            },
            "WindowMaterial:Glazing": {
                "光学": ["Solar_Transmittance_at_Normal_Incidence", 
                         "Front_Side_Solar_Reflectance_at_Normal_Incidence", 
                         "Back_Side_Solar_Reflectance_at_Normal_Incidence"],
                "厚度": ["Thickness"],
            },
            "ZoneInfiltration:DesignFlowRate": {
                "渗透": ["Air_Changes_per_Hour", "Flow_Rate_per_Exterior_Surface_Area"],
                "渗透率": ["Air_Changes_per_Hour", "Flow_Rate_per_Exterior_Surface_Area"],
                "换气": ["Air_Changes_per_Hour"],
                "infiltration": ["Air_Changes_per_Hour", "Flow_Rate_per_Exterior_Surface_Area"],
            },
            "ElectricEquipment": {
                "功率": ["Watts_per_Floor_Area", "Watts_per_Person"],
                "密度": ["Watts_per_Floor_Area", "Watts_per_Person"],
                "equipment": ["Watts_per_Floor_Area", "Watts_per_Person"],
            },
        }
    
    def _build_field_semantics(self) -> Dict[str, Dict[str, Dict]]:
        """
        构建字段语义库：为每个对象的字段提供物理含义、数据范围、修改建议等
        这样LLM可以理解字段的实际含义，而不是靠硬编码的映射
        格式: {object_type: {field_name: {description, unit, range, semantic}}}
        """
        return {
            "WindowMaterial:Glazing": {
                "Solar_Transmittance_at_Normal_Incidence": {
                    "description": "太阳光透射率：表示有多少百分比的太阳光能透射进入室内",
                    "unit": "无量纲 (0-1)",
                    "range": [0, 1],
                    "semantic": "值越高，越多太阳热进入室内",
                    "related_concept": ["透射率", "太阳透射", "透光性"],
                },
                "Front_Side_Solar_Reflectance_at_Normal_Incidence": {
                    "description": "前面太阳反射率：表示玻璃前面反射的太阳光比例，用于遮挡太阳辐射",
                    "unit": "无量纲 (0-1)",
                    "range": [0, 1],
                    "semantic": "值越高，越多太阳光被反射回去，遮阳效果越好",
                    "related_concept": ["遮阳系数", "太阳遮挡", "隔热", "反射率"],
                },
                "Back_Side_Solar_Reflectance_at_Normal_Incidence": {
                    "description": "后面太阳反射率：表示玻璃后面反射的太阳光比例",
                    "unit": "无量纲 (0-1)",
                    "range": [0, 1],
                    "semantic": "值越高，越多太阳光被反射，遮阳效果越好",
                    "related_concept": ["遮阳系数", "太阳遮挡"],
                },
                "Thickness": {
                    "description": "玻璃厚度",
                    "unit": "mm",
                    "range": [0, 100],
                    "semantic": "厚度越大，隔音隔热效果越好，但成本增加",
                    "related_concept": ["厚度", "保温"],
                },
            },
            "WindowMaterial:SimpleGlazingSystem": {
                "Solar_Heat_Gain_Coefficient": {
                    "description": "太阳热得系数(SHGC)：综合考虑透射和吸收，表示通过窗户进入室内的太阳热比例",
                    "unit": "无量纲 (0-1)",
                    "range": [0, 1],
                    "semantic": "值越低，进入室内的太阳热越少，夏季隔热效果越好",
                    "related_concept": ["遮阳系数", "SHGC", "太阳热得系数", "隔热"],
                },
                "UFactor": {
                    "description": "传热系数(U值)：表示窗户的保温性能，值越低隔热性能越好",
                    "unit": "W/m²K",
                    "range": [0.1, 6],
                    "semantic": "值越低，保温性能越好，冬季热损失越少",
                    "related_concept": ["U值", "传热系数", "保温"],
                },
            },
            "Lights": {
                "Watts_per_Floor_Area": {
                    "description": "单位楼板面积的照明功率密度",
                    "unit": "W/m²",
                    "range": [0, 50],
                    "semantic": "值越高，照明越亮，耗电越多",
                    "related_concept": ["照明密度", "功率密度"],
                },
                "Watts_per_Person": {
                    "description": "人均照明功率",
                    "unit": "W/person",
                    "range": [0, 100],
                    "semantic": "值越高，单个人需要的照明功率越大",
                    "related_concept": ["人均功率"],
                },
                "Lighting_Level": {
                    "description": "目标照度（亮度水平）",
                    "unit": "lux",
                    "range": [0, 2000],
                    "semantic": "值越高，照度越高，通常需要更多功率",
                    "related_concept": ["照度", "亮度"],
                },
            },
            "ZoneInfiltration:DesignFlowRate": {
                "Air_Changes_per_Hour": {
                    "description": "换气次数：每小时房间空气被完全交换的次数",
                    "unit": "次/小时",
                    "range": [0, 10],
                    "semantic": "值越高，渗透越严重，冬季热损失越大",
                    "related_concept": ["换气", "渗透率", "ACH"],
                },
                "Flow_Rate_per_Exterior_Surface_Area": {
                    "description": "单位外表面积的渗透流量",
                    "unit": "m³/s-m²",
                    "range": [0, 0.01],
                    "semantic": "值越高，通过单位外墙面积的漏风越多",
                    "related_concept": ["渗透", "外墙面积渗透"],
                },
            },
        }
    
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
    
    def match_objects_by_keywords(self, user_request: str) -> List[str]:
        """
        根据用户请求中的关键词匹配对象类型
        返回匹配的对象类型列表（按相关性排序）
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
        
        return valid_objects
    
    def match_fields_by_keywords(self, object_type: str, user_request: str) -> List[str]:
        """
        根据用户请求为特定对象类型匹配字段
        返回匹配的字段列表
        """
        user_request_lower = user_request.lower()
        field_mapping = self._build_field_keyword_mapping()
        
        if object_type not in field_mapping:
            # 如果没有预定义的字段映射，返回该对象的所有字段
            return self.objects_metadata.get(object_type, {}).get('fields', [])
        
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
        intent = {
            'raw_request': user_request,
            'likely_goals': [],
            'possible_actions': []
        }
        
        request_lower = user_request.lower()
        
        # 一些常见的用户表述模式和其真实含义的映射
        user_intent_patterns = {
            # 太阳辐射和遮阳相关
            '降低.*太阳.*辐射': {
                'goals': ['减少太阳热进入室内', '提高夏季隔热能力'],
                'actions': ['增加反射率', '减少透射率', '提高SHGC'],
            },
            '减少.*遮阳': {
                'goals': ['减少遮阳效果', '增加光透入'],
                'actions': ['降低反射率', '提高透射率', '降低SHGC'],
            },
            '降低.*遮阳': {
                'goals': ['减少太阳热进入', '增加遮阳效果'],
                'actions': ['增加反射率', '减少透射率', '降低SHGC'],
            },
            '提高.*隔热': {
                'goals': ['增加隔热性能', '减少热传递'],
                'actions': ['减少透射率', '增加反射率', '增加厚度'],
            },
            '改善.*保温': {
                'goals': ['提高保温性能', '减少热损失'],
                'actions': ['增加热阻', '降低导热系数', '增加厚度'],
            },
            # 渗透和通风相关
            '降低.*渗透': {
                'goals': ['减少冬季热损失', '提高气密性'],
                'actions': ['减少换气次数', '减少单位面积渗透量'],
            },
            '减少.*空气.*渗透': {
                'goals': ['减少冬季热损失', '提高气密性'],
                'actions': ['减少换气次数', '减少单位面积渗透量'],
            },
            # 照明相关
            '降低.*照明': {
                'goals': ['减少照明能耗', '降低照明功率'],
                'actions': ['降低照明功率密度', '降低人均照明功率'],
            },
            '提高.*照明.*效率': {
                'goals': ['提高照明效率', '在相同亮度下减少功耗'],
                'actions': ['降低照明功率密度'],
            },
        }
        
        # 查看请求是否匹配任何模式
        for pattern, analysis in user_intent_patterns.items():
            import re
            if re.search(pattern, request_lower):
                intent['likely_goals'] = analysis.get('goals', [])
                intent['possible_actions'] = analysis.get('actions', [])
                break
        
        # 如果没有匹配到特定模式，尝试关键词匹配
        if not intent['likely_goals']:
            keyword_intent_map = {
                '降低': {
                    'goals': ['降低该参数的值或影响'],
                    'actions': ['减少相关字段值', '提高相关阻力']
                },
                '提高': {
                    'goals': ['提高该参数的值或影响'],
                    'actions': ['增加相关字段值', '降低相关阻力']
                },
                '减少': {
                    'goals': ['减少相关影响'],
                    'actions': ['降低相关参数值']
                },
            }
            
            for keyword, analysis in keyword_intent_map.items():
                if keyword in request_lower:
                    intent['likely_goals'] = analysis.get('goals', [])
                    intent['possible_actions'] = analysis.get('actions', [])
                    break
        
        return intent
    
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
