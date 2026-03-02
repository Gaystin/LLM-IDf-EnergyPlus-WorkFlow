import json
import os
import re
import sqlite3
import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path
from eppy.modeleditor import IDF
from openai import OpenAI
from knowledge_base import EnergyPlusKnowledgeBase

class EnergyPlusOptimizationIterator:
    """
    EnergyPlus 参数优化迭代系统
    包含5次迭代的自动化优化流程
    """
    def __init__(self, idf_path, idd_path, api_key_path, epw_path="weather.epw", log_dir="optimization_logs"):
        self.idf_path = idf_path
        self.idd_path = idd_path
        self.epw_path = epw_path
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化日志系统
        self.logger = self._setup_logging()
        
        # 验证文件存在
        if not os.path.exists(idf_path): 
            self.logger.error(f"IDF file not found: {idf_path}")
            raise FileNotFoundError(f"IDF file not found: {idf_path}")
        if not os.path.exists(idd_path): 
            self.logger.error(f"IDD file not found: {idd_path}")
            raise FileNotFoundError(f"IDD file not found: {idd_path}")
        if not os.path.exists(epw_path):
            self.logger.error(f"EPW file not found: {epw_path}")
            raise FileNotFoundError(f"EPW file not found: {epw_path}")
        
        # 设置 IDD 并加载 IDF
        try:
            IDF.setiddname(idd_path)
            self.base_idf = IDF(idf_path)
            self.logger.info(f"✓ IDF文件加载成功: {idf_path}")
        except Exception as e:
            self.logger.error(f"加载 IDF/IDD 失败: {e}")
            raise

        # 初始化知识库
        try:
            self.knowledge_base = EnergyPlusKnowledgeBase(idd_path)
            self.logger.info("✓ 知识库加载成功")
        except Exception as e:
            self.logger.warning(f"知识库加载失败: {e}，将使用传统模式")
            self.knowledge_base = None

        # 加载 API Key
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
            self.logger.info("✓ OpenAI API客户端初始化成功")
        except Exception as e:
            self.logger.error(f"加载 API Key 失败: {e}")
            self.client = None

        # 定位 EnergyPlus
        self.eplus_exe = self._locate_energyplus()
        if not self.eplus_exe:
            self.logger.error("未找到 EnergyPlus 安装")
            raise RuntimeError("EnergyPlus installation not found")
        
        # 追踪最优值
        self.best_metrics = None
        self.iteration_history = []  # 保存所有迭代的数据
    
    def _setup_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(self.log_dir, f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _locate_energyplus(self):
        """定位EnergyPlus安装位置"""
        common_paths = [
            r"C:\EnergyPlusV24-2-0",
            r"C:\EnergyPlusV24-1-0",
            r"C:\EnergyPlusV23-2-0",
            r"C:\EnergyPlusV23-1-0",
            r"C:\EnergyPlusV22-2-0",
            r"C:\EnergyPlus",
        ]
        
        for path in common_paths:
            exe_path = os.path.join(path, "energyplus.exe")
            if os.path.exists(exe_path):
                self.logger.info(f"✓ 找到 EnergyPlus: {path}")
                return exe_path
        
        return None
    
    def run_energyplus_simulation(self, idf_path, output_dir, run_name):
        """运行EnergyPlus模拟，返回输出目录"""
        self.logger.info(f"【EnergyPlus模拟】开始运行模拟: {run_name}")
        
        try:
            # 创建输出目录
            run_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            
            # 复制IDF文件到运行目录
            idf_name = os.path.splitext(os.path.basename(idf_path))[0]
            run_idf = os.path.join(run_dir, f"{idf_name}.idf")
            with open(idf_path, 'r', encoding='utf-8') as src:
                with open(run_idf, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            # 运行EnergyPlus
            cmd = [
                self.eplus_exe,
                "-w", self.epw_path,
                "-d", run_dir,
                "-r",  # 生成SQL结果
                run_idf
            ]
            
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.logger.info(f"✓ 模拟成功: {run_name}")
                self.logger.info(f"输出目录: {run_dir}")
                return run_dir
            else:
                self.logger.error(f"✗ 模拟失败: {run_name}")
                self.logger.error(f"错误信息: {result.stderr}")
                
                # 尝试读取错误文件
                err_file = os.path.join(run_dir, "eplusout.err")
                if os.path.exists(err_file):
                    with open(err_file, 'r', encoding='latin-1') as f:
                        err_content = f.read()
                        self.logger.error(f"EnergyPlus 错误:\n{err_content}")
                
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ 模拟超时: {run_name}")
            return None
        except Exception as e:
            self.logger.error(f"✗ 模拟异常: {run_name} - {e}")
            return None
    
    def extract_energy_metrics(self, sim_output_dir):
        """从SQLite数据库提取能耗数据"""
        try:
            # 查找SQL数据库文件
            sql_file = os.path.join(sim_output_dir, "eplusout.sql")
            
            if not os.path.exists(sql_file):
                self.logger.error(f"SQL文件不存在: {sql_file}")
                return None
            
            # 连接数据库并查询
            conn = sqlite3.connect(sql_file)
            cursor = conn.cursor()
            
            metrics = {}
            
            # 查询1: 总建筑能耗 (kWh)
            try:
                cursor.execute(
                    "SELECT Value FROM TablesData WHERE ReportVariableData_TableName='Site Outdoor Air Drybulb Temperature' LIMIT 1"
                )
                # 实际应该查询能耗相关字段，这里用示例
                # 正确的查询应该是针对能耗变量的
                pass
            except:
                pass
            
            # 使用eppy读取结果文件替代SQL查询
            result_file = os.path.join(sim_output_dir, "eplusout.eio")
            energy_kwh = self._parse_eio_file(result_file)
            
            conn.close()
            
            if energy_kwh:
                self.logger.info(f"✓ 提取能耗数据成功")
                return energy_kwh
            else:
                self.logger.warning(f"⚠ 无法提取能耗数据")
                return None
                
        except Exception as e:
            self.logger.error(f"✗ 提取能耗数据异常: {e}")
            return None
    
    def _parse_eio_file(self, eio_file):
        """从EIO文件解析能耗数据"""
        try:
            if not os.path.exists(eio_file):
                self.logger.warning(f"EIO文件不存在: {eio_file}")
                return None
            
            metrics = {
                'total_site_energy_kwh': 0,
                'eui_kwh_per_m2': 0,
                'total_cooling_kwh': 0,
                'total_heating_kwh': 0
            }
            
            with open(eio_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    
                    # 解析关键行
                    if 'Total Site Energy' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                val = float(parts[1].strip())
                                metrics['total_site_energy_kwh'] = val / 1000.0 if val > 10000 else val
                        except:
                            pass
                    
                    elif 'Energy per Total Building Area' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                val = float(parts[1].strip())
                                metrics['eui_kwh_per_m2'] = val
                        except:
                            pass
                    
                    elif 'Total Cooling' in line and 'Energy' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                val = float(parts[1].strip())
                                metrics['total_cooling_kwh'] = val / 1000.0 if val > 10000 else val
                        except:
                            pass
                    
                    elif 'Total Heating' in line and 'Energy' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                val = float(parts[1].strip())
                                metrics['total_heating_kwh'] = val / 1000.0 if val > 10000 else val
                        except:
                            pass
            
            return metrics if any(metrics.values()) else None
            
        except Exception as e:
            self.logger.error(f"✗ 解析EIO文件异常: {e}")
            return None

    def get_idf_object_summary(self):
        """
        提取仅包含对象类型、数量和示例名称的概览，用于第一阶段对象选择，减少 token。
        """
        summary = {}
        for obj_type in self.base_idf.idfobjects:
            objs = self.base_idf.idfobjects[obj_type]
            if len(objs) > 0:
                summary[obj_type] = {
                    "count": len(objs),
                    "example_names": [getattr(o, 'Name', 'N/A') for o in objs[:3]]
                }
        return summary


    def generate_object_plan(self, user_request):
        """
        第一阶段：仅基于对象概览（不含字段列表）生成候选对象类型，降低 token。
        返回 JSON：{clarification_needed, question, options:[{object_type}], modifications:[]}
        """
        if not self.client:
            print("API Client 未初始化，无法调用 LLM。")
            return None

        object_summary = self.get_idf_object_summary()
        # 调试输出：第一阶段喂给 LLM 的对象类型列表
        try:
            obj_list = sorted(object_summary.keys())
            print("\n[DEBUG] 第一阶段对象类型列表（传递给 LLM）:")
            print(", ".join(obj_list))
        except Exception:
            pass
        system_prompt = """
你是 EnergyPlus 对象选择助手。只输出严格 JSON，不要解释。

【目标】仅根据对象类型列表（含数量与示例名称），为用户需求挑选可能相关的 object_type 候选项。
【约束】不输出字段信息；若不唯一，设置 clarification_needed=true 并给出多个 object_type 选项。

输出格式：
{
  "clarification_needed": true,
  "question": "请选择最相关的对象类型",
  "options": [ {"object_type": "Lights"}, {"object_type": "ElectricEquipment"} ],
  "modifications": []
}
"""
        user_prompt = f"""
用户需求（REQUESTS）: "{user_request}"

对象概览（仅对象类型、数量、示例名称）：
{json.dumps(object_summary, indent=2, ensure_ascii=False)}

请返回候选对象类型列表，尽量覆盖所有可能性。
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            plan = json.loads(response.choices[0].message.content)
            # 打印 token 使用统计
            usage = response.usage
            print(f"[Token] 第一阶段（对象选择）: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            return plan
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return None

    def generate_field_plan(self, user_request, object_type):
        """
        第二阶段：仅针对用户选择的 object_type，提供该对象的字段列表给 LLM，让其给出候选字段或直接修改方案。
        返回 JSON：当有多个字段时设置 clarification_needed=true 并在 options 中给出 fields。
        """
        if not self.client:
            print("API Client 未初始化，无法调用 LLM。")
            return None

        # 获取该对象的字段列表
        if object_type not in self.base_idf.idfobjects or len(self.base_idf.idfobjects[object_type]) == 0:
            print(f"对象类型 '{object_type}' 不存在或没有实例。")
            return None
        fields = self.base_idf.idfobjects[object_type][0].fieldnames

        system_prompt = """
你是 EnergyPlus 字段选择助手。只输出严格 JSON，不要解释。

【目标】基于给定的对象类型与其字段列表，挑选满足用户需求的字段集合。
【约束】字段来自提供的列表；若不唯一，设置 clarification_needed=true 并在 options 中列出多个字段。

输出示例：
{
  "clarification_needed": true,
  "question": "该对象有多个相关字段，请选择需要修改的字段",
  "options": [{"object_type": "Lights", "fields": ["Watts_per_Floor_Area", "Watts_per_Person"]}],
  "modifications": []
}
"""
        user_prompt = f"""
用户需求（REQUESTS）: "{user_request}"
对象类型：{object_type}
字段列表（仅该对象）：{json.dumps(fields, ensure_ascii=False)}
请返回候选字段（或直接给出 modifications）。
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            plan = json.loads(response.choices[0].message.content)
            # 打印 token 使用统计
            usage = response.usage
            print(f"[Token] 第二阶段（字段选择，对象={object_type}）: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            return plan
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return None

    def generate_plan_with_knowledge_base(self, user_request):
        """
        【新方法】使用知识库增强检索，直接生成修改方案，减少交互
        利用知识库预先匹配对象和字段，只将相关的候选项传递给 LLM，
        让 LLM 能够一次性准确识别并生成完整的修改方案。
        """
        if not self.client:
            print("API Client 未初始化，无法调用 LLM。")
            return None
        
        if not self.knowledge_base:
            print("知识库未加载，退回传统模式。")
            return self.generate_object_plan(user_request)
        
        # 使用知识库获取增强的上下文
        enhanced_context = self.knowledge_base.get_enhanced_context(user_request)
        
        if not enhanced_context['matched_objects']:
            print("⚠ 知识库未找到匹配对象，退回传统模式。")
            return self.generate_object_plan(user_request)
        
        # 打印知识库匹配结果
        print("\n【知识库匹配结果】")
        for obj in enhanced_context['matched_objects']:
            candidate_count = len(obj['candidate_fields'])
            print(f"  ✓ {obj['object_type']}: {candidate_count} 个候选字段及其语义信息")
        
        # 构建详细的字段语义信息供 LLM 参考
        kb_context = {
            'user_intent': enhanced_context['user_intent_analysis'],
            'candidates': []
        }
        
        for obj in enhanced_context['matched_objects']:
            obj_type = obj['object_type']
            candidate_fields = obj['candidate_fields']

            # 按对象的 Calculation Method 过滤字段，避免选错计算方式
            if obj_type in self.base_idf.idfobjects and len(self.base_idf.idfobjects[obj_type]) > 0:
                sample_obj = self.base_idf.idfobjects[obj_type][0]
                candidate_field_names = [f['field_name'] for f in candidate_fields]
                filtered_names = self._filter_fields_by_method(obj_type, sample_obj, candidate_field_names)
                if filtered_names:
                    candidate_fields = [f for f in candidate_fields if f['field_name'] in filtered_names]

            # 构建字段语义信息
            field_semantics_list = []
            for field_info in candidate_fields:
                field_semantics_list.append({
                    'field_name': field_info['field_name'],
                    'description': field_info.get('description', ''),
                    'unit': field_info.get('unit', ''),
                    'range': field_info.get('range', []),
                    'semantic': field_info.get('semantic', ''),
                    'related_concepts': field_info.get('related_concepts', [])
                })
            
            # 获取对象实例的当前值
            current_values = {}
            if obj_type in self.base_idf.idfobjects and len(self.base_idf.idfobjects[obj_type]) > 0:
                sample_obj = self.base_idf.idfobjects[obj_type][0]
                for field_info in candidate_fields:
                    field = field_info['field_name']
                    val = getattr(sample_obj, field, None)
                    current_values[field] = str(val) if val is not None else "未设置"
            
            kb_context['candidates'].append({
                'object_type': obj_type,
                'description': obj['description'],
                'object_count': len(self.base_idf.idfobjects.get(obj_type, [])),
                'field_semantics': field_semantics_list,
                'current_sample_values': current_values
            })
        
        system_prompt = """
你是 EnergyPlus 专家。你的任务不是简单查表，而是理解用户需求的物理含义，并基于字段的物理性质进行推理。

【重要概念】
某些用户提到的参数（如"遮阳系数"）在 IDF 中可能没有直接对应字段。你需要根据物理原理推断：
- 用户说什么目标？（物理意义）
- 使用哪个字段最合适？（根据字段语义）
- 应该增加还是减少该字段的值？（根据字段的物理含义）

例如：
- 用户说"降低遮阳系数"实际上想降低太阳热进入
- 在 IDF 中，这对应两个方向：
  a) 增加 Front_Side_Solar_Reflectance_at_Normal_Incidence（反射率越高，越多光被反射出去）
  b) 减少 Solar_Transmittance_at_Normal_Incidence（透射率越低，越少光透进来）
- 你需要选择最合理的字段和修改方向

【输入信息】
1. 用户需求及其隐含的物理意图
2. 候选对象的所有字段及其物理含义（已按 Calculation Method 过滤）
3. 每个字段的当前值

【任务】
1. 分析用户的真实物理目标
2. 理解每个候选字段的物理含义
3. 推理应该修改哪些字段，以及增加还是减少
4. 生成修改方案

【输出格式】严格 JSON：
{
  "clarification_needed": false,
  "reasoning": "你的推理过程（为什么选择这些字段，为什么增加或减少）",
  "confidence": "high/medium/low",
  "modifications": [
    {
      "object_type": "对象类型",
      "name_filter": null,
      "fields": {
        "字段名": "修改表达式"
      }
    }
  ]
}

【修改表达式规则】
- 使用 existing_value 代表原值，coefficient 代表系数
- 示例：
  * "existing_value * coefficient" （减少/增加原有值）
  * "existing_value - coefficient" （各取所需）
  * "coefficient" （设置为固定值）
- 如果无法确定，设置 clarification_needed=true
"""
        
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
            user_prompt += f"  该对象的字段及其物理含义:\n"
            
            for field in candidate['field_semantics']:
                user_prompt += f"\n    • {field['field_name']}\n"
                user_prompt += f"      物理含义: {field['description']}\n"
                if field['unit']:
                    user_prompt += f"      单位: {field['unit']}\n"
                if field['range']:
                    user_prompt += f"      范围: {field['range'][0]} - {field['range'][1]}\n"
                user_prompt += f"      物理解释: {field['semantic']}\n"
                if field['related_concepts']:
                    user_prompt += f"      用户可能称呼: {', '.join(field['related_concepts'])}\n"
        
        user_prompt += f"""

【请根据以上信息进行推理】
1. 用户说的是什么物理目标？
2. 哪些字段与这个目标最相关？
3. 应该增加还是减少这些字段的值？
4. 是否所有相关对象都应该修改？

然后生成修改方案。
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            plan = json.loads(response.choices[0].message.content)
            
            # 打印 token 使用统计
            usage = response.usage
            print(f"[Token] 知识库增强模式（含推理）: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            
            # 显示推理过程和置信度
            if plan.get('reasoning'):
                print(f"\n【LLM推理过程】\n{plan['reasoning']}\n")
            if plan.get('confidence'):
                print(f"[置信度] {plan['confidence']}")
            
            return plan
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return None

    def _filter_fields_by_method(self, object_type, sample_obj, candidate_fields):
        """
        根据对象的 Calculation Method 字段过滤候选字段，避免选错计算方式。
        仅在能明确匹配时收缩字段，否则返回原候选列表。
        """
        if not candidate_fields:
            return candidate_fields

        method_field_map = {
            "Lights": ("Design_Level_Calculation_Method", {
                "WATTS/AREA": "Watts_per_Floor_Area",
                "WATTS/PERSON": "Watts_per_Person",
                "LIGHTINGLEVEL": "Lighting_Level",
            }),
            "ElectricEquipment": ("Design_Level_Calculation_Method", {
                "WATTS/AREA": "Watts_per_Floor_Area",
                "WATTS/PERSON": "Watts_per_Person",
                "EQUIPMENTLEVEL": "Design_Level",
            }),
            "ZoneInfiltration:DesignFlowRate": ("Design_Flow_Rate_Calculation_Method", {
                "FLOW/EXTERIORAREA": "Flow_Rate_per_Exterior_Surface_Area",
                "FLOW/EXTERIORWALLAREA": "Flow_Rate_per_Exterior_Surface_Area",
                "FLOW/AREA": "Flow_Rate_per_Floor_Area",
                "FLOW/ZONE": "Design_Flow_Rate",
                "AIRCHANGES/HOUR": "Air_Changes_per_Hour",
            }),
        }

        if object_type not in method_field_map:
            return candidate_fields

        method_field, value_map = method_field_map[object_type]
        method_val = getattr(sample_obj, method_field, None)
        if not method_val:
            return candidate_fields

        key = str(method_val).upper().replace(" ", "")
        target_field = value_map.get(key)
        if not target_field:
            return candidate_fields

        if target_field in candidate_fields:
            return [target_field]
        return candidate_fields

    def _apply_coefficient_to_expr(self, expr, coef):
        """
        将外部系数应用到表达式，避免重复乘导致系数被平方。
        """
        if not isinstance(expr, str):
            return expr

        expr_stripped = expr.strip()

        # 显式占位符优先
        if re.search(r"\bcoefficient\b", expr_stripped, flags=re.IGNORECASE):
            return re.sub(r"\bcoefficient\b", str(coef), expr_stripped, flags=re.IGNORECASE)

        # 仅在表达式是单纯 existing_value 时应用系数
        if re.fullmatch(r"\{?existing_value\}?", expr_stripped, flags=re.IGNORECASE):
            return f"existing_value * {coef}"

        # 如果已有 existing_value * 数值，则替换数值为系数
        m = re.search(r"existing_value\s*\*\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", expr_stripped, flags=re.IGNORECASE)
        if m:
            return re.sub(r"existing_value\s*\*\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
                          f"existing_value * {coef}",
                          expr_stripped,
                          flags=re.IGNORECASE)

        # 其他复杂表达式不强行改写
        return expr_stripped

    def interactive_select_object_and_fields(self, user_request):
        """
        两阶段交互：先对象选择（支持多选），再为每个对象进行字段选择。
        返回列表：[{object_type, fields, name_filter}, ...]
        """
        # 阶段1：对象选择（支持多选）
        obj_plan = self.generate_object_plan(user_request)
        if not obj_plan:
            return None
        options = obj_plan.get("options", [])
        if not options:
            print("✗ LLM 未提供对象候选。")
            return None
        print("\n对象候选（支持多选）：")
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {opt.get('object_type', '?')}")
        
        # 获取用户选择的对象（支持多选，逗号分隔）
        chosen_objs = []
        while True:
            try:
                choice_str = input(f"请选择对象 (1-{len(options)}，可用逗号分隔多选): ").strip()
                indices = [int(x.strip())-1 for x in choice_str.split(',') if x.strip()]
                chosen_objs = [options[i] for i in indices if 0 <= i < len(options)]
                if not chosen_objs:
                    print("✗ 请至少选择一个对象")
                    continue
                break
            except ValueError:
                print("✗ 请输入有效数字")
        
        print(f"✓ 已选择对象: {', '.join([opt.get('object_type', '?') for opt in chosen_objs])}")

        # 阶段2：为每个选中的对象进行字段选择
        selected_options = []
        for obj_opt in chosen_objs:
            selected_obj_type = obj_opt.get("object_type")
            print(f"\n━━━ 处理对象: {selected_obj_type} ━━━")
            
            # 获取该对象的候选字段
            field_plan = self.generate_field_plan(user_request, selected_obj_type)
            if not field_plan:
                print(f"⚠ 跳过 {selected_obj_type}：字段选择失败")
                continue
            
            candidate_fields = []
            if field_plan.get("clarification_needed"):
                # 从 options 中拿字段列表
                for opt in field_plan.get("options", []):
                    if opt.get("object_type") == selected_obj_type:
                        candidate_fields = opt.get("fields", [])
                        break
            else:
                mods = field_plan.get("modifications", [])
                if mods:
                    fields_obj = mods[0].get("fields", {})
                    # 处理 fields 既可能是字典也可能是列表的情况
                    if isinstance(fields_obj, dict):
                        candidate_fields = list(fields_obj.keys())
                    elif isinstance(fields_obj, list):
                        candidate_fields = fields_obj
                    else:
                        candidate_fields = []
            
            if not candidate_fields:
                # 如果 LLM 没给出，退回到全部字段
                candidate_fields = self.base_idf.idfobjects[selected_obj_type][0].fieldnames

            print(f"该对象有 {len(candidate_fields)} 个候选字段：")
            sample_obj = self.base_idf.idfobjects[selected_obj_type][0]
            for i, fld in enumerate(candidate_fields, 1):
                val = getattr(sample_obj, fld, None)
                if val is None or (isinstance(val, str) and val.strip() == ''):
                    val_str = "None"
                else:
                    val_str = str(val)
                print(f"  {i}. {fld} = {val_str}")
            
            field_input = input(f"请为 [{selected_obj_type}] 输入要修改的字段编号(可逗号多选，留空表示全选): ").strip()
            if field_input:
                try:
                    indices = [int(x.strip())-1 for x in field_input.split(',') if x.strip()]
                    chosen_fields = [candidate_fields[i] for i in indices if 0 <= i < len(candidate_fields)]
                    if not chosen_fields:
                        print("⚠ 无效输入，使用全部字段")
                        chosen_fields = candidate_fields
                except Exception:
                    print("⚠ 输入错误，使用全部字段")
                    chosen_fields = candidate_fields
            else:
                chosen_fields = candidate_fields
                print(f"✓ 使用全部字段")
            
            selected_options.append({
                "object_type": selected_obj_type,
                "fields": chosen_fields,
                "name_filter": None
            })
        
        return selected_options if selected_options else None
    
    def _do_modification_work(self, plan, output_idf_path):
        """
        【修改版】执行修改并保存。
        为了完全保留原始 IDF 的结构、缩进和注释，
        本方法采用"Eppy计算逻辑 + 纯文本替换"的混合模式。
        1. 使用 eppy 找到目标对象和原值，计算出新值。
        2. 读取原始 IDF 文本，定位特定行进行字符串替换。
        """
        # 1. 预计算所有需要修改的目标值
        # 结构: target_updates = [ {type, name, field, value}, ... ]
        target_updates = []
        
        # 使用 base_idf 查找对象并计算新值（但不修改 base_idf）
        for mod in plan["modifications"]:
            obj_type_input = mod.get("object_type")
            name_filter = mod.get("name_filter")
            fields_to_mod = mod.get("fields", {})
            
            # 查找匹配的对象类型
            target_obj_type = None
            if obj_type_input in self.base_idf.idfobjects:
                target_obj_type = obj_type_input
            else:
                normalized_input = obj_type_input.lower().replace(":", "").replace("_", "").replace(" ", "")
                for available_type in self.base_idf.idfobjects.keys():
                    norm_avail = available_type.lower().replace(":", "").replace("_", "").replace(" ", "")
                    if norm_avail == normalized_input or (norm_avail.startswith(normalized_input) and len(normalized_input) >= 5):
                        target_obj_type = available_type
                        break
            
            if not target_obj_type:
                print(f"✗ 跳过: 找不到对象类型 '{obj_type_input}'")
                continue

            for obj in self.base_idf.idfobjects[target_obj_type]:
                obj_name = getattr(obj, 'Name', '')
                if name_filter and obj_name != name_filter:
                    continue
                
                # 处理每个字段
                for field_input, value_expr in fields_to_mod.items():
                    # 字段匹配逻辑
                    clean_field = field_input.split('{')[0].strip()
                    valid_attrs = obj.fieldnames
                    target_attr = None
                    
                    # 1. 精确
                    if hasattr(obj, clean_field.replace(" ", "_")):
                        target_attr = clean_field.replace(" ", "_")
                    else:
                        # 2. 模糊匹配
                        norm_field = clean_field.lower().replace("_", "").replace(" ", "")
                        for attr in valid_attrs:
                            if attr.lower().replace("_", "").replace(" ", "") == norm_field:
                                target_attr = attr
                                break
                    
                    if target_attr:
                        # 获取原值并计算
                        old_value = getattr(obj, target_attr, 0)
                        final_value = self._evaluate_expression(value_expr, old_value)
                        
                        target_updates.append({
                            "type": target_obj_type,
                            "name": obj_name,
                            "field": target_attr,
                            "value": final_value
                        })
                        print(f"  [计划修改] {target_obj_type} ({obj_name}): {target_attr} -> {final_value}(原值: {old_value})")

        if not target_updates:
            print("没有产生有效的修改计划，跳过保存。")
            return

        # 2. 读取原始文本进行"外科手术式"替换
        try:
            with open(self.idf_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(self.idf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        new_lines = []
        current_obj_type = None
        current_obj_name = None
        in_object = False
        
        # 为了处理不区分大小写的匹配
        updates_map = {} # type -> { name -> { field -> value } }
        for item in target_updates:
            t = item['type'].upper()
            n = item['name'].upper()
            f = item['field'].upper()
            if t not in updates_map: updates_map[t] = {}
            if n not in updates_map[t]: updates_map[t][n] = {}
            updates_map[t][n][f] = item['value']

        for line in lines:
            stripped = line.strip()
            
            # 检测对象开始
            if not in_object and stripped and not stripped.startswith('!'):
                parts = stripped.split('!')[0].strip().split(',')
                possible_type = parts[0].strip().replace(';', '')
                
                if possible_type.upper() in updates_map:
                    current_obj_type = possible_type.upper()
                    in_object = True
                    current_obj_name = "N/A"
                else:
                    if ',' in stripped or ';' in stripped:
                        in_object = True
                        current_obj_type = "UNKNOWN"
                        current_obj_name = "N/A"

            # 如果在对象内，尝试匹配字段
            if in_object:
                # 检查是否结束
                if ';' in line.split('!')[0]:
                    in_object = False
                
                # 1. 尝试识别 Name 字段以锁定对象名
                if "!- Name" in line:
                    val_part = line.split('!')[0].strip()
                    val_part = val_part.replace(',', '').replace(';', '').strip()
                    current_obj_name = val_part.upper()
                
                # 2. 检查是否是我们需要修改的字段
                if current_obj_type in updates_map:
                    if '!' in line:
                        comment_part = line.split('!')[1].strip()
                        if comment_part.startswith('- '):
                            field_comment = comment_part[2:].strip().upper()
                        else:
                            field_comment = comment_part.upper()
                        
                        target_fields = updates_map[current_obj_type].get(current_obj_name, {})
                        
                        # 核心替换逻辑：匹配字段名
                        matched_val = None
                        for target_field_key, target_val in target_fields.items():
                            # 归一化比较
                            f1 = target_field_key.replace(" ", "").replace("_", "")
                            f2 = field_comment.replace(" ", "").replace("_", "")
                            
                            if f1 in f2: # 包含匹配
                                matched_val = target_val
                                break
                        
                        if matched_val is not None:
                            # 执行替换（严格保留原始空格、分隔符、注释与换行）
                            idx_bang = line.find('!')
                            if idx_bang == -1:
                                # 没有注释，保守处理：仅替换数值，尽量不改变其他字符
                                original_content = line
                                comment_full = ''
                            else:
                                original_content = line[:idx_bang]
                                comment_full = line[idx_bang:]

                            # 使用正则定位数值并替换，保留前后空格和分隔符
                            import re
                            m = re.match(r"^(\s*)([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)(\s*)([,;].*)$", original_content.rstrip('\r\n'))
                            if m:
                                leading = m.group(1)
                                # old_val = m.group(2)  # 原值，不使用，只用于定位
                                spaces_after = m.group(3)
                                rest = m.group(4)
                                new_original_content = f"{leading}{matched_val}{spaces_after}{rest}"
                            else:
                                # 兜底：如果无法精准匹配，尝试替换第一个逗号/分号之前的非空内容为新值
                                # 仍然保留原有前导空格与分隔符及其后续内容
                                parts = original_content.rstrip('\r\n').split(',', 1)
                                sep = ','
                                if len(parts) == 1:
                                    parts = original_content.rstrip('\r\n').split(';', 1)
                                    sep = ';'
                                if len(parts) == 2:
                                    # 保留前导空格
                                    leading = parts[0][:len(parts[0]) - len(parts[0].lstrip())]
                                    tail = parts[1]
                                    new_original_content = f"{leading}{matched_val}{sep}{tail}"
                                else:
                                    # 无法判断结构，直接用原行（不做替换以避免破坏格式）
                                    new_original_content = original_content.rstrip('\r\n')

                            # 重新拼接完整行，严格保留原有注释与换行
                            new_line = new_original_content + comment_full
                            line = new_line
                            print(f"✓ [文本替换] {current_obj_type}: {field_comment} -> {matched_val}")

            new_lines.append(line)

        # 3. 保存
        with open(output_idf_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"算例已保存至: {output_idf_path}")

    def _evaluate_expression(self, value, old_value):
        """
        计算单个表达式。输入原值，返回计算结果。
        处理空值、None、字符串等各种情况。
        """
        if not isinstance(value, str):
            # 非字符串，尝试数值格式化
            try:
                num = float(value)
                # 近似整数处理
                if abs(num - round(num)) < 1e-6:
                    return float(round(num))
                return round(num, 6)
            except (ValueError, TypeError):
                return value
        
        # 检测是否为表达式
        value_lower = value.lower()
        has_expr = any(var in value_lower for var in ['old_value', 'original_value', 'existing_value', 'current_value'])
        
        if not has_expr:
            # 直接尝试转换为数值
            try:
                num = float(value)
                if abs(num - round(num)) < 1e-6:
                    return float(round(num))
                return round(num, 6)
            except (ValueError, TypeError):
                return value
        
        try:
            # 获取原值的数值
            # 处理 None、空字符串、空格等情况
            if old_value is None or old_value == '' or (isinstance(old_value, str) and old_value.strip() == ''):
                old_val_num = 0.0
            else:
                try:
                    old_val_num = float(old_value)
                except (ValueError, TypeError):
                    # 如果无法转换，记录警告并使用 0
                    print(f"警告: 无法将原值 '{old_value}' 转换为数值，使用 0")
                    old_val_num = 0.0
            
            # 替换表达式中的各种变量名
            expr = value
            expr = re.sub(r'\{?old_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?original_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?existing_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?current_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            
            # 安全检查
            if any(dangerous in expr.lower() for dangerous in ['import', 'exec', 'eval', '__', 'open', 'os.']):
                print(f"错误: 表达式包含不安全操作: {expr}")
                return value
            
            # 计算
            result = eval(expr, {"__builtins__": {}}, {})
            try:
                num = float(result)
            except (ValueError, TypeError):
                return result
            # 近似整数处理 + 限定小数位，避免 20.9999999 这类误差
            if abs(num - round(num)) < 1e-6:
                return float(round(num))
            return round(num, 6)
        except Exception as e:
            print(f"错误: 计算表达式 '{value}' 失败: {e}")
            return value

    def batch_run(self, requests, output_dir="output_cases"):
        """
        批量生成算例。支持多对象选择。
        """
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        for i, req in enumerate(requests):
            print(f"\n[算例 {i+1}] 处理需求: {req}")
            # 新流程：两阶段交互以减少 token（支持多对象）
            selected_options = self.interactive_select_object_and_fields(req)
            if not selected_options:
                continue
            
            # 第二步：询问修改系数（对所有选中的对象统一应用）
            print(f"\n已选择 {len(selected_options)} 个对象：")
            for opt in selected_options:
                print(f"  • {opt.get('object_type')}: {', '.join(opt.get('fields', []))}")
            
            coefficient_str = input("\n请输入修改系数，支持以下格式:\n  单个：0.8\n  多个：0.3,0.5,0.8\n  范围：0.1-0.8(步长0.1)\n输入: ").strip()
            coefficients = self._parse_coefficients(coefficient_str)
            
            if not coefficients:
                coefficients = [0.8]  # 默认降低20%
            
            print(f"✓ 将生成 {len(coefficients)} 个案例，系数为: {coefficients}")
            
            # 第三步：对每个系数生成一个案例
            for coef_idx, coefficient in enumerate(coefficients, 1):
                # 生成输出文件名
                if len(coefficients) == 1:
                    output_file = os.path.join(output_dir, f"case_{i+1}.idf")
                else:
                    output_file = os.path.join(output_dir, f"case_{i+1}_coef_{coefficient:.2f}.idf")
                
                # 为所有选中的对象构建修改方案
                modifications = []
                for selected_option in selected_options:
                    modifications.append({
                        "object_type": selected_option.get("object_type"),
                        "name_filter": selected_option.get("name_filter"),
                        "fields": {field: f"existing_value * {coefficient}" for field in selected_option.get("fields", [])}
                    })
                
                refined_plan = {
                    "clarification_needed": False,
                    "question": None,
                    "options": [],
                    "modifications": modifications
                }
                if len(coefficients) > 1:
                    print(f"\n  生成 case_{i+1}_coef_{coefficient:.2f}.idf (修改系数: {coefficient})")
                else:
                    print(f"\n  生成 case_{i+1}.idf (修改系数: {coefficient})")
                self._execute_modifications(refined_plan, output_file)
    
    def batch_run_with_kb(self, requests, output_dir="output_cases", auto_confirm=False):
        """
        【新方法】使用知识库增强的批量生成算例，减少交互
        
        参数:
        - requests: 用户需求列表
        - output_dir: 输出目录
        - auto_confirm: 是否自动确认（True=完全自动，False=显示计划并要求确认）
        """
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        
        if not self.knowledge_base:
            print("⚠ 知识库未加载，退回传统模式")
            return self.batch_run(requests, output_dir)
        
        for i, req in enumerate(requests, 1):
            print("\n" + "=" * 60)
            print(f"[算例 {i}] 处理需求: {req}")
            print("=" * 60)
            
            # 使用知识库增强的方法生成计划
            plan = self.generate_plan_with_knowledge_base(req)
            
            if not plan:
                print("✗ 无法生成计划，跳过此需求")
                continue
            
            # 检查是否需要人工澄清
            if plan.get("clarification_needed"):
                print(f"\n⚠ LLM 需要澄清: {plan.get('question', '未知问题')}")
                # 这里可以添加交互逻辑，当前跳过
                continue
            
            modifications = plan.get("modifications", [])
            if not modifications:
                print("✗ 计划中无有效修改项，跳过")
                continue
            
            # 显示计划摘要
            print("\n【生成的修改计划】")
            for mod in modifications:
                obj_type = mod.get("object_type", "?")
                fields = mod.get("fields", {})
                print(f"  • {obj_type}: {len(fields)} 个字段")
                for field_name, expr in list(fields.items())[:3]:  # 只显示前3个
                    print(f"    - {field_name}: {expr}")
                if len(fields) > 3:
                    print(f"    ... 还有 {len(fields) - 3} 个字段")
            
            # 确认或自动执行
            if not auto_confirm:
                confirm = input("\n是否执行此计划? (y/n，直接回车=是): ").strip().lower()
                if confirm and confirm not in ['y', 'yes', '']:
                    print("✗ 用户取消，跳过此需求")
                    continue
            
            # 询问修改系数
            if auto_confirm:
                coefficient_str = "0.8"  # 默认值
                print(f"[自动模式] 使用默认系数: {coefficient_str}")
            else:
                coefficient_str = input("\n请输入修改系数 (单个/多个/范围，默认0.8): ").strip()
                if not coefficient_str:
                    coefficient_str = "0.8"
            
            coefficients = self._parse_coefficients(coefficient_str)
            if not coefficients:
                coefficients = [0.8]
            
            print(f"✓ 将生成 {len(coefficients)} 个案例，系数为: {coefficients}")
            
            # 为每个系数生成案例
            for coef in coefficients:
                # 生成输出文件名
                if len(coefficients) == 1:
                    output_file = os.path.join(output_dir, f"case_{i}.idf")
                else:
                    output_file = os.path.join(output_dir, f"case_{i}_coef_{coef:.2f}.idf")
                
                # 应用系数到计划
                coef_plan = {
                    "clarification_needed": False,
                    "modifications": []
                }
                
                for mod in modifications:
                    modified_fields = {}
                    for field_name, expr in mod.get("fields", {}).items():
                        modified_expr = self._apply_coefficient_to_expr(expr, coef)
                        modified_fields[field_name] = modified_expr
                    
                    coef_plan["modifications"].append({
                        "object_type": mod.get("object_type"),
                        "name_filter": mod.get("name_filter"),
                        "fields": modified_fields
                    })
                
                print(f"\n  生成 {os.path.basename(output_file)}")
                self._execute_modifications(coef_plan, output_file)
        
        print("\n" + "=" * 60)
        print("✓ 批量生成完成")
        print("=" * 60)
    
    def _execute_modifications(self, plan, output_idf_path):
        """
        执行实际的 IDF 修改。
        """
        return self._do_modification_work(plan, output_idf_path)
    
    def _parse_coefficients(self, coefficient_str):
        """
        解析用户输入的系数字符串，支持多种格式：
        - 单个：0.8
        - 多个：0.3,0.5,0.8
        - 范围：0.1-0.8(自动按0.1步长)、0.2-0.9/5(5个点)
        返回排序后的系数列表。
        """
        coefficient_str = coefficient_str.strip()
        if not coefficient_str:
            return []
        
        coefficients = []
        
        # 检查是否包含范围表达式 (start-end 或 start-end/count)
        if '-' in coefficient_str and ',' not in coefficient_str:
            try:
                if '/' in coefficient_str:
                    range_part, count_str = coefficient_str.rsplit('/', 1)
                    start, end = map(float, range_part.split('-'))
                    count = int(count_str)
                else:
                    start, end = map(float, coefficient_str.split('-'))
                    # 默认按 0.1 步长
                    count = int((end - start) / 0.1) + 1
                
                if start >= end:
                    return []
                
                coefficients = [round(start + i * (end - start) / (count - 1), 2) for i in range(count)]
            except Exception:
                return []
        else:
            # 多个系数，用逗号分隔
            try:
                coefficients = [round(float(x.strip()), 2) for x in coefficient_str.split(',') if x.strip()]
            except Exception:
                return []
        
        # 去重并排序
        coefficients = sorted(set(coefficients))
        return coefficients

if __name__ == "__main__":
    # ========== 可运行示例 ==========
    
    # 1. 配置文件路径
    IDF_PATH = "in.idf"
    IDD_PATH = "Energy+.idd"
    API_KEY_PATH = "api_key.txt"
    OUTPUT_DIR = "generated_cases"
    
    # 2. 创建自动化实例
    print("=" * 60)
    print("EnergyPlus 自动化算例生成系统 (知识库增强版)")
    print("=" * 60)
    
    try:
        automation = EnergyPlusOptimizationIterator(IDF_PATH, IDD_PATH, API_KEY_PATH)
        print("✓ 成功加载 IDF 和 IDD 文件")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        exit(1)
    
    # 3. 定义需求列表（可根据实际需求修改）
    DESIGN_REQUESTS = [
        "提高照明效率，降低照明功率密度",
        "提高墙体保温性能，降低外墙材料的导热系数",
        "降低夏季太阳辐射得热，减少窗户的遮阳系数",
        "减少空气渗透热损失，降低渗透率",
        "优化室内设备热负荷，降低设备功率密度"
    ]
    
    print(f"\n共有 {len(DESIGN_REQUESTS)} 个设计需求待处理")
    
    # 4. 选择运行模式
    print("\n请选择运行模式:")
    print("  1. 知识库增强模式 (推荐) - 自动匹配对象和字段，减少交互")
    print("  2. 传统交互模式 - 逐步选择对象和字段")
    
    mode = input("请输入模式编号 (1/2，默认1): ").strip()
    
    if mode == "2":
        print("\n使用传统交互模式...")
        automation.batch_run(DESIGN_REQUESTS, output_dir=OUTPUT_DIR)
    else:
        print("\n使用知识库增强模式...")
        auto_mode = input("是否启用全自动模式? (y/n，默认n): ").strip().lower()
        auto_confirm = (auto_mode in ['y', 'yes'])
        automation.batch_run_with_kb(DESIGN_REQUESTS, output_dir=OUTPUT_DIR, auto_confirm=auto_confirm)
    
    print("\n" + "=" * 60)
    print(f"✓ 所有算例已生成完成，保存在 {OUTPUT_DIR}/ 目录")
    print("=" * 60)
