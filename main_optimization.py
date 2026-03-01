"""
EnergyPlus 5-轮迭代优化闭环系统
完整实现：模拟→提取→LLM优化→修改→循环
"""
import json
import os
import re
import sqlite3
import subprocess
import logging
import sys
import shutil
from datetime import datetime
from pathlib import Path
from eppy.modeleditor import IDF
from openai import OpenAI
from knowledge_base import EnergyPlusKnowledgeBase


class EnergyPlusOptimizer:
    """EnergyPlus 5轮迭代自动优化系统"""
    
    def __init__(self, idf_path, idd_path, api_key_path, epw_path="weather.epw", log_dir="optimization_logs"):
        self.idf_path = idf_path
        self.idd_path = idd_path
        self.epw_path = epw_path
        self.log_dir = log_dir
        self.optimization_dir = "optimization_results"
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.optimization_dir, exist_ok=True)
        
        # 初始化日志
        self.logger = self._setup_logging()
        
        self.logger.info("="*80)
        self.logger.info("EnergyPlus 5轮迭代优化系统启动")
        self.logger.info("="*80)
        
        # 验证文件
        if not os.path.exists(idf_path): 
            self.logger.error(f"IDF文件不存在: {idf_path}")
            raise FileNotFoundError(f"IDF not found: {idf_path}")
        if not os.path.exists(idd_path): 
            self.logger.error(f"IDD文件不存在: {idd_path}")
            raise FileNotFoundError(f"IDD not found: {idd_path}")
        if not os.path.exists(epw_path):
            self.logger.error(f"EPW文件不存在: {epw_path}")
            raise FileNotFoundError(f"EPW not found: {epw_path}")
        
        # 加载IDF
        try:
            IDF.setiddname(idd_path)
            self.base_idf = IDF(idf_path)
            self.logger.info(f"✓ IDF文件加载成功")
        except Exception as e:
            self.logger.error(f"加载IDF失败: {e}")
            raise
        
        # 初始化知识库
        try:
            self.knowledge_base = EnergyPlusKnowledgeBase(idd_path)
            self.logger.info("✓ 知识库加载成功")
        except Exception as e:
            self.logger.warning(f"知识库加载失败: {e}")
            self.knowledge_base = None
        
        # 初始化OpenAI客户端
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
            self.logger.info("✓ OpenAI客户端初始化成功")
        except Exception as e:
            self.logger.error(f"初始化OpenAI失败: {e}")
            self.client = None
        
        # 定位EnergyPlus
        self.eplus_exe = self._locate_energyplus()
        if not self.eplus_exe:
            self.logger.error("未找到EnergyPlus安装")
            raise RuntimeError("EnergyPlus not found")
        
        # 初始化追踪数据
        self.best_metrics = None
        self.best_iteration = 0
        self.iteration_history = []
        self.current_idf_path = idf_path  # 当前工作IDF
    
    def _setup_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(
            self.log_dir, 
            f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logger = logging.getLogger(__name__)
        logger.handlers = []  # 清除之前的处理器
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
        """定位EnergyPlus安装"""
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
                self.logger.info(f"✓ 找到EnergyPlus: {path}")
                return exe_path
        
        return None
    
    def run_simulation(self, idf_path, iteration_name):
        """运行EnergyPlus模拟"""
        self.logger.info(f"\n【模拟】{iteration_name}")
        self.logger.info("-" * 80)
        
        try:
            run_dir = os.path.join(self.optimization_dir, iteration_name)
            os.makedirs(run_dir, exist_ok=True)
            
            # 复制IDF到运行目录
            idf_name = os.path.splitext(os.path.basename(idf_path))[0]
            sim_idf = os.path.join(run_dir, f"{idf_name}.idf")
            shutil.copy(idf_path, sim_idf)
            
            # 运行EnergyPlus
            cmd = [
                self.eplus_exe,
                "-w", self.epw_path,
                "-d", run_dir,
                "-r",
                sim_idf
            ]
            
            self.logger.info(f"执行EnergyPlus: {iteration_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.logger.info(f"✓ 模拟完成: {iteration_name}")
                return run_dir
            else:
                self.logger.error(f"✗ 模拟失败: {iteration_name}")
                self.logger.error(f"返回码: {result.returncode}")
                if result.stderr:
                    self.logger.error(f"错误: {result.stderr[:500]}")
                return None
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ 模拟超时: {iteration_name}")
            return None
        except Exception as e:
            self.logger.error(f"✗ 模拟异常: {str(e)}")
            return None
    
    def extract_metrics(self, sim_dir):
        """从SQLite数据库提取能耗指标"""
        try:
            sql_file = os.path.join(sim_dir, "eplusout.sql")
            if not os.path.exists(sql_file):
                self.logger.error(f"SQL文件不存在: {sql_file}")
                return None
            
            conn = sqlite3.connect(sql_file)
            cursor = conn.cursor()
            
            metrics = {
                'total_site_energy_kwh': 0,
                'eui_kwh_per_m2': 0,
                'total_cooling_kwh': 0,
                'total_heating_kwh': 0
            }
            
            try:
                # 从TabularData表中查询能耗数据（单位：GJ）
                # 数据结构：Total End Uses行包含所有能源类型的汇总
                # 包括：Electricity, Natural Gas, Coal, District Cooling/Heating等
                # 过滤条件：
                # - 不包含"Source"前缀（因为Source是换算后的，会重复计算）
                # - 不包含"Water"（这是用水量，不是能耗）
                
                # 1. 查询总建筑能耗 - Total End Uses行的所有非Source、非Water能源求和
                cursor.execute("""
                    SELECT COALESCE(SUM(CAST(t.Value AS FLOAT)), 0) as TotalEnergy
                    FROM TabularData t 
                    JOIN Strings sr ON t.RowNameIndex = sr.StringIndex 
                    JOIN Strings sc ON t.ColumnNameIndex = sc.StringIndex 
                    JOIN Strings su ON t.UnitsIndex = su.StringIndex 
                    WHERE sr.Value = 'Total End Uses'
                    AND su.Value = 'GJ'
                    AND sc.Value NOT LIKE 'Source%'
                    AND sc.Value != 'Water'
                """)
                
                total_result = cursor.fetchone()
                if total_result:
                    total_gj = float(total_result[0])
                    metrics['total_site_energy_kwh'] = round(total_gj * 277.778, 2)
                    self.logger.debug(f"总建筑能耗: {total_gj} GJ = {metrics['total_site_energy_kwh']} kWh")
                
                # 2. 查询冷却能耗 - Cooling行的所有非Source、非Water能源求和
                cursor.execute("""
                    SELECT COALESCE(SUM(CAST(t.Value AS FLOAT)), 0) as CoolingEnergy
                    FROM TabularData t 
                    JOIN Strings sr ON t.RowNameIndex = sr.StringIndex 
                    JOIN Strings sc ON t.ColumnNameIndex = sc.StringIndex 
                    JOIN Strings su ON t.UnitsIndex = su.StringIndex 
                    WHERE sr.Value = 'Cooling'
                    AND su.Value = 'GJ'
                    AND sc.Value NOT LIKE 'Source%'
                    AND sc.Value != 'Water'
                """)
                
                cooling_result = cursor.fetchone()
                if cooling_result:
                    cooling_gj = float(cooling_result[0])
                    metrics['total_cooling_kwh'] = round(cooling_gj * 277.778, 2)
                    self.logger.debug(f"冷却能耗: {cooling_gj} GJ = {metrics['total_cooling_kwh']} kWh")
                
                # 3. 查询供暖能耗 - Heating行的所有非Source、非Water能源求和
                cursor.execute("""
                    SELECT COALESCE(SUM(CAST(t.Value AS FLOAT)), 0) as HeatingEnergy
                    FROM TabularData t 
                    JOIN Strings sr ON t.RowNameIndex = sr.StringIndex 
                    JOIN Strings sc ON t.ColumnNameIndex = sc.StringIndex 
                    JOIN Strings su ON t.UnitsIndex = su.StringIndex 
                    WHERE sr.Value = 'Heating'
                    AND su.Value = 'GJ'
                    AND sc.Value NOT LIKE 'Source%'
                    AND sc.Value != 'Water'
                """)
                
                heating_result = cursor.fetchone()
                if heating_result:
                    heating_gj = float(heating_result[0])
                    metrics['total_heating_kwh'] = round(heating_gj * 277.778, 2)
                    self.logger.debug(f"供暖能耗: {heating_gj} GJ = {metrics['total_heating_kwh']} kWh")
                
                # 4. 查询能耗强度 (MJ/m²) 并转换为 kWh/m²
                cursor.execute("""
                    SELECT t.Value 
                    FROM TabularData t 
                    JOIN Strings sr ON t.RowNameIndex = sr.StringIndex 
                    JOIN Strings su ON t.UnitsIndex = su.StringIndex 
                    WHERE sr.Value = 'Total Site Energy'
                    AND su.Value = 'MJ/m2'
                """)
                
                eui_result = cursor.fetchone()
                if eui_result:
                    try:
                        eui_mj_m2 = float(eui_result[0])
                        metrics['eui_kwh_per_m2'] = round(eui_mj_m2 / 3.6, 2)
                        self.logger.debug(f"能耗强度: {eui_mj_m2} MJ/m² = {metrics['eui_kwh_per_m2']} kWh/m²")
                    except Exception as e:
                        self.logger.warning(f"无法从SQL查询能耗强度: {e}")
                
                conn.close()
                
                # 验证数据完整性
                if metrics['total_site_energy_kwh'] > 0:
                    self.logger.info(f"✓ 从SQL数据库成功提取能耗数据")
                    self.logger.info(f"  总建筑能耗: {metrics['total_site_energy_kwh']} kWh")
                    self.logger.info(f"  能耗强度: {metrics['eui_kwh_per_m2']} kWh/m²")
                    self.logger.info(f"  冷却能耗: {metrics['total_cooling_kwh']} kWh")
                    self.logger.info(f"  供暖能耗: {metrics['total_heating_kwh']} kWh")
                    return metrics
                else:
                    self.logger.warning(f"⚠ 无法从SQL数据库提取能耗数据")
                    return None
                    
            except Exception as e:
                self.logger.error(f"✗ SQL查询异常: {e}")
                conn.close()
                return None
        
        except Exception as e:
            self.logger.error(f"✗ 提取数据异常: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_optimization_suggestions(self, metrics, iteration):
        """基于能耗数据调用LLM获取优化建议"""
        self.logger.info(f"\n【LLM优化分析】第{iteration}轮")
        self.logger.info("-" * 80)
        
        if not self.client:
            self.logger.error("OpenAI客户端未初始化")
            return None
        
        user_request = self._build_optimization_request(metrics, iteration)
        
        try:
            IDF.setiddname(self.idd_path)
            self.base_idf = IDF(self.current_idf_path)
        except Exception as e:
            self.logger.error(f"加载当前IDF失败: {e}")
            return None
        
        plan = self.generate_plan_with_knowledge_base(user_request)
        if not plan:
            self.logger.error("LLM未返回有效修改方案")
            return None
        
        if plan.get('reasoning'):
            self.logger.info(f"\n【LLM推理过程】\n{plan['reasoning']}")
        if plan.get('confidence'):
            self.logger.info(f"[置信度] {plan['confidence']}")
        
        # 如果LLM无法找到与5个建议相关的修改方案，返回None以使用默认方案
        if plan.get('clarification_needed') or not plan.get('modifications'):
            self.logger.warning(f"LLM无法推荐有效修改方案，将使用默认优化策略")
            return None
        
        return plan
    
    def apply_optimization(self, plan, iteration):
        """根据LLM计划修改IDF（保持与 main.py 一致的过滤与修改逻辑）"""
        if not plan or 'modifications' not in plan:
            self.logger.warning("无有效的修改计划")
            return None
        
        self.logger.info(f"\n【应用优化】第{iteration}轮优化修改")
        self.logger.info("-" * 80)

        try:
            IDF.setiddname(self.idd_path)
            self.base_idf = IDF(self.current_idf_path)
        except Exception as e:
            self.logger.error(f"加载当前IDF失败: {e}")
            return None
        
        coefficient = 0.85
        
        refined_plan = {
            "clarification_needed": False,
            "question": None,
            "options": [],
            "modifications": []
        }
        
        for mod in plan.get("modifications", []):
            fields = mod.get("fields", {})
            refined_fields = {}
            for field_name, expr in fields.items():
                refined_fields[field_name] = self._apply_coefficient_to_expr(expr, coefficient)
            
            refined_plan["modifications"].append({
                "object_type": mod.get("object_type"),
                "name_filter": mod.get("name_filter"),
                "apply_to_all": mod.get("apply_to_all", True),
                "fields": refined_fields
            })
        
        output_idf = os.path.join(self.optimization_dir, f"iteration_{iteration}_optimized.idf")
        self._execute_modifications(refined_plan, output_idf, self.current_idf_path)
        
        if os.path.exists(output_idf):
            self.logger.info(f"✓ 优化修改完成: {output_idf}")
            self.current_idf_path = output_idf
            return output_idf
        
        self.logger.warning("⚠ 优化修改未生成输出文件")
        return None

    def _build_optimization_request(self, metrics, iteration):
        """构造用于知识库检索与LLM推理的需求描述"""
        if iteration == 1:
            status_note = "初始基准模拟"
        else:
            status_note = f"第{iteration}轮迭代优化"
        
        # 构建优化请求，描述当前状态和目标
        request = (
            f"{status_note}。当前能耗指标：\n"
            f"- 总建筑能耗 {metrics['total_site_energy_kwh']} kWh\n"
            f"- 能耗强度 {metrics['eui_kwh_per_m2']} kWh/m²\n"
            f"- 冷却能耗 {metrics['total_cooling_kwh']} kWh\n"
            f"- 供暖能耗 {metrics['total_heating_kwh']} kWh\n"
            f"\n【核心优化目标】必须同时降低总能耗、供暖能耗、制冷能耗三者，不允许其中任何一项上升。"
        )

        # 反馈上一轮效果，避免重复沿错误方向调整
        if len(self.iteration_history) >= 2:
            prev_metrics = self.iteration_history[-2]['metrics']
            delta_total = metrics['total_site_energy_kwh'] - prev_metrics['total_site_energy_kwh']
            delta_cooling = metrics['total_cooling_kwh'] - prev_metrics['total_cooling_kwh']
            delta_heating = metrics['total_heating_kwh'] - prev_metrics['total_heating_kwh']

            request += (
                f"\n【上一轮效果反馈】\n"
                f"  总能耗变化: {delta_total:+.2f} kWh\n"
                f"  制冷能耗变化: {delta_cooling:+.2f} kWh\n"
                f"  供暖能耗变化: {delta_heating:+.2f} kWh"
            )

            # 详细分析每一项的变化
            problems = []
            if delta_total > 0:
                problems.append("总能耗上升")
            if delta_cooling > 0:
                problems.append("制冷能耗上升")
            if delta_heating > 0:
                problems.append("供暖能耗上升")
            
            if problems:
                request += (
                    f"\n⚠️ 上一轮存在问题：{', '.join(problems)}。\n"
                    f"本轮必须针对性修正：\n"
                )
                if delta_cooling > 0:
                    request += f"  - 制冷能耗上升了{delta_cooling:.2f} kWh，需要调整制冷相关参数（如提高制冷供风温度、增加窗户反射率、降低内部热源等）\n"
                if delta_heating > 0:
                    request += f"  - 供暖能耗上升了{delta_heating:.2f} kWh，需要调整供暖相关参数（如降低供暖供风温度、增加围护结构保温、减少渗透等）\n"
                request += f"  - 避免重复上一轮的修改方向，应反向调整或采用更小步长。"
            else:
                request += "\n✓ 上一轮三项指标均下降，本轮继续沿此方向优化，但注意保持平衡。"
        
        request += f"\n请根据物理原理，推荐可执行的IDF参数修改方案（允许某些字段增大、另一些字段减小）。"
        
        return request

    def generate_plan_with_knowledge_base(self, user_request):
        """使用知识库生成修改计划（与 main.py 逻辑保持一致）"""
        if not self.knowledge_base:
            self.logger.error("知识库未加载，无法生成计划")
            return None
        
        enhanced_context = self.knowledge_base.get_enhanced_context(user_request)
        
        if not enhanced_context['matched_objects']:
            self.logger.warning("知识库未找到匹配对象")
            return None
        
        kb_context = {
            'user_intent': enhanced_context['user_intent_analysis'],
            'candidates': []
        }
        
        for obj in enhanced_context['matched_objects']:
            obj_type = obj['object_type']
            candidate_fields = obj['candidate_fields']

            if obj_type in self.base_idf.idfobjects and len(self.base_idf.idfobjects[obj_type]) > 0:
                sample_obj = self.base_idf.idfobjects[obj_type][0]
                candidate_field_names = [f['field_name'] for f in candidate_fields]
                filtered_names = self._filter_fields_by_method(obj_type, sample_obj, candidate_field_names)
                if filtered_names:
                    candidate_fields = [f for f in candidate_fields if f['field_name'] in filtered_names]

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
你是 EnergyPlus 专家。你的任务是理解用户的能耗优化目标，并基于字段的物理含义进行推理，推荐合适的IDF参数修改方案。

【🎯 核心优化目标】
经过多轮迭代优化，最终要实现：
- 总建筑能耗降低 ≥30%
- 供暖能耗降低 ≥30%  
- 制冷能耗降低 ≥30%
- 所有参数修改必须符合实际工程应用，不得超出合理范围

【🚨 关键警告：特殊关键字不可修改】
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

6. **提高HVAC系统效率**
   - 热回收效率：可从0提升至0.65-0.80（显热和潜热）
   - 供暖供风温度：可适度降低（50°C→42-45°C）
   - 制冷供风温度：可适度提高（13°C→15-17°C）
   - COP/EER：提高冷热源效率20-40%

【关键理解】
这些方向是你优化推荐的首选范围。对于知识库返回的每个候选对象，你需要评估：
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
    必须根据该对象的 Calculation Method 只选择当前生效字段（如 Watts/Area -> Watts_per_Floor_Area）。
- 若你不能确定当前生效字段，必须设置 clarification_needed=true，绝不能猜测字段名。

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
- **严格禁止**：不允许为了降低制冷而牺牲供暖，也不允许为了降低供暖而牺牲制冷。

【输出格式】严格 JSON，必须包含以下内容：
⚠️ **CRITICAL：modifications 数组必须包含至少3-5个修改项，涉及至少2-3个不同的对象类别**
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
                field_name = field['field_name']
                current_value = candidate.get('current_sample_values', {}).get(field_name, "未设置")
                modifiable = self._is_numeric_value(current_value) and (not self._is_special_value(current_value))
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
        
        user_prompt += f"""

【强制多对象修改要求】
**本轮优化必须涉及至少2-3个不同类别的对象修改。**修改方案应覆盖以下多个类别：
- 照明  - 照明功率密度
- 设备 - 设备功率密度  
- 围护结构 - 导热系数
- 窗户 - 太阳热得系数
- 渗透 - 渗透率
- HVAC - 供风温度、热回收效率

modifications数组应包含至少3-5条修改记录，优先做到多类别覆盖。

在"current_sample_values"中显示的是该字段的当前值。需要你检查这些值：
- 如果当前值是数字（如5.5、100、0.8），则该字段可以被修改
- 如果当前值包含特殊关键字（autosize、AutoCalculate、LimitFlowRateAndCapacity等），绝对不能修改！
- 如果当前值是"未设置"，说明该字段为空，不应该修改

【推理步骤】
1. 用户目标与哪些对象类别相关？
2. 用户说的是什么物理目标？
3. 确保从至少2-3个不同类别中各选择至少一个对象
4. 每个对象类别选择1-2个最相关的字段进行修改  
5. 检查所有字段当前值必须是数字（不能是特殊关键字）
6. 应该增加还是减少字段值？必须按物理机理判断
7. 是否所有相关对象都应该修改？优先选择标记“可修改=是”的字段。
8. 验证modifications数组包含至少3-5条、涉及多类别的修改
9. 同时检查对总能耗、供暖、制冷的综合影响，而不是只优化其中一个。

然后生成修改方案。如果至少有一个“可修改=是”的字段，必须输出可执行修改，不要设置clarification_needed=true。
若上一轮总能耗变差，优先给出“反向或减小步长”的方案。
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
            usage = response.usage
            self.logger.info(f"LLM分析完成 (tokens: {usage.total_tokens})")
            return plan
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return None

    def _filter_fields_by_method(self, object_type, sample_obj, candidate_fields):
        """
        根据对象的 Calculation Method 字段过滤候选字段，避免选错计算方式。
        仅在能明确匹配时收缩字段，否则返回原候选列表。
        """
        if not candidate_fields:
            return candidate_fields

        target_field = self._get_method_target_field(object_type, sample_obj)
        if not target_field:
            return candidate_fields

        if target_field in candidate_fields:
            return [target_field]
        return candidate_fields

    def _get_method_field_map(self):
        """返回按对象类型定义的 Calculation Method → 有效字段映射。"""
        return {
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

    def _has_method_mapping(self, object_type):
        return object_type in self._get_method_field_map()

    def _get_method_field_name(self, object_type):
        method_map = self._get_method_field_map()
        if object_type not in method_map:
            return None
        return method_map[object_type][0]

    def _get_method_target_field(self, object_type, sample_obj):
        """根据对象的 Calculation Method 解析当前生效字段名，无法解析时返回 None。"""
        method_field_map = self._get_method_field_map()

        if object_type not in method_field_map:
            return None

        method_field, value_map = method_field_map[object_type]
        method_val = getattr(sample_obj, method_field, None)
        if not method_val:
            return None

        key = str(method_val).upper().replace(" ", "")
        return value_map.get(key)

    def _get_valid_fields_for_method(self, object_type, sample_obj):
        """获取某个对象当前Calculation Method对应的所有有效字段集合。"""
        method_field_map = self._get_method_field_map()

        if object_type not in method_field_map:
            return set()

        method_field, value_map = method_field_map[object_type]
        return set(value_map.values())

    def _is_field_for_method(self, object_type, sample_obj, target_field):
        """检查target_field是否与该对象的当前Calculation Method相匹配。"""
        valid_fields = self._get_valid_fields_for_method(object_type, sample_obj)
        if not valid_fields:
            # 没有method定义则无法验证，返回True（允许修改）
            return True
        return target_field in valid_fields

    def _is_special_value(self, value):
        """
        检测字段值是否为EnergyPlus特殊关键字（不应修改）。
        这些关键字代表内部自动计算或特定行为，不能改成0
        """
        if not isinstance(value, str):
            return False
        
        value_lower = value.lower().strip()
        
        # 空值视为未设置
        if value_lower == '':
            return True
        
        # EnergyPlus特殊关键字列表（精确匹配或前缀匹配）
        # 使用元组 (keyword, match_type) 其中 match_type 为 'exact' 或 'contains'
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
        
        # 精确匹配：值必须完全等于关键字
        for keyword, match_type in special_keywords:
            if match_type == 'exact' and value_lower == keyword:
                return True
            elif match_type == 'contains' and keyword in value_lower:
                return True
        
        return False

    def _is_numeric_value(self, value):
        """判断字段值是否为可计算的数值"""
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
    
    def _apply_coefficient_to_expr(self, expr, coef):
        """将外部系数应用到表达式"""
        if not isinstance(expr, str):
            return expr

        expr_stripped = expr.strip()

        if re.search(r"\bcoefficient\b", expr_stripped, flags=re.IGNORECASE):
            return re.sub(r"\bcoefficient\b", str(coef), expr_stripped, flags=re.IGNORECASE)

        if re.fullmatch(r"\{?existing_value\}?", expr_stripped, flags=re.IGNORECASE):
            return f"existing_value * {coef}"

        return expr_stripped

    def _evaluate_expression(self, value, old_value):
        """计算单个表达式"""
        if not isinstance(value, str):
            try:
                num = float(value)
                if abs(num - round(num)) < 1e-6:
                    return float(round(num))
                return round(num, 6)
            except (ValueError, TypeError):
                return value
        
        value_lower = value.lower()
        has_expr = any(var in value_lower for var in ['old_value', 'original_value', 'existing_value', 'current_value'])
        
        if not has_expr:
            try:
                num = float(value)
                if abs(num - round(num)) < 1e-6:
                    return float(round(num))
                return round(num, 6)
            except (ValueError, TypeError):
                return value
        
        try:
            if old_value is None or old_value == '' or (isinstance(old_value, str) and old_value.strip() == ''):
                old_val_num = 0.0
            else:
                try:
                    old_val_num = float(old_value)
                except (ValueError, TypeError):
                    self.logger.warning(f"无法将原值 '{old_value}' 转换为数值，使用 0")
                    old_val_num = 0.0
            
            expr = value
            expr = re.sub(r'\{?old_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?original_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?existing_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            expr = re.sub(r'\{?current_value\}?', str(old_val_num), expr, flags=re.IGNORECASE)
            
            if any(dangerous in expr.lower() for dangerous in ['import', 'exec', 'eval', '__', 'open', 'os.']):
                self.logger.error(f"表达式包含不安全操作: {expr}")
                return value
            
            result = eval(expr, {"__builtins__": {}}, {})
            try:
                num = float(result)
            except (ValueError, TypeError):
                return result
            if abs(num - round(num)) < 1e-6:
                return float(round(num))
            return round(num, 6)
        except Exception as e:
            self.logger.error(f"计算表达式 '{value}' 失败: {e}")
            return value

    def _do_modification_work(self, plan, output_idf_path, source_idf_path):
        """执行修改并保存，保留原始IDF格式。
        默认策略：同一对象类型下的同字段修改，应用到全部实例，避免漏改。
        """
        target_updates = []
        
        for mod in plan["modifications"]:
            obj_type_input = mod.get("object_type")
            name_filter = mod.get("name_filter")
            apply_to_all = mod.get("apply_to_all", True)
            fields_to_mod = mod.get("fields", {})
            
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
                self.logger.warning(f"跳过: 找不到对象类型 '{obj_type_input}'")
                continue

            if name_filter and apply_to_all:
                self.logger.info(
                    f"  [批量应用] {target_obj_type} 提供了 name_filter='{name_filter}'，"
                    f"但当前策略为同字段批量修改，将应用到该对象类型全部实例。"
                )

            for obj in self.base_idf.idfobjects[target_obj_type]:
                obj_name = getattr(obj, 'Name', '')
                if (not apply_to_all) and name_filter and obj_name != name_filter:
                    continue
                
                for field_input, value_expr in fields_to_mod.items():
                    clean_field = field_input.split('{')[0].strip()
                    valid_attrs = obj.fieldnames
                    target_attr = None
                    method_target = self._get_method_target_field(target_obj_type, obj)

                    # 对有 Method 约束的对象，强制只改当前 Method 对应字段，避免 LLM 误改
                    if self._has_method_mapping(target_obj_type):
                        method_field_name = self._get_method_field_name(target_obj_type)
                        method_value = getattr(obj, method_field_name, None)
                        if not method_target or method_target not in valid_attrs:
                            self.logger.warning(
                                f"  [跳过修改] {target_obj_type} ({obj_name}): "
                                f"Method字段 '{method_field_name}' 当前值 '{method_value}' 无法解析为可修改字段，"
                                f"跳过 LLM 建议字段 '{clean_field}'"
                            )
                            continue

                        target_attr = method_target
                        norm_llm_field = clean_field.lower().replace("_", "").replace(" ", "")
                        norm_target_attr = target_attr.lower().replace("_", "").replace(" ", "")
                        if norm_llm_field != norm_target_attr:
                            self.logger.info(
                                f"  [Method强制字段] {target_obj_type} ({obj_name}): "
                                f"LLM建议字段 '{clean_field}' 已忽略，按 Method 使用 '{target_attr}'"
                            )
                    else:
                        if hasattr(obj, clean_field.replace(" ", "_")):
                            target_attr = clean_field.replace(" ", "_")
                        else:
                            norm_field = clean_field.lower().replace("_", "").replace(" ", "")
                            for attr in valid_attrs:
                                if attr.lower().replace("_", "").replace(" ", "") == norm_field:
                                    target_attr = attr
                                    break

                    if not target_attr:
                        self.logger.warning(
                            f"  [跳过修改] {target_obj_type} ({obj_name}): "
                            f"无法匹配字段 '{clean_field}'，可用字段数={len(valid_attrs)}"
                        )
                        continue
                    
                    old_value = getattr(obj, target_attr, 0)
                    
                    # 检查是否为EnergyPlus特殊关键字，如果是则跳过修改
                    # 这些关键字代表自动计算、特定行为等，不能被改成0或其他数值
                    if self._is_special_value(old_value):
                        self.logger.info(f"  [跳过修改] {target_obj_type} ({obj_name}): {target_attr} 原值为 '{old_value}'（特殊关键字），跳过")
                        continue

                    # 非数值字符串（如 None、DifferentialDryBulb 等枚举）不进行数值修改
                    if isinstance(old_value, str) and not self._is_numeric_value(old_value):
                        self.logger.info(f"  [跳过修改] {target_obj_type} ({obj_name}): {target_attr} 原值为 '{old_value}'（非数值字符串），跳过")
                        continue
                    
                    final_value = self._evaluate_expression(value_expr, old_value)
                    
                    # 计算修改系数
                    try:
                        old_num = float(old_value) if old_value else 0.0
                        new_num = float(final_value) if final_value else 0.0
                        if old_num != 0:
                            coefficient = new_num / old_num
                        else:
                            coefficient = None
                    except (ValueError, TypeError):
                        coefficient = None
                    
                    target_updates.append({
                        "type": target_obj_type,
                        "name": obj_name,
                        "field": target_attr,
                        "old_value": old_value,
                        "value": final_value,
                        "coefficient": coefficient,
                        "expression": value_expr
                    })
                    self.logger.info(f"  [计划修改] {target_obj_type} ({obj_name}): {target_attr} -> {final_value} (原值: {old_value})")

        if not target_updates:
            self.logger.warning("没有产生有效的修改计划，跳过保存")
            return

        # ========== 新增：在执行修改前打印完整的修改摘要 ==========
        self.logger.info("\n" + "="*100)
        self.logger.info("【修改摘要】执行修改前的详细信息")
        self.logger.info("="*100)
        
        # 按对象型号和名称分组统计
        updates_by_obj = {}
        for update in target_updates:
            key = f"{update['type']} ({update['name']})"
            if key not in updates_by_obj:
                updates_by_obj[key] = []
            updates_by_obj[key].append(update)
        
        total_modifications = len(target_updates)
        self.logger.info(f"总修改数: {total_modifications}")
        self.logger.info("")
        
        for obj_key, updates_list in sorted(updates_by_obj.items()):
            self.logger.info(f"▶ {obj_key}")
            for update in updates_list:
                old_val = update.get('old_value', '?')
                new_val = update['value']
                expression = update.get('expression', '')
                coef = update.get('coefficient')
                field = update['field']
                
                # 格式化修改系数信息
                if coef is not None:
                    if coef >= 1.0:
                        coef_str = f"增加 {coef:.2%}" if coef > 1 else "不变"
                    else:
                        coef_str = f"降低 {(1-coef):.2%}"
                    coef_info = f" [{coef_str}]"
                else:
                    coef_info = " [计算系数失败]"
                
                self.logger.info(f"  • 字段: {field}")
                self.logger.info(f"    原值: {old_val} → 新值: {new_val}{coef_info}")
                if expression:
                    self.logger.info(f"    表达式: {expression}")
            self.logger.info("")
        
        self.logger.info("="*100)
        self.logger.info("开始执行IDF文本替换...")
        self.logger.info("="*100 + "\n")
        # ========== END 新增 ==========

        try:
            with open(source_idf_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(source_idf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        new_lines = []
        current_obj_type = None
        current_obj_name = None
        in_object = False
        
        updates_map = {}
        for item in target_updates:
            t = item['type'].upper()
            n = item['name'].upper()
            f = item['field'].upper()
            if t not in updates_map:
                updates_map[t] = {}
            if n not in updates_map[t]:
                updates_map[t][n] = {}
            updates_map[t][n][f] = item['value']

        for line in lines:
            stripped = line.strip()
            
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

            if in_object:
                if ';' in line.split('!')[0]:
                    in_object = False
                
                if "!- Name" in line:
                    val_part = line.split('!')[0].strip()
                    val_part = val_part.replace(',', '').replace(';', '').strip()
                    current_obj_name = val_part.upper()
                
                if current_obj_type in updates_map:
                    if '!' in line:
                        comment_part = line.split('!')[1].strip()
                        if comment_part.startswith('- '):
                            field_comment = comment_part[2:].strip().upper()
                        else:
                            field_comment = comment_part.upper()
                        
                        target_fields = updates_map[current_obj_type].get(current_obj_name, {})
                        
                        matched_val = None
                        for target_field_key, target_val in target_fields.items():
                            f1 = target_field_key.replace(" ", "").replace("_", "")
                            f2 = field_comment.replace(" ", "").replace("_", "")
                            if f1 in f2:
                                matched_val = target_val
                                break
                        
                        if matched_val is not None:
                            idx_bang = line.find('!')
                            if idx_bang == -1:
                                original_content = line
                                comment_full = ''
                            else:
                                original_content = line[:idx_bang]
                                comment_full = line[idx_bang:]

                            m = re.match(r"^(\s*)([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)(\s*)([,;].*)$", original_content.rstrip('\r\n'))
                            if m:
                                leading = m.group(1)
                                spaces_after = m.group(3)
                                rest = m.group(4)
                                new_original_content = f"{leading}{matched_val}{spaces_after}{rest}"
                            else:
                                parts = original_content.rstrip('\r\n').split(',', 1)
                                sep = ','
                                if len(parts) == 1:
                                    parts = original_content.rstrip('\r\n').split(';', 1)
                                    sep = ';'
                                if len(parts) == 2:
                                    leading = parts[0][:len(parts[0]) - len(parts[0].lstrip())]
                                    tail = parts[1]
                                    new_original_content = f"{leading}{matched_val}{sep}{tail}"
                                else:
                                    new_original_content = original_content.rstrip('\r\n')

                            new_line = new_original_content + comment_full
                            line = new_line
                            self.logger.info(f"✓ [文本替换] {current_obj_type}: {field_comment} -> {matched_val}")

            new_lines.append(line)

        with open(output_idf_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        self.logger.info(f"算例已保存至: {output_idf_path}")

    def _execute_modifications(self, plan, output_idf_path, source_idf_path):
        """执行实际的 IDF 修改"""
        return self._do_modification_work(plan, output_idf_path, source_idf_path)
    
    def update_best_metrics(self, metrics, iteration):
        """更新最优指标"""
        if self.best_metrics is None:
            self.best_metrics = metrics
            self.best_iteration = iteration
            self.logger.info(f"\n【最优值】第{iteration}轮为新的最优方案")
        else:
            if metrics['total_site_energy_kwh'] < self.best_metrics['total_site_energy_kwh']:
                energy_saved = self.best_metrics['total_site_energy_kwh'] - metrics['total_site_energy_kwh']
                save_pct = (energy_saved / self.best_metrics['total_site_energy_kwh'] * 100)
                
                self.best_metrics = metrics
                self.best_iteration = iteration
                
                self.logger.info(f"\n【最优值更新】第{iteration}轮为新的最优方案！")
                self.logger.info(f"节能 {energy_saved:.2f} kWh ({save_pct:.1f}%)")
            else:
                energy_diff = metrics['total_site_energy_kwh'] - self.best_metrics['total_site_energy_kwh']
                self.logger.info(f"\n【对比最优值】本轮能耗增加 {energy_diff:.2f} kWh，未超过最优值")

    def _print_iteration_savings(self, metrics, iteration):
        """打印每轮节能明细（终端+日志）"""
        if not self.iteration_history:
            return

        baseline_metrics = self.iteration_history[0]['metrics']

        def _safe_pct(saved_value, base_value):
            if base_value == 0:
                return 0.0
            return saved_value / base_value * 100

        total_saved_vs_baseline = baseline_metrics['total_site_energy_kwh'] - metrics['total_site_energy_kwh']
        eui_saved_vs_baseline = baseline_metrics['eui_kwh_per_m2'] - metrics['eui_kwh_per_m2']
        cooling_saved_vs_baseline = baseline_metrics['total_cooling_kwh'] - metrics['total_cooling_kwh']
        heating_saved_vs_baseline = baseline_metrics['total_heating_kwh'] - metrics['total_heating_kwh']

        self.logger.info("\n【本轮节能明细（相对初始基准）】")
        self.logger.info(
            f"- 总建筑能耗: {total_saved_vs_baseline:+.2f} kWh "
            f"({_safe_pct(total_saved_vs_baseline, baseline_metrics['total_site_energy_kwh']):+.2f}%)"
        )
        self.logger.info(
            f"- 能耗强度(EUI): {eui_saved_vs_baseline:+.2f} kWh/m² "
            f"({_safe_pct(eui_saved_vs_baseline, baseline_metrics['eui_kwh_per_m2']):+.2f}%)"
        )
        self.logger.info(
            f"- 制冷能耗: {cooling_saved_vs_baseline:+.2f} kWh "
            f"({_safe_pct(cooling_saved_vs_baseline, baseline_metrics['total_cooling_kwh']):+.2f}%)"
        )
        self.logger.info(
            f"- 供暖能耗: {heating_saved_vs_baseline:+.2f} kWh "
            f"({_safe_pct(heating_saved_vs_baseline, baseline_metrics['total_heating_kwh']):+.2f}%)"
        )

        if iteration > 1 and len(self.iteration_history) >= 2:
            prev_metrics = self.iteration_history[-2]['metrics']

            total_saved_vs_prev = prev_metrics['total_site_energy_kwh'] - metrics['total_site_energy_kwh']
            eui_saved_vs_prev = prev_metrics['eui_kwh_per_m2'] - metrics['eui_kwh_per_m2']
            cooling_saved_vs_prev = prev_metrics['total_cooling_kwh'] - metrics['total_cooling_kwh']
            heating_saved_vs_prev = prev_metrics['total_heating_kwh'] - metrics['total_heating_kwh']

            self.logger.info("【本轮节能明细（相对上一轮）】")
            self.logger.info(
                f"- 总建筑能耗: {total_saved_vs_prev:+.2f} kWh "
                f"({_safe_pct(total_saved_vs_prev, prev_metrics['total_site_energy_kwh']):+.2f}%)"
            )
            self.logger.info(
                f"- 能耗强度(EUI): {eui_saved_vs_prev:+.2f} kWh/m² "
                f"({_safe_pct(eui_saved_vs_prev, prev_metrics['eui_kwh_per_m2']):+.2f}%)"
            )
            self.logger.info(
                f"- 制冷能耗: {cooling_saved_vs_prev:+.2f} kWh "
                f"({_safe_pct(cooling_saved_vs_prev, prev_metrics['total_cooling_kwh']):+.2f}%)"
            )
            self.logger.info(
                f"- 供暖能耗: {heating_saved_vs_prev:+.2f} kWh "
                f"({_safe_pct(heating_saved_vs_prev, prev_metrics['total_heating_kwh']):+.2f}%)"
            )
    
    def run_optimization_loop(self, max_iterations=5):
        """运行5次迭代优化循环"""
        self.logger.info(f"\n\n{'█'*80}")
        self.logger.info(f"启动{max_iterations}轮迭代优化")
        self.logger.info(f"{'█'*80}\n")
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"\n\n{'═'*80}")
            self.logger.info(f"【第{iteration}轮/{max_iterations}】")
            self.logger.info(f"{'═'*80}\n")
            
            # 第一轮使用原始IDF，后续使用优化后的IDF
            if iteration == 1:
                sim_idf = self.idf_path
                iter_name = "initial_baseline"
            else:
                sim_idf = self.current_idf_path
                iter_name = f"iteration_{iteration}"
            
            # 1. 运行模拟
            sim_dir = self.run_simulation(sim_idf, iter_name)
            if not sim_dir:
                self.logger.error(f"第{iteration}轮模拟失败，中断优化")
                break
            
            # 2. 提取能耗指标
            metrics = self.extract_metrics(sim_dir)
            if not metrics:
                self.logger.error(f"第{iteration}轮无法提取能耗数据，中断优化")
                break
            
            # 保存到历史
            self.iteration_history.append({
                'iteration': iteration,
                'metrics': metrics,
                'idf_path': sim_idf
            })

            # 3.5 打印每轮节能明细
            self._print_iteration_savings(metrics, iteration)
            
            # 3. 更新最优值
            self.update_best_metrics(metrics, iteration)
            
            # 4. 最后一轮不需要优化
            if iteration == max_iterations:
                self.logger.info(f"\n✓ 优化循环完成！")
                break
            
            # 5. LLM分析建议
            plan = self.generate_optimization_suggestions(metrics, iteration)
            if not plan:
                self.logger.warning(f"第{iteration}轮无法获得优化建议，使用默认优化")
                plan = self._get_default_suggestions(iteration)
            
            # 6. 应用优化
            new_idf = self.apply_optimization(plan, iteration)
            if not new_idf:
                self.logger.warning(f"第{iteration}轮优化应用失败，继续下一轮")
                # 继续使用当前IDF
        
        # 优化循环结束
        self._print_final_report()
    
    def _get_default_suggestions(self, iteration):
        """无法获得LLM建议时的默认优化方案"""
        defaults = {
            "clarification_needed": False,
            "reasoning": f"第{iteration}轮使用默认优化方案",
            "confidence": "low",
            "modifications": [
                {
                    "object_type": "Lights",
                    "name_filter": None,
                    "fields": {
                        "Watts_per_Floor_Area": "existing_value * coefficient"
                    }
                }
            ]
        }
        return defaults
    
    def _extract_and_print_optimal_parameters(self):
        """提取和打印最优方案中的所有参数修改"""
        if self.best_iteration == 0 or not self.iteration_history:
            return
        
        self.logger.info(f"\n\n{'═'*80}")
        self.logger.info(f"【最优优化方案 - 参数修改详情】")
        self.logger.info(f"{'═'*80}\n")
        
        baseline_idf_path = self.idf_path
        optimal_idf_path = self.iteration_history[self.best_iteration-1]['idf_path']
        
        if not os.path.exists(optimal_idf_path):
            self.logger.warning(f"最优IDF文件不存在: {optimal_idf_path}")
            return
        
        try:
            # 加载基准和最优IDF进行比较
            baseline_idf = IDF(baseline_idf_path)
            optimal_idf = IDF(optimal_idf_path)
            
            # 收集所有修改
            modifications = []
            modified_objects = set()
            
            # 遍历最优IDF中的所有对象类别
            for obj_type in optimal_idf.idfobjects:
                if obj_type not in baseline_idf.idfobjects:
                    continue
                
                baseline_objs = {getattr(o, 'Name', str(i)): o for i, o in enumerate(baseline_idf.idfobjects[obj_type])}
                optimal_objs = {getattr(o, 'Name', str(i)): o for i, o in enumerate(optimal_idf.idfobjects[obj_type])}
                
                for obj_name in optimal_objs:
                    if obj_name not in baseline_objs:
                        continue
                    
                    baseline_obj = baseline_objs[obj_name]
                    optimal_obj = optimal_objs[obj_name]
                    
                    # 比较每个字段
                    for field in optimal_obj.fieldnames:
                        try:
                            baseline_val = getattr(baseline_obj, field, None)
                            optimal_val = getattr(optimal_obj, field, None)
                            
                            # 字符串/数值比较
                            if baseline_val != optimal_val:
                                # 过滤掉非相关字段（如Name、Type等）
                                if field.lower() not in ['name', 'type']:
                                    try:
                                        # 尝试转换为数值进行对比
                                        baseline_num = float(baseline_val) if baseline_val else 0
                                        optimal_num = float(optimal_val) if optimal_val else 0
                                        
                                        if baseline_num != optimal_num:
                                            change_pct = ((optimal_num - baseline_num) / baseline_num * 100) if baseline_num != 0 else 0
                                            modifications.append({
                                                'object_type': obj_type,
                                                'object_name': obj_name,
                                                'field': field,
                                                'baseline_value': baseline_num,
                                                'optimal_value': optimal_num,
                                                'change_pct': change_pct
                                            })
                                            modified_objects.add(obj_type)
                                    except (ValueError, TypeError):
                                        # 非数值字段，做字符串对比
                                        if str(baseline_val) != str(optimal_val):
                                            modifications.append({
                                                'object_type': obj_type,
                                                'object_name': obj_name,
                                                'field': field,
                                                'baseline_value': baseline_val,
                                                'optimal_value': optimal_val,
                                                'change_pct': 'N/A'
                                            })
                                            modified_objects.add(obj_type)
                        except Exception as e:
                            continue
            
            # 按对象类型分组打印
            if modifications:
                self.logger.info(f"【修改对象类型】({len(modified_objects)}种)")
                self.logger.info(f"{', '.join(sorted(modified_objects))}\n")
                
                self.logger.info(f"【参数修改详情】(共{len(modifications)}项修改)\n")
                
                current_obj_type = None
                for mod in sorted(modifications, key=lambda x: (x['object_type'], x['object_name'])):
                    if mod['object_type'] != current_obj_type:
                        current_obj_type = mod['object_type']
                        self.logger.info(f"\n━━ {current_obj_type} ━━")
                    
                    obj_name = mod['object_name']
                    field = mod['field']
                    baseline = mod['baseline_value']
                    optimal = mod['optimal_value']
                    change_pct = mod['change_pct']
                    
                    if isinstance(change_pct, str):
                        change_str = f"{baseline} → {optimal}"
                        self.logger.info(f"  • {obj_name}")
                        self.logger.info(f"    └─ {field}: {change_str}")
                    else:
                        change_str = f"{baseline:.4f} → {optimal:.4f}"
                        pct_str = f"({change_pct:+.2f}%)" if change_pct != 0 else ""
                        self.logger.info(f"  • {obj_name}")
                        self.logger.info(f"    └─ {field}: {change_str} {pct_str}")
            else:
                self.logger.info("未检测到参数修改（可能原始文件和优化文件相同）")
            
            self.logger.info(f"\n{'═'*80}\n")
            
        except Exception as e:
            self.logger.error(f"解析最优方案修改详情时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _print_final_report(self):
        """打印最终优化报告"""
        self.logger.info(f"\n\n{'█'*80}")
        self.logger.info(f"优化完成总结报告")
        self.logger.info(f"{'█'*80}\n")
        
        # 先打印最优参数修改细节
        self._extract_and_print_optimal_parameters()
        
        if not self.iteration_history:
            self.logger.info("无优化数据")
            return
        
        initial_metrics = self.iteration_history[0]['metrics']
        
        self.logger.info("【各轮能耗对比】\n")
        self.logger.info(f"{'轮次':<8} {'总建筑能耗':<15} {'EUI':<15} {'冷却能耗':<15} {'供暖能耗':<15} {'状态'}")
        self.logger.info("-" * 85)
        
        for item in self.iteration_history:
            iteration = item['iteration']
            m = item['metrics']
            
            if iteration == 1:
                status = "基准"
            elif iteration == self.best_iteration:
                status = "最优 ✓"
            else:
                status = ""
            
            self.logger.info(
                f"{iteration:<8} {m['total_site_energy_kwh']:<15.2f} "
                f"{m['eui_kwh_per_m2']:<15.2f} {m['total_cooling_kwh']:<15.2f} "
                f"{m['total_heating_kwh']:<15.2f} {status}"
            )
        
        # 计算最优值相比初始值的节能效果
        if self.best_metrics:
            total_savings = initial_metrics['total_site_energy_kwh'] - self.best_metrics['total_site_energy_kwh']
            total_savings_pct = (total_savings / initial_metrics['total_site_energy_kwh'] * 100)
            
            eui_savings = initial_metrics['eui_kwh_per_m2'] - self.best_metrics['eui_kwh_per_m2']
            eui_savings_pct = (eui_savings / initial_metrics['eui_kwh_per_m2'] * 100)
            
            cool_savings = initial_metrics['total_cooling_kwh'] - self.best_metrics['total_cooling_kwh']
            cool_savings_pct = (cool_savings / initial_metrics['total_cooling_kwh'] * 100) if initial_metrics['total_cooling_kwh'] > 0 else 0
            
            heat_savings = initial_metrics['total_heating_kwh'] - self.best_metrics['total_heating_kwh']
            heat_savings_pct = (heat_savings / initial_metrics['total_heating_kwh'] * 100) if initial_metrics['total_heating_kwh'] > 0 else 0
            
            self.logger.info(f"\n【优化效果】(第{self.best_iteration}轮最优)\n")
            self.logger.info(f"总建筑能耗节能: {total_savings:.2f} kWh ({total_savings_pct:.1f}%)")
            self.logger.info(f"能耗强度改善: {eui_savings:.2f} kWh/m² ({eui_savings_pct:.1f}%)")
            self.logger.info(f"冷却能耗节能: {cool_savings:.2f} kWh ({cool_savings_pct:.1f}%)")
            self.logger.info(f"供暖能耗节能: {heat_savings:.2f} kWh ({heat_savings_pct:.1f}%)")
            
            self.logger.info(f"\n【最优参数】\n")
            self.logger.info(f"最优方案来自: 第{self.best_iteration}轮优化")
            self.logger.info(f"对应IDF: {self.iteration_history[self.best_iteration-1]['idf_path']}")
        
        self.logger.info(f"\n{'█'*80}\n")


if __name__ == "__main__":
    # 配置文件路径
    IDF_PATH = "in.idf"
    IDD_PATH = "Energy+.idd"
    API_KEY_PATH = "api_key.txt"
    EPW_PATH = "weather.epw"
    
    try:
        # 创建优化器实例
        optimizer = EnergyPlusOptimizer(
            idf_path=IDF_PATH,
            idd_path=IDD_PATH,
            api_key_path=API_KEY_PATH,
            epw_path=EPW_PATH
        )
        
        # 运行5轮迭代优化
        optimizer.run_optimization_loop(max_iterations=5)
        
    except Exception as e:
        print(f"✗ 优化过程异常: {e}")
        import traceback
        traceback.print_exc()
