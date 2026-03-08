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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
from pathlib import Path
from eppy.modeleditor import IDF
from openai import OpenAI
from knowledge_base import EnergyPlusKnowledgeBase


class EnergyPlusOptimizer:
    """EnergyPlus 5轮并行工作流迭代自动优化系统"""
    
    def __init__(self, idf_path, idd_path, api_key_path, epw_path="weather.epw", log_dir="optimization_logs_并行2", num_workflows=2):
        self.idf_path = idf_path
        self.idd_path = idd_path
        self.epw_path = epw_path
        self.log_dir = log_dir
        self.optimization_dir = "optimization_results_并行2"
        # 线程级日志上下文：仅用于汇总日志添加[workflow_x]前缀
        self._log_context = threading.local()
        
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
        
        # 初始化知识库（传入idf_path以分析实际可修改字段）
        try:
            self.knowledge_base = EnergyPlusKnowledgeBase(idd_path, idf_path)
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
        
        # 并行工作流配置
        self.num_workflows = num_workflows
        self.workflows = {}  # 存储各工作流的独立数据
        self.workflows_lock = Lock()  # 保护共享资源的锁
        self.logger.info(f"初始化{num_workflows}条并行工作流")
        
        # 初始化各工作流的数据结构
        for i in range(num_workflows):
            workflow_id = f"workflow_{i+1}"
            self.workflows[workflow_id] = {
                'best_metrics': None,
                'best_iteration': 0,
                'iteration_history': [],
                'current_idf_path': idf_path,
                'field_modification_history': {},
                'last_round_fields': set(),
                'logger': self._setup_workflow_logger(workflow_id)  # 为每个workflow创建独立logger
            }
        
        # 全局最优指标（所有工作流中最优）
        self.best_metrics_global = None
        self.best_workflow_id = None
        
        # 初始化追踪数据（兼容旧版本）
        self.best_metrics = None
        self.best_iteration = 0
        self.iteration_history = []
        self.current_idf_path = idf_path  # 当前工作IDF
        
        # Token使用追踪
        self.total_tokens_used = 0
        self.llm_calls_count = 0
        
        # 当前优化建议文本（用于关键词检查）
        self.current_user_request = ""
        self.current_plan_context = ""
        
        # 字段修改频率统计（格式：{"OBJECT_TYPE.Field_Name": count}）
        # 用于追踪各字段的使用频率，避免过度集中修改少数字段
        self.field_modification_history = {}
        
        # 上一轮修改的字段列表（用于检查轮次间的差异性）
        self.last_round_fields = set()
        
        # 字段修改的关键词限制规则（配置化，非硬编码）
        # 格式：{"对象类型": {"字段名": ["必需关键词1", "必需关键词2"]}}
        # 只有当建议文本中包含至少一个必需关键词时，才允许修改该字段
        self.field_keyword_rules = {
            "ZONEHVAC:IDEALLOADSAIRSYSTEM": {
                # 阈值字段需要出现阈值/上下限/限制等语义即可修改
                "MINIMUM_COOLING_SUPPLY_AIR_TEMPERATURE": ["阈值", "上下限", "上限", "下限", "最大", "最小", "限制", "limit", "threshold"],
                "MAXIMUM_HEATING_SUPPLY_AIR_TEMPERATURE": ["阈值", "上下限", "上限", "下限", "最大", "最小", "限制", "limit", "threshold"],
                # 热回收效率无需关键词限制，可自由修改
            },
            "SIZING:ZONE": {
                # 设计温度参数：出现供风/送风/设计温度语义即可修改
                "ZONE_COOLING_DESIGN_SUPPLY_AIR_TEMPERATURE": ["供冷", "制冷", "供风温度", "送风温度", "设计供风温度", "设计温度", "供暖供冷温度"],
                "ZONE_HEATING_DESIGN_SUPPLY_AIR_TEMPERATURE": ["供暖", "制热", "供风温度", "送风温度", "设计供风温度", "设计温度", "供暖供冷温度"],
            }
        }
    
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
        
        # 仅在汇总日志处理器上加工作流前缀，分工作流日志保持老版格式
        class _WorkflowPrefixFilter(logging.Filter):
            def __init__(self, context):
                super().__init__()
                self._context = context

            def filter(self, record):
                workflow_id = getattr(self._context, 'workflow_id', None)
                record.workflow_prefix = f"[{workflow_id}] " if workflow_id else ""
                return True

        prefix_filter = _WorkflowPrefixFilter(self._log_context)
        fh.addFilter(prefix_filter)
        ch.addFilter(prefix_filter)

        # 格式器（汇总日志会自动加 workflow 前缀）
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(workflow_prefix)s%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _setup_workflow_logger(self, workflow_id):
        """为单个workflow初始化独立的日志系统
        
        参数:
            workflow_id: 工作流ID（如 "workflow_1"）
        
        返回:
            独立的logger实例
        """
        # 生成工作流专属日志文件名
        log_file = os.path.join(
            self.log_dir, 
            f"optimization_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # 创建独立的logger实例（使用workflow_id作为名称避免冲突）
        workflow_logger = logging.getLogger(f"{__name__}.{workflow_id}")
        workflow_logger.handlers = []  # 清除之前的处理器
        workflow_logger.setLevel(logging.DEBUG)
        workflow_logger.propagate = False  # 不向父logger传播，避免重复输出
        
        # 文件处理器 - 输出到workflow专属日志文件
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器 - 不添加，避免在控制台重复输出
        # 控制台输出由主logger统一管理
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        
        workflow_logger.addHandler(fh)
        
        return workflow_logger

    def _attach_workflow_thread_handler(self, workflow_id):
        """按线程分流日志到工作流文件，确保子方法日志完整落盘。"""
        log_file = os.path.join(
            self.log_dir,
            f"optimization_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        current_thread_name = threading.current_thread().name

        class _ThreadFilter(logging.Filter):
            def filter(self, record):
                return record.threadName == current_thread_name

        handler.addFilter(_ThreadFilter())
        self.logger.addHandler(handler)
        return handler

    def _detach_workflow_thread_handler(self, handler):
        """卸载线程分流handler，防止句柄泄漏。"""
        try:
            self.logger.removeHandler(handler)
            handler.close()
        except Exception:
            pass
    
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
    
    def run_simulation(self, idf_path, iteration_name, workflow_id=None):
        """运行EnergyPlus模拟"""
        self.logger.info(f"\n【模拟】{iteration_name}")
        self.logger.info("-" * 80)
        
        try:
            # 运行目录按工作流隔离，避免并行时同名轮次目录冲突
            if workflow_id:
                run_dir = os.path.join(self.optimization_dir, workflow_id, iteration_name)
            else:
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
            # 使用二进制捕获后手动容错解码，避免Windows默认GBK解码失败
            result = subprocess.run(cmd, capture_output=True, text=False, timeout=600)
            
            if result.returncode == 0:
                self.logger.info(f"✓ 模拟完成: {iteration_name}")
                return run_dir
            else:
                self.logger.error(f"✗ 模拟失败: {iteration_name}")
                self.logger.error(f"返回码: {result.returncode}")
                stderr_text = ""
                if result.stderr:
                    stderr_text = result.stderr.decode('utf-8', errors='replace')
                if stderr_text:
                    self.logger.error(f"错误: {stderr_text[:500]}")

                # 追加读取 eplusout.err，提取 Severe/Fatal 关键行，便于快速定位失败原因
                err_file = os.path.join(run_dir, "eplusout.err")
                if os.path.exists(err_file):
                    try:
                        with open(err_file, 'r', encoding='utf-8', errors='replace') as ef:
                            err_lines = ef.readlines()
                        key_lines = [ln.strip() for ln in err_lines if ("** Severe" in ln or "**  Fatal" in ln or "Terminated" in ln)]
                        if key_lines:
                            self.logger.error("EnergyPlus错误摘要（eplusout.err）:")
                            for ln in key_lines[-8:]:
                                self.logger.error(f"  {ln}")
                    except Exception as _e:
                        self.logger.warning(f"读取eplusout.err失败: {_e}")
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
                
                # 4. 查询单位面积总建筑能耗 (MJ/m²) 并转换为 kWh/m²
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
                        self.logger.debug(f"单位面积总建筑能耗: {eui_mj_m2} MJ/m² = {metrics['eui_kwh_per_m2']} kWh/m²")
                    except Exception as e:
                        self.logger.warning(f"无法从SQL查询单位面积总建筑能耗: {e}")
                
                # 在关闭连接前，尝试根据 Total Site EUI 计算建筑面积并把冷/暖能耗转换为 kWh/m²
                building_area_m2 = None
                try:
                    if metrics.get('eui_kwh_per_m2') and metrics.get('eui_kwh_per_m2') > 0 and metrics.get('total_site_energy_kwh') and metrics.get('total_site_energy_kwh') > 0:
                        building_area_m2 = metrics['total_site_energy_kwh'] / metrics['eui_kwh_per_m2']
                        self.logger.debug(f"计算建筑面积: {building_area_m2:.2f} m² (total_kWh / eui_kWh_per_m2)")
                        # 保存为全局属性以便后续轮次回退使用
                        try:
                            self.building_area_m2 = float(building_area_m2)
                        except Exception:
                            self.building_area_m2 = None
                        # 将冷/暖能耗从 kWh 转为 kWh/m²（按建筑总面积）
                        if building_area_m2 and building_area_m2 > 0:
                            old_cooling = metrics.get('total_cooling_kwh', 0)
                            old_heating = metrics.get('total_heating_kwh', 0)
                            metrics['total_cooling_kwh'] = round(old_cooling / building_area_m2, 4)
                            metrics['total_heating_kwh'] = round(old_heating / building_area_m2, 4)
                            self.logger.debug(f"冷却能耗: {old_cooling} kWh -> {metrics['total_cooling_kwh']} kWh/m²")
                            self.logger.debug(f"供暖能耗: {old_heating} kWh -> {metrics['total_heating_kwh']} kWh/m²")
                except Exception as e:
                    self.logger.warning(f"计算建筑面积或转换冷/暖能耗为kWh/m²时出错: {e}")

                # 如果本次未能从EUI计算出建筑面积，但之前已知建筑面积，则使用历史建筑面积进行转换
                if (not building_area_m2 or building_area_m2 is None) and hasattr(self, 'building_area_m2') and self.building_area_m2:
                    try:
                        old_cooling = metrics.get('total_cooling_kwh', 0)
                        old_heating = metrics.get('total_heating_kwh', 0)
                        metrics['total_cooling_kwh'] = round(old_cooling / self.building_area_m2, 4)
                        metrics['total_heating_kwh'] = round(old_heating / self.building_area_m2, 4)
                        self.logger.debug(f"回退使用已知建筑面积 {self.building_area_m2:.2f} m² 进行冷/暖能耗转换: {old_cooling}->{metrics['total_cooling_kwh']} kWh/m²")
                        building_area_m2 = self.building_area_m2
                    except Exception as e:
                        self.logger.warning(f"使用历史建筑面积转换冷/暖能耗时出错: {e}")

                conn.close()

                # 验证数据完整性
                if metrics['total_site_energy_kwh'] > 0:
                    self.logger.info(f"✓ 从SQL数据库成功提取能耗数据")
                    self.logger.info(f"  总建筑能耗: {metrics['total_site_energy_kwh']} kWh")
                    self.logger.info(f"  单位面积总建筑能耗: {metrics['eui_kwh_per_m2']} kWh/m²")
                    # 基于是否成功计算到建筑面积来标注单位
                    if building_area_m2:
                        self.logger.info(f"  冷却能耗: {metrics['total_cooling_kwh']} kWh/m²")
                        self.logger.info(f"  供暖能耗: {metrics['total_heating_kwh']} kWh/m²")
                    else:
                        self.logger.info(f"  冷却能耗: {metrics['total_cooling_kwh']} kWh (未转换为 kWh/m²)")
                        self.logger.info(f"  供暖能耗: {metrics['total_heating_kwh']} kWh (未转换为 kWh/m²)")
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
        # 保存当前建议文本，用于后续关键词检查
        self.current_user_request = user_request
        
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

        # 关键词过滤上下文：同时包含用户请求与LLM建议内容
        self.current_plan_context = self._build_plan_context_text(plan)
        
        coefficient = 0.85
        
        refined_plan = {
            "clarification_needed": False,
            "question": None,
            "options": [],
            "modifications": []
        }
        
        for mod in plan.get("modifications", []):
            obj_type = mod.get("object_type", "")
            fields = mod.get("fields", {})
            refined_fields = {}
            
            # 对每个字段进行关键词检查和应用系数
            for field_name, expr in fields.items():
                # 检查该字段是否允许修改（基于关键词限制）
                if not self._check_modification_allowed(obj_type, field_name):
                    self.logger.info(
                        f"  【关键词过滤】跳过 {obj_type}.{field_name} - "
                        f"建议文本未包含必需关键词，不允许修改此字段"
                    )
                    continue  # 跳过这个字段，不将其添加到refined_fields
                
                # 应用系数到表达式
                refined_fields[field_name] = self._apply_coefficient_to_expr(expr, coefficient)
            
            # 只有当该对象还有待修改的字段时，才添加到refined_plan
            if refined_fields:
                refined_plan["modifications"].append({
                    "object_type": mod.get("object_type"),
                    "name_filter": mod.get("name_filter"),
                    "apply_to_all": mod.get("apply_to_all", True),
                    "fields": refined_fields
                })
            else:
                # 所有字段都被过滤掉了，记录警告
                self.logger.info(
                    f"  【关键词过滤】对象类型 {obj_type} 的所有字段因关键词限制被过滤，不会进行任何修改"
                )
        
        # 生成输出文件名：如果存在 workflow_id，则加入文件名以区分不同工作流
        if hasattr(self, 'current_workflow_id') and self.current_workflow_id:
            output_idf = os.path.join(self.optimization_dir, f"{self.current_workflow_id}_iteration_{iteration}_optimized.idf")
        else:
            output_idf = os.path.join(self.optimization_dir, f"iteration_{iteration}_optimized.idf")
        self._execute_modifications(refined_plan, output_idf, self.current_idf_path)
        
        if os.path.exists(output_idf):
            self.logger.info(f"✓ 优化修改完成: {output_idf}")
            self.current_idf_path = output_idf
            return output_idf
        
        self.logger.warning("⚠ 优化修改未生成输出文件")
        return None

    def _build_optimization_request(self, metrics, iteration):
        """构造用于知识库检索与LLM推理的需求描述
        
        关键改进：基于历史修改频率，动态生成优化方向关键词，
        确保知识库能匹配到多样化的对象（人员、遮阳、新风等）
        """
        if iteration == 1:
            status_note = "初始基准模拟"
        else:
            status_note = f"第{iteration}轮迭代优化"
        
        # 构建优化请求，描述当前状态和目标
        request = (
            f"{status_note}。当前能耗指标：\n"
            f"- 总建筑能耗 {metrics['total_site_energy_kwh']} kWh\n"
            f"- 单位面积总建筑能耗 {metrics['eui_kwh_per_m2']} kWh/m²\n"
            f"- 冷却能耗 {metrics['total_cooling_kwh']} kWh/m²\n"
            f"- 供暖能耗 {metrics['total_heating_kwh']} kWh/m²\n"
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
                f"  制冷能耗变化: {delta_cooling:+.2f} kWh/m²\n"
                f"  供暖能耗变化: {delta_heating:+.2f} kWh/m²"
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
                
                # ========== 关键改进：供暖和制冷的平衡优化 ==========
                # 检查是否制冷和供暖都在上升（最棘手的情况）
                if delta_cooling > 0 and delta_heating > 0:
                    request += f"  ⚠️ 制冷（↑{delta_cooling:.2f} kWh/m²）和供暖（↑{delta_heating:.2f} kWh/m²）都上升，说明改键修改效果不理想。\n"
                    request += f"  本轮应采用['同时降低制冷和供暖'的综合策略，优先选择对两者都有利的措施：\n"
                    request += f"    • 增加围护结构保温性能（降低导热系数）- 减少冬季热损失 & 减少夏季散热失\n"
                    request += f"    • 减少空气渗透（降低渗透率）- 减少冬季冷空气进入 & 减少夏季热空气进入\n"
                    request += f"    • 提高热回收效率 - 冬季回收室外新风热量 & 夏季回收排风冷量\n"
                    request += f"    • 降低内部热源（照明、设备、人员）- 直接减少制冷需求 & 增加供暖需求（但总能耗下降）\n"
                    request += f"  ⚠️ 避免：修改遮阳系数（对制冷有利但对供暖有害）、修改供暖供风温度（影响设计）等相冲突的措施。\n"
                elif delta_cooling > 0:
                    request += f"  - 制冷能耗上升了{delta_cooling:.2f} kWh/m²，需要调整制冷相关参数（如提高制冷供风温度、增加窗户反射率、降低内部热源等）\n"
                    request += f"  - 同时注意不要增加供暖负荷，避免修改会增加冬季需热的参数。\n"
                if delta_heating > 0:
                    request += f"  - 供暖能耗上升了{delta_heating:.2f} kWh/m²，需要调整供暖相关参数（如降低供暖供风温度、增加围护结构保温、减少渗透等）\n"
                    request += f"  - 同时注意不要增加制冷负荷，避免修改会增加夏季散热困难的参数。\n"
                request += f"  - 避免重复上一轮的修改方向，应反向调整或采用更小步长。"
            else:
                request += "\n✓ 上一轮三项指标均下降，本轮继续沿此方向优化，但注意保持平衡。"
        
        # ========== 关键改进：动态添加优化方向关键词 ==========
        # 这确保知识库能匹配到多样化的对象类型
        # 传递供暖和制冷的变化信息，让优化方向对目标更有针对性
        if len(self.iteration_history) >= 2:
            prev_metrics = self.iteration_history[-2]['metrics']
            delta_cooling_for_directions = metrics['total_cooling_kwh'] - prev_metrics['total_cooling_kwh']
            delta_heating_for_directions = metrics['total_heating_kwh'] - prev_metrics['total_heating_kwh']
            optimization_directions = self._get_dynamic_optimization_directions(
                iteration, 
                delta_cooling=delta_cooling_for_directions, 
                delta_heating=delta_heating_for_directions
            )
        else:
            optimization_directions = self._get_dynamic_optimization_directions(iteration)
        
        request += f"\n{optimization_directions}"
        
        request += f"\n请根据物理原理，推荐可执行的IDF参数修改方案（允许某些字段增大、另一些字段减小）。"
        
        return request
    
    def _get_dynamic_optimization_directions(self, iteration, delta_cooling=None, delta_heating=None):
        """基于历史修改频率和能耗变化，动态生成优化方向关键词
        
        关键改进：
        - 当供暖和制冷都需要优化时，重点推荐对两者都有利的措施
        - 避免或降低那些会产生冲突的措施
        
        核心逻辑：
        1. 统计各类别的修改频率
        2. 识别低频类别（被忽视的优化机会）
        3. 当供暖和制冷都上升时，优先选择"对两者都有利"的类别
        4. 每轮侧重3-4个不同的低频类别
        5. 在请求中明确提及关键词，确保知识库能匹配到相关对象
        
        这解决了"为什么总是修改相同字段"和"供暖制冷无法同时优化"的问题
        """
        # 判断是否"制冷和供暖都需要优化"
        has_heating_cooling_conflict = (delta_cooling is not None and delta_heating is not None 
                                        and delta_cooling > 0 and delta_heating > 0)
        
        # 定义核心优化类别及其关键词（用于知识库匹配）
        # 添加"conflict_level"标注：0=对两者都有利，1=对一方有利，2=可能冲突
        optimization_categories = {
            "人员密度": {
                "keywords": ["人员密度", "人员"],
                "description": "降低人员密度可减少内部产热，对制冷和供暖都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "照明功率": {
                "keywords": ["照明功率密度", "照明"],
                "description": "降低照明功率密度可减少内部发热，对制冷和供暖都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "设备功率": {
                "keywords": ["设备功率密度", "设备"],
                "description": "降低设备功率密度可减少内部发热，对制冷和供暖都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "新风量": {
                "keywords": ["新风量", "新风", "室外空气"],
                "description": "优化新风量可在满足健康标准前提下降低HVAC负荷，对制冷和供暖都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "保温性能": {
                "keywords": ["导热系数", "保温", "材料"],
                "description": "降低材料导热系数可增强保温性能，减少冬季热损失和夏季散热困难，对两者都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "渗透率": {
                "keywords": ["渗透率", "空气渗透"],
                "description": "降低渗透率可减少冷热空气交换，显著降低供暖和制冷能耗，对两者都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "HVAC效率": {
                "keywords": ["热回收效率", "供风温度", "HVAC"],
                "description": "提高热回收效率、优化供风温度可提升HVAC系统效率，对两者都有利",
                "conflict_level": 0  # 对两者都有利
            },
            "遮阳系数": {
                "keywords": ["遮阳系数", "SHGC", "太阳热得系数", "窗"],
                "description": "降低窗户遮阳系数(SHGC)可减少太阳辐射进入，但会增加冬季供暖需求",
                "conflict_level": 1 if has_heating_cooling_conflict else 2  # 当两者都冲突时，标记为1；否则完全避免
            }
        }
        
        # 统计各类别的修改频率
        category_usage = {cat: 0 for cat in optimization_categories.keys()}
        
        for field_key, count in self.field_modification_history.items():
            obj_type = field_key.split('.')[0].upper()
            field_name = field_key.split('.')[1].upper() if '.' in field_key else ""
            
            # 根据对象类型和字段名映射到类别
            if "PEOPLE" in obj_type:
                category_usage["人员密度"] += count
            elif "LIGHTS" in obj_type:
                category_usage["照明功率"] += count
            elif "ELECTRICEQUIPMENT" in obj_type or "EQUIPMENT" in obj_type:
                category_usage["设备功率"] += count
            elif "OUTDOORAIR" in obj_type or "VENTILATION" in field_name:
                category_usage["新风量"] += count
            elif "WINDOW" in obj_type or "SHGC" in field_name or "SOLARHEATGAIN" in field_name:
                category_usage["遮阳系数"] += count
            elif "MATERIAL" in obj_type and "CONDUCTIVITY" in field_name:
                category_usage["保温性能"] += count
            elif "INFILTRATION" in obj_type:
                category_usage["渗透率"] += count
            elif "HVAC" in obj_type or "HEATRECOVERY" in field_name or "TEMPERATURE" in field_name:
                category_usage["HVAC效率"] += count
        
        # 按频率和冲突级别排序
        # 策略：当供暖制冷冲突时，优先选择"对两者都有利"的（conflict_level=0）
        if has_heating_cooling_conflict:
            # 分类：对两者都有利的 vs 可能冲突的
            beneficial_categories = {cat: usage for cat, usage in category_usage.items()
                                    if optimization_categories[cat]['conflict_level'] == 0}
            conflict_categories = {cat: usage for cat, usage in category_usage.items()
                                  if optimization_categories[cat]['conflict_level'] >= 1}
            
            # 优先排序"对两者都有利"的类别（按使用频率）
            sorted_beneficial = sorted(beneficial_categories.items(), 
                                      key=lambda x: (x[1], x[0]))  # 按frequency升序，x[1]是usage
            sorted_conflict = sorted(conflict_categories.items(), 
                                    key=lambda x: (x[1], x[0]))  # 按frequency升序，x[1]是usage
            
            # 组合：先取"对两者都有利"的，再补充冲突类别
            sorted_categories = sorted_beneficial + sorted_conflict
        else:
            # 正常情况：按频率排序（升序，低频在前）
            sorted_categories = sorted(category_usage.items(), key=lambda x: (x[1], x[0]))
        
        # 选择3-4个低频类别（用于本轮优化）
        # 使用迭代次数作为偏移，确保不同轮侧重不同类别
        num_focus = min(4, len(sorted_categories))
        offset = (iteration - 1) % len([c for c in sorted_categories 
                                        if optimization_categories[c[0]]['conflict_level'] < 2])  # 只在有效类别中轮转
        
        # 循环选择，确保覆盖所有类别
        focus_categories = []
        category_idx = 0
        for i in range(num_focus):
            while category_idx < len(sorted_categories):
                cat_name = sorted_categories[category_idx][0]
                category_idx += 1
                # 当存在冲突时，跳过"可能冲突"的类别（conflict_level=2）
                if has_heating_cooling_conflict and optimization_categories[cat_name]['conflict_level'] >= 2:
                    continue
                focus_categories.append(cat_name)
                break
        
        # 构造优化方向文本（包含关键词，确保知识库匹配）
        if has_heating_cooling_conflict:
            directions_text = """【🚨 关键警告 - 供暖和制冷都需要优化】
当前状态：供暖能耗和制冷能耗都需要降低。
强制要求：
1. 本轮修改必须同时降低供暖和制冷能耗，绝对不允许只优化一个而牺牲另一个
2. 只选择"对供暖和制冷都有利"的优化方向（conflict_level=0的措施）
3. 完全避免可能导致冷热失衡的措施（如单独调整遮阳系数）

推荐优化方向（按优先级排序）：
"""
        else:
            directions_text = "【建议优化方向】本轮应重点考虑以下优化方向：\n"
        
        for cat_name in focus_categories:
            cat_info = optimization_categories[cat_name]
            usage_count = category_usage[cat_name]
            keywords_str = "、".join(cat_info["keywords"])
            
            if usage_count == 0:
                priority = "【高优先级-从未优化】"
            elif usage_count <= 2:
                priority = "【推荐优化】"
            else:
                priority = ""
            
            directions_text += f"  {priority} {cat_name}（{keywords_str}）：{cat_info['description']}\n"
        
        # 当供暖和制冷都需要优化时，额外强调
        if has_heating_cooling_conflict:
            directions_text += """
⚠️ 再次强调：
- 本轮所有修改必须确保供暖和制冷能耗都降低
- 如果一个参数可能对供暖有利但对制冷不利（或反之），不要修改它
- 优先选择人员密度、照明功率、设备功率、新风量、渗透率、保温性能、HVAC效率等对两者都有利的措施
- 新风的两个参数（per_person和per_area）必须同时降低，绝对不能一增一减
"""
        
        # 额外提示：避免过度集中
        most_used_cat = sorted_categories[-1]
        if most_used_cat[1] >= 3:
            directions_text += f"\n⚠️ 注意：'{most_used_cat[0]}'已被修改{most_used_cat[1]}次，本轮应尽量避免"
        
        return directions_text

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

【🚨 强制性约束 - 每次优化必须遵守】
1. **每轮修改必须同时降低供暖和制冷能耗**
   - 绝对不允许只优化一个而牺牲另一个
   - 优先选择"对供暖和制冷都有利"的措施
   - 避免选择会导致冷热失衡的措施

2. **相关参数必须同向修改**
   - 新风的 Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area 必须同时降低或同时增加
   - 绝对禁止：一个降低另一个增加（会导致总新风量反而增加，制冷能耗大幅上升）
   - 推荐：如果要降低新风能耗，两个参数都降低 10-20%

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
✅ **优先推荐修改 Sizing:Zone 的设计供风温度**（用自然语言描述为"调整系统设计温度参数"）
✅ **在 reasoning 中强调"调整系统设计温度参数"的节能效果**
⚠️ **只有当用户明确提到"阈值"、"上下限"、"最大最小"时，才考虑运行阈值修改**

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
│ 设计供风温度    │ 空调系统初始设计参数     │ Sizing:Zone │ 影响系统设计  │
│ (Sizing:Zone)   │ 例：14°C制冷，50°C供暖  │ (设计参数)  │ 和设备选型    │
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
- ✅ **提高制冷阈值**（16→17°C）：供冷时的下限更高→供风更温和→舒适度OK但能耗低
- ✅ **降低温控设定点**（22→20°C）：空调需要维持的目标温度更低→负荷减少→能耗降低

10. **优化新风量（关键：两个新风参数必须同向修改）**
   - 人均新风量（Outdoor_Air_Flow_per_Person）：在满足健康标准前提下可适度优化10-20%
   - 单位面积新风量（Outdoor_Air_Flow_per_Zone_Floor_Area）：在满足健康标准前提下可适度优化10-20%
   - 办公室标准≥0.008 m³/s/人（GB/T18883），典型值0.008-0.010
   - 过高的新风量会显著增加供暖/制冷负荷
   
   **物理原理**：实际新风量 = max(per_person × 人数, per_area × 建筑面积)
   - ❌ 若只降低per_area而增加per_person，或反向修改，会导致总新风量反而增加，反优化！
   - ✅ 必须同时降低or同时增加这两个参数，不能反向修改

11. **优化人员密度**
   - 人员密度：可降低20-40%（灵活办公、共享工位）
   - 典型值：办公室0.05-0.08 人/m²，共享工位0.03-0.05 人/m²
   - 影响：人员产热负荷降低，夏季制冷能耗减少

12. **优化围护结构组合**
    - 在Material基础上，可整体优化Construction层级
    - 调整各层厚度、顺序、材料组合

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
  * **新风参数组必须同向修改**：
    - Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area 必须同时降低或同时增加
    - 绝对禁止一个降低另一个增加（会导致总新风量反而增加，制冷能耗大幅上升）
    - 推荐：如果修改新风，两个参数都降低10-20%
- **严格禁止**：不允许为了降低制冷而牺牲供暖，也不允许为了降低供暖而牺牲制冷。

【输出格式】严格 JSON，必须包含以下内容：
⚠️ **CRITICAL：modifications 数组必须覆盖至少5个不同优化方面（对象类别）**
⚠️ **建议给出5-8个修改项，且每个方面至少1条有效字段修改**
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
- 严禁修改 `Sizing:Zone` 中的 `...Input_Method`、`...Difference` 等语义不同字段。

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
        
        # 添加字段使用频率摘要
        field_usage_summary = self._get_field_usage_summary()
        user_prompt += f"\n\n{field_usage_summary}\n"
        
        user_prompt += f"""

【强制多对象修改要求】
**本轮优化必须涉及至少5个不同类别的对象修改。**修改方案应覆盖以下多个类别：
- 照明  - 照明功率密度
- 设备 - 设备功率密度  
- 围护结构 - 导热系数
- 窗户 - 太阳热得系数
- 渗透 - 渗透率
- HVAC - 供风温度、热回收效率
- 人员 - 人员密度
- 新风 - 新风量

modifications数组应包含至少5-8条修改记录，并覆盖至少5个对象类别。

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
   - 如果知识库返回Sizing:Zone → 考虑"供暖供冷温度优化"
   - 如果知识库返回ZoneHVAC:IdealLoadsAirSystem → 考虑"热回收效率提升"
   - 如果知识库返回Material → 考虑"保温性能改善"
   - 如果知识库返回ZoneInfiltration → 考虑"减少空气渗透"
   - **不要只依赖知识库返回的对象，要主动思考相关优化方向**

3. **从至少5个不同类别中选择优化方向（强制要求）**
   - 不允许只优化一个类别（如只改Sizing:Zone）
   - 必须涉及多个维度（HVAC + 围护 + 内热源等）
    - 温度相关场景默认优先考虑“设计供风温度”（Sizing:Zone）
    - 只有当文本出现“阈值/上下限/限制/最大/最小”等语义时，才启用运行阈值字段（IdealLoads）

4. **为每个优化方向确定具体修改字段**
   - 参考"建议→字段修改的明确映射"部分
   - 确保建议与modifications中的字段修改相对应

5. **验证modifications数组的完整性**
    - 至少5-8条修改记录
    - 至少5个不同的对象类别
   - 包含reasoning中提到的所有优化方向

然后生成修改方案。如果至少有一个"可修改=是"的字段，必须输出可执行修改，不要设置clarification_needed=true。
若上一轮总能耗变差，优先给出"反向或减小步长"的方案。
"""
        
        try:
            plan = self._request_plan_from_llm(system_prompt, user_prompt)
            if not plan:
                return None

            min_categories = 5
            category_count = self._count_plan_categories(plan)
            if category_count < min_categories:
                self.logger.warning(
                    f"LLM方案仅覆盖{category_count}个优化方面，低于要求的{min_categories}个，触发一次纠偏重生成"
                )
                repair_prompt = user_prompt + f"""

【纠偏重生成（必须遵守）】
你上一版仅覆盖{category_count}个优化方面，未达到要求。
请重新生成JSON，并满足：
1) modifications 至少覆盖5个不同对象类别（优化方面）
2) 优先覆盖：HVAC设计温度、热回收、照明、设备、渗透、新风、人员、围护中的至少5类
3) 若提到“阈值/上下限/限制/最大/最小”，可修改IdealLoads阈值字段；否则优先修改Sizing:Zone设计温度字段
4) 不得修改特殊关键字字段（autosize/autocalculate等）
"""
                repaired_plan = self._request_plan_from_llm(system_prompt, repair_prompt)
                if repaired_plan:
                    repaired_count = self._count_plan_categories(repaired_plan)
                    if repaired_count >= category_count:
                        plan = repaired_plan
                        category_count = repaired_count

            if category_count < min_categories:
                self.logger.warning(
                    f"最终LLM方案覆盖{category_count}个优化方面，仍低于{min_categories}个；将继续执行当前最佳方案"
                )

            temp_issue = self._get_temperature_mapping_issue(plan, user_request)
            if temp_issue:
                self.logger.warning(f"温度字段映射检查未通过：{temp_issue}，触发一次纠偏重生成")
                temp_repair_prompt = user_prompt + f"""

【温度映射纠偏（必须遵守）】
{temp_issue}

规则：
1) 如果文本没有“阈值/上下限/限制/最大/最小”等语义，禁止修改 IdealLoads 的
   Minimum_Cooling_Supply_Air_Temperature / Maximum_Heating_Supply_Air_Temperature。
2) 此时应优先修改 Sizing:Zone 的
   Zone_Cooling_Design_Supply_Air_Temperature / Zone_Heating_Design_Supply_Air_Temperature。
3) 如果文本出现阈值语义，允许修改 IdealLoads 阈值字段；并鼓励同时保留设计温度优化。
请重新生成严格JSON。
"""
                temp_repaired_plan = self._request_plan_from_llm(system_prompt, temp_repair_prompt)
                if temp_repaired_plan:
                    repaired_issue = self._get_temperature_mapping_issue(temp_repaired_plan, user_request)
                    if not repaired_issue:
                        plan = temp_repaired_plan
                    else:
                        self.logger.warning(f"温度字段映射纠偏后仍未完全满足：{repaired_issue}，将继续执行当前最佳方案")
            
            # ========== 字段多样性检查 ==========
            is_diverse, diversity_issue = self._check_field_diversity(plan)
            if not is_diverse and diversity_issue:
                self.logger.warning(f"字段多样性检查未通过：{diversity_issue}，触发一次纠偏重生成")
                diversity_repair_prompt = user_prompt + f"""

【字段多样性纠偏（必须遵守）】
{diversity_issue}

要求：
1) 避免过度集中于少数高频字段，应从多个类别中选择字段进行修改
2) 优先选择低频字段，如：人员密度、新风量、遮阳系数、设备功率密度等
3) 至少覆盖3个核心优化类别（人员、照明、设备、新风、遮阳、保温、渗透、HVAC等）
4) 参考【历史字段修改频率统计】，优先使用低频字段，避免高频字段
请重新生成包含更多样化字段的JSON方案。
"""
                diversity_repaired_plan = self._request_plan_from_llm(system_prompt, diversity_repair_prompt)
                if diversity_repaired_plan:
                    repaired_is_diverse, _ = self._check_field_diversity(diversity_repaired_plan)
                    if repaired_is_diverse:
                        plan = diversity_repaired_plan
                        self.logger.info("✓ 多样性纠偏成功，已采用新方案")
                    else:
                        # 多样性纠偏LLM仍未完全满足时，强制执行系统自动多样性重排
                        self.logger.warning(f"多样性纠偏后仍未完全满足，启动强制多样性重排补救")
                        diversity_forced_plan = self._force_plan_diversity(plan, kb_context)
                        if diversity_forced_plan:
                            plan = diversity_forced_plan
                            self.logger.info("✓ 强制多样性重排已完成，已采用重排方案")

            return plan
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return None

    def _force_plan_diversity(self, origin_plan, kb_context):
        """强制执行多样性重排：当LLM纠偏仍失败时的兜底方案。
        
        策略：
        1. 收集原方案中的高频字段，标记为已使用
        2. 从知识库的候选对象中额外补充低频字段
        3. 确保总方案覆盖>=4个不同优化类别
        4. 优先使用物理合理的表达式（避免反优化）
        """
        if not isinstance(origin_plan, dict):
            return None
        
        try:
            # 收集原方案高频字段
            used_fields = set()
            obj_field_count = {}  # 按对象类型统计已有字段数
            
            for mod in origin_plan.get('modifications', []):
                obj_type = str(mod.get('object_type', '')).upper().strip()
                for field_name in mod.get('fields', {}).keys():
                    used_fields.add(f"{obj_type}.{str(field_name).upper().strip()}")
                    obj_field_count[obj_type] = obj_field_count.get(obj_type, 0) + 1
            
            # 收集原方案涉及的对象类别
            origin_categories = set(obj_field_count.keys())
            
            # 如果已有方案覆盖>=4个对象类别，说明多样性已足够
            if len(origin_categories) >= 4:
                self.logger.info(f"原方案已覆盖{len(origin_categories)}个对象类别，多样性充分")
                return origin_plan
            
            forced_plan = {
                "reasoning": origin_plan.get("reasoning", "") + "\n[系统强制多样性重排] 补充低频字段以提升多样性",
                "confidence": origin_plan.get("confidence", "medium"),
                "modifications": list(origin_plan.get("modifications", []))
            }
            
            # 从知识库候选中补充低频字段
            # 优先选择：人员、照明、设备、新风、遮阳、保温、渗透等
            priority_keywords = {
                "PEOPLE": ["OCCUPANCY", "PEOPLE"],
                "LIGHTS": ["WATTS", "POWER"],
                "ELECTRICEQUIPMENT": ["WATTS", "POWER"],
                "INFILTRATION": ["FLOW"],
                "MATERIAL": ["CONDUCTIVITY"],
                "CONSTRUCTION": ["NAME"]
            }
            
            # 尝试补充新的低频参数
            candidates_by_type = {}
            if isinstance(kb_context, dict) and 'candidates' in kb_context:
                for cand in kb_context['candidates']:
                    obj_type = cand.get('object_type', '')
                    if obj_type not in candidates_by_type:
                        candidates_by_type[obj_type] = []
                    candidates_by_type[obj_type].append(cand)
            
            added_count = 0
            for target_type in priority_keywords.keys():
                if target_type in origin_categories:
                    continue  # 跳过已有的对象类型
                
                # 查找匹配的候选
                for cand in candidates_by_type.get(target_type, []):
                    for field_info in cand.get('field_semantics', []):
                        field_name = field_info.get('field_name', '')
                        # 检查是否为特殊关键字
                        current_val = cand.get('current_sample_values', {}).get(field_name, '')
                        if self._is_special_value(current_val):
                            continue
                        
                        # 避免已经修改过的字段
                        if f"{target_type}.{str(field_name).upper().strip()}" in used_fields:
                            continue
                        
                        # 构造保守表达式
                        expr = self._build_balanced_expression_for_field(target_type, field_name)
                        
                        # 补充到修改列表
                        forced_plan['modifications'].append({
                            "object_type": target_type,
                            "name_filter": None,
                            "apply_to_all": True,
                            "reasoning": f"补充{target_type}的{field_name}优化以提升多样性",
                            "fields": {field_name: expr}
                        })
                        added_count += 1
                        if added_count >= 2:  # 最多补充2种对象类型
                            break
                
                if added_count >= 2:
                    break
            
            self.logger.info(f"强制多样性重排：补充了{added_count}种新的对象类型及参数修改")
            return forced_plan
        
        except Exception as e:
            self.logger.error(f"强制多样性重排失败: {e}")
            return None

    def _build_balanced_expression_for_field(self, object_type, field_name):
        """为指定字段构造保守平衡的表达式（用于多样性补充）。
        
        物理原理：
        - 降低功率密度字段：* 0.97（降低3%，保守）
        - 提高效率字段：* 1.02（提高2%，保守）
        - 温度字段：±1°C
        """
        obj_upper = str(object_type).upper()
        field_upper = str(field_name).upper()
        
        # 应该提高的字段（效率相关）
        increase_patterns = ["EFFECTIVENESS", "EFFICIENCY", "COP", "EER", "HEATRECOVERY"]
        for pattern in increase_patterns:
            if pattern in field_upper:
                return "existing_value * 1.02"  # 保守增长2%
        
        # 应该降低的字段（功率、渗透、导热相关）
        decrease_patterns = ["WATTS", "POWER", "DENSITY", "INFILTRATION", "CONDUCTIVITY", "FLOW", "PEOPLE", "SOLAR", "SHGC"]
        for pattern in decrease_patterns:
            if pattern in field_upper:
                return "existing_value * 0.97"  # 保守降低3%
        
        # 温度字段特殊处理
        if "TEMPERATURE" in field_upper:
            if "COOLING" in field_upper or "COOL" in field_upper:
                return "existing_value + 1.0"  # 提高1°C（降低冷机负荷）
            elif "HEATING" in field_upper or "HEAT" in field_upper:
                return "existing_value - 1.0"  # 降低1°C（降低热负荷）
        
        # 默认保守降低
        return "existing_value * 0.96"

    def _request_plan_from_llm(self, system_prompt, user_prompt):
        """统一的LLM请求入口，负责调用与token日志和详细token统计。"""
        self.llm_calls_count += 1
        self.logger.info(f"\n🤖 【调用LLM】第{self.llm_calls_count}次调用 - 模型: gpt-5.2")

        response = self.client.chat.completions.create(
            model="gpt-5.2",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        plan = json.loads(response.choices[0].message.content)
        usage = response.usage

        # 详细token统计
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        cached_input_tokens = getattr(usage, 'cached_prompt_tokens', 0) if hasattr(usage, 'cached_prompt_tokens') else 0
        total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)

        # 分别累计
        if not hasattr(self, 'total_input_tokens'): self.total_input_tokens = 0
        if not hasattr(self, 'total_output_tokens'): self.total_output_tokens = 0
        if not hasattr(self, 'total_cached_input_tokens'): self.total_cached_input_tokens = 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_input_tokens += cached_input_tokens
        self.total_tokens_used = getattr(self, 'total_tokens_used', 0) + total_tokens

        self.logger.info("✓ LLM分析完成")
        self.logger.info(f"  - Input Tokens: {input_tokens:,}")
        self.logger.info(f"  - Output Tokens: {output_tokens:,}")
        self.logger.info(f"  - Cached Input Tokens: {cached_input_tokens:,}")
        self.logger.info(f"  - 本次总计: {total_tokens:,} tokens")
        self.logger.info(f"  - 累计 Input: {self.total_input_tokens:,} tokens")
        self.logger.info(f"  - 累计 Output: {self.total_output_tokens:,} tokens")
        self.logger.info(f"  - 累计 Cached Input: {self.total_cached_input_tokens:,} tokens")
        self.logger.info(f"  - 累计总计: {self.total_tokens_used:,} tokens")
        return plan

    def _count_plan_categories(self, plan):
        """统计计划中涉及的对象类别数量。"""
        if not isinstance(plan, dict):
            return 0

        categories = set()
        for mod in plan.get("modifications", []):
            object_type = str(mod.get("object_type", "")).strip()
            fields = mod.get("fields", {})
            if not object_type or not isinstance(fields, dict) or not fields:
                continue
            categories.add(object_type.upper())

        return len(categories)

    def _contains_threshold_intent(self, text):
        """判断文本是否包含阈值/上下限语义。"""
        text_lower = str(text or "").lower()
        threshold_keywords = [
            "阈值", "上下限", "上限", "下限", "限制", "最大", "最小",
            "threshold", "limit", "upper", "lower"
        ]
        return any(keyword in text_lower for keyword in threshold_keywords)
    
    def _print_modification_statistics(self, target_updates):
        """打印本轮修改的统计信息：修改了多少个object，每个object修改了多少个field"""
        if not target_updates:
            self.logger.info("\n【本轮修改统计】无修改")
            return
        
        # 按对象类型和名称分组
        objects_modified = {}
        for update in target_updates:
            obj_key = f"{update['type']} ({update['name']})"
            if obj_key not in objects_modified:
                objects_modified[obj_key] = []
            objects_modified[obj_key].append(update['field'])
        
        total_objects = len(objects_modified)
        total_fields = len(target_updates)
        
        self.logger.info("\n" + "="*100)
        self.logger.info("【本轮修改统计】")
        self.logger.info("="*100)
        self.logger.info(f"修改对象总数: {total_objects}")
        self.logger.info(f"修改字段总数: {total_fields}")
        self.logger.info("")
        
        for obj_key, fields in sorted(objects_modified.items()):
            self.logger.info(f"  ▶ {obj_key}: {len(fields)} 个字段")
            for field in fields:
                self.logger.info(f"      • {field}")
        
        self.logger.info("="*100 + "\n")
    
    def _update_field_modification_history(self, target_updates):
        """更新字段修改频率历史记录，并保存本轮字段列表供下轮差异性检查"""
        # 保存本轮修改的字段（用于下一轮的差异性检查）
        current_round_fields = set()
        
        for update in target_updates:
            field_key = f"{update['type']}.{update['field']}"
            
            # 更新频率统计
            if field_key not in self.field_modification_history:
                self.field_modification_history[field_key] = 0
            self.field_modification_history[field_key] += 1
            
            # 记录本轮字段
            current_round_fields.add(field_key.upper())
        
        # 更新上一轮字段列表（供下一轮对比）
        self.last_round_fields = current_round_fields
    
    def _get_field_usage_summary(self, max_high_freq=10, max_low_freq=10):
        """生成字段使用频率摘要，用于LLM prompt"""
        if not self.field_modification_history:
            return "暂无历史修改记录，这是首轮优化。"
        
        # 按频率排序
        sorted_fields = sorted(
            self.field_modification_history.items(),
            key=lambda x: x[1],
            reverse=True
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
        
        # 统计覆盖的对象类型
        object_types_count = {}
        for field_key in self.field_modification_history.keys():
            obj_type = field_key.split('.')[0]
            object_types_count[obj_type] = object_types_count.get(obj_type, 0) + 1
        
        summary += f"\n已修改的对象类型数: {len(object_types_count)}\n"
        summary += "对象类型覆盖情况：\n"
        for obj_type, count in sorted(object_types_count.items(), key=lambda x: x[1], reverse=True):
            summary += f"  - {obj_type}: {count} 个字段被修改过\n"
        
        return summary
    
    def _check_field_diversity(self, plan):
        """检查优化方案的字段多样性
        
        改进：不仅检查类别覆盖度，还检查与上一轮的差异性
        用户要求：不同轮的修改建议要有50%的不一样
        
        返回: (is_diverse, issue_description)
        """
        if not isinstance(plan, dict) or 'modifications' not in plan:
            return True, None
        
        modifications = plan.get('modifications', [])
        if not modifications:
            return True, None
        
        # 提取本轮涉及的字段
        current_fields = set()
        object_types = set()
        for mod in modifications:
            obj_type = str(mod.get('object_type', '')).upper()
            object_types.add(obj_type)
            fields = mod.get('fields', {})
            for field_name in fields.keys():
                field_key = f"{obj_type}.{str(field_name).upper()}"
                current_fields.add(field_key)
        
        # ========== 新增检查：与上一轮的差异性（50%不同要求） ==========
        if self.last_round_fields and len(current_fields) > 0:
            overlap_fields = current_fields & self.last_round_fields
            overlap_ratio = len(overlap_fields) / len(current_fields)
            
            # 如果重复率超过50%，则多样性不足
            if overlap_ratio > 0.5:
                return False, (
                    f"本轮方案与上一轮重复率过高（{overlap_ratio:.1%}），"
                    f"{len(overlap_fields)}/{len(current_fields)}个字段与上一轮相同。"
                    f"要求不同轮次至少50%字段不同，请选择新的优化方向。"
                )
        
        # 检查1：是否过度集中在高频字段
        if self.field_modification_history:
            sorted_fields = sorted(
                self.field_modification_history.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_5_high_freq = {field.upper() for field, _ in sorted_fields[:5]}
            
            overlap_count = 0
            for field in current_fields:
                if field.upper() in top_5_high_freq:
                    overlap_count += 1
            
            # 如果本轮80%以上的字段都是高频字段，认为多样性不足
            if len(current_fields) > 0 and overlap_count / len(current_fields) >= 0.8:
                return False, f"本轮方案过度集中在高频字段（{overlap_count}/{len(current_fields)}个字段为高频字段）"
        
        # 检查2：是否缺少核心优化类别
        # 定义核心字段关键词（大类）
        core_categories = {
            "人员密度": ["PEOPLE", "PERSON", "DENSITY"],
            "照明功率": ["LIGHTS", "LIGHTING", "POWER"],
            "设备功率": ["ELECTRICEQUIPMENT", "EQUIPMENT"],
            "新风量": ["OUTDOORAIR", "VENTILATION"],
            "遮阳系数": ["SHGC", "SOLARHEATGAINCOEFFICIENT", "WINDOW"],
            "保温性能": ["MATERIAL", "CONDUCTIVITY", "THERMAL"],
            "渗透率": ["INFILTRATION", "AIRCHANGES"],
            "HVAC效率": ["HEATRECOVERY", "EFFICIENCY", "COP"],
        }
        
        # 统计已覆盖的核心类别
        covered_categories = set()
        for field in current_fields:
            field_upper = field.upper()
            for category, keywords in core_categories.items():
                if any(keyword in field_upper for keyword in keywords):
                    covered_categories.add(category)
                    break
        
        # 如果少于3个核心类别，提示多样性不足
        if len(covered_categories) < 3:
            missing_categories = set(core_categories.keys()) - covered_categories
            return False, f"本轮方案仅覆盖 {len(covered_categories)} 个核心优化类别（建议至少3个）。缺失类别：{', '.join(list(missing_categories)[:5])}"
        
        return True, None

    def _get_temperature_mapping_issue(self, plan, user_request):
        """校验温度建议是否符合“设计温度优先、阈值按语义启用”的规则。"""
        if not isinstance(plan, dict):
            return "计划格式无效"

        reasoning = str(plan.get("reasoning", ""))
        text_for_intent = f"{user_request}\n{reasoning}"
        has_threshold_intent = self._contains_threshold_intent(text_for_intent)

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

    def _filter_conflicting_parameter_modifications(self, target_updates):
        """在修改应用前，强制过滤相关参数反向修改的情况。
        
        检查规则：
        1. 新风参数组：Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area
           - 如果两者都被修改，检查是否同向（都增或都减）
           - 如果反向，过滤掉会导致能耗增加的那个修改
        2. 其他相关参数组可以在此扩展
        
        返回：过滤后的 target_updates 列表
        """
        if not target_updates:
            return target_updates
        
        # 收集新风相关参数的修改
        outdoor_air_mods = {}
        other_mods = []
        
        for update in target_updates:
            obj_type = str(update['type']).upper()
            field_name = str(update['field']).upper()
            
            # 检查是否为新风参数
            is_outdoor_air_obj = ('OUTDOORAIR' in obj_type or 'DESIGNSPECIFICATION' in obj_type)
            field_normalized = field_name.replace('_', '').replace(' ', '')
            
            if is_outdoor_air_obj:
                # 更精确的字段匹配
                if ('PERPERSON' in field_normalized or 'FLOWPERPERSON' in field_normalized) and 'PEOPLEPERPERSON' not in field_normalized:
                    outdoor_air_mods['per_person'] = update
                    self.logger.debug(f"检测到 per_person 新风参数: {field_name}")
                elif 'PERZONE' in field_normalized and 'AREA' in field_normalized:
                    outdoor_air_mods['per_area'] = update
                    self.logger.debug(f"检测到 per_area 新风参数: {field_name}")
                else:
                    other_mods.append(update)
            else:
                other_mods.append(update)
        
        # 检查新风参数是否反向修改
        if 'per_person' in outdoor_air_mods and 'per_area' in outdoor_air_mods:
            person_update = outdoor_air_mods['per_person']
            area_update = outdoor_air_mods['per_area']
            
            person_coef = person_update.get('coefficient')
            area_coef = area_update.get('coefficient')
            
            if person_coef is not None and area_coef is not None:
                # 检查是否反向修改（一增一减）
                is_conflict = (person_coef > 1.0 and area_coef < 1.0) or (person_coef < 1.0 and area_coef > 1.0)
                
                if is_conflict:
                    self.logger.error(
                        f"✗ 检测到新风参数反向修改冲突！\n"
                        f"  - Outdoor_Air_Flow_per_Person: {person_coef:.2%}\n"
                        f"  - Outdoor_Air_Flow_per_Zone_Floor_Area: {area_coef:.2%}\n"
                        f"  这会导致总新风量反而增加！强制过滤掉会增加能耗的修改。"
                    )
                    
                    # 过滤策略：保留降低的修改，过滤增加的修改
                    if person_coef < 1.0:
                        other_mods.append(person_update)
                        self.logger.info(f"  ✓ 保留 per_person 修改（降低 {(1-person_coef)*100:.1f}%）")
                    else:
                        self.logger.warning(f"  ✗ 过滤 per_person 修改（增加 {(person_coef-1)*100:.1f}%，会导致反优化）")
                    
                    if area_coef < 1.0:
                        other_mods.append(area_update)
                        self.logger.info(f"  ✓ 保留 per_area 修改（降低 {(1-area_coef)*100:.1f}%）")
                    else:
                        self.logger.warning(f"  ✗ 过滤 per_area 修改（增加 {(area_coef-1)*100:.1f}%，会导致反优化）")
                else:
                    # 同向修改，都保留
                    other_mods.append(person_update)
                    other_mods.append(area_update)
            else:
                # 系数无法计算，保守起见都保留
                other_mods.append(person_update)
                other_mods.append(area_update)
        else:
            # 只修改了其中一个，保留
            if 'per_person' in outdoor_air_mods:
                other_mods.append(outdoor_air_mods['per_person'])
            if 'per_area' in outdoor_air_mods:
                other_mods.append(outdoor_air_mods['per_area'])
        
        return other_mods

    def _check_parameter_coherence(self, object_type, field_name, old_value, new_value, coefficient):
        """检查参数修改的相关性与物理一致性。
        
        针对相关参数的修改（如新风的两个参数），确保其修改方向一致，避免反优化。
        
        相关参数关系：
        - Outdoor_Air_Flow_per_Person 和 Outdoor_Air_Flow_per_Zone_Floor_Area：
          实际新风量 = max(per_person * occupants, per_area * zone_area)
          两者应该同向修改（都增或都减）
        """
        if not coefficient or coefficient is None:
            return None
        
        obj_upper = str(object_type).upper()
        field_upper = str(field_name).upper()
        
        # 检查新风相关参数的一致性
        if "OUTDOORAIR" in obj_upper and coefficient is not None:
            # 检查本轮其他新风参数的修改情况
            # 记录这次修改，用于后续回溯检查
            
            # 新风参数1：人均新风量
            if "PERP" in field_upper and "PERSON" in field_upper:
                # 检查是否同时修改了单位面积新风量
                # 存储当前修改信息以供后续检查
                self.current_outdoor_air_per_person_coef = coefficient
                
                # 如果之前已经修改了单位面积新风量，检查一致性
                if hasattr(self, 'current_outdoor_air_per_area_coef'):
                    area_coef = self.current_outdoor_air_per_area_coef
                    # 检查修改方向：应该同向（都>1或都<1）
                    if (coefficient > 1.0 and area_coef < 1.0) or (coefficient < 1.0 and area_coef > 1.0):
                        return f"新风参数修改方向相反（per_person={coefficient:.2%} vs per_area={area_coef:.2%}），可能导致反优化。应同向修改。"
            
            # 新风参数2：单位面积新风量
            elif "PERZ" in field_upper and "AREA" in field_upper:
                self.current_outdoor_air_per_area_coef = coefficient
                
                if hasattr(self, 'current_outdoor_air_per_person_coef'):
                    person_coef = self.current_outdoor_air_per_person_coef
                    if (coefficient > 1.0 and person_coef < 1.0) or (coefficient < 1.0 and person_coef > 1.0):
                        return f"新风参数修改方向相反（per_area={coefficient:.2%} vs per_person={person_coef:.2%}），可能导致反优化。应同向修改。"
        
        return None

    def _enforce_physical_constraints(self, target_updates):
        """对计划修改进行物理约束修正，避免生成非法 IDF。"""
        if not target_updates:
            return target_updates

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        def _norm(s):
            return str(s).upper().replace(" ", "")

        # 基础对象索引：用于读取未被修改字段的现值
        base_obj_lookup = {}
        for obj_type, objs in self.base_idf.idfobjects.items():
            obj_type_u = str(obj_type).upper()
            for obj in objs:
                obj_name_u = self._get_object_identifier(obj).upper()
                base_obj_lookup[(obj_type_u, obj_name_u)] = obj

        updates_by_key = {}
        for upd in target_updates:
            key = (str(upd.get('type', '')).upper(), str(upd.get('name', '')).upper(), str(upd.get('field', '')).upper())
            updates_by_key[key] = upd

        def _find_field_name(obj, candidate):
            candidate_norm = _norm(candidate)
            for fn in getattr(obj, 'fieldnames', []):
                if _norm(fn) == candidate_norm:
                    return fn
            return candidate

        def _get_value(obj_type_u, obj_name_u, field_name):
            key = (obj_type_u, obj_name_u, field_name.upper())
            if key in updates_by_key:
                return updates_by_key[key].get('value'), True

            obj = base_obj_lookup.get((obj_type_u, obj_name_u))
            if obj is None:
                return None, False

            resolved = _find_field_name(obj, field_name)
            return getattr(obj, resolved, None), False

        def _set_value(obj_type_u, obj_name_u, field_name, new_value, reason):
            key = (obj_type_u, obj_name_u, field_name.upper())
            if key in updates_by_key:
                upd = updates_by_key[key]
                old_plan_val = upd.get('value')
                upd['value'] = new_value
                old_num = _to_float(upd.get('old_value'))
                new_num = _to_float(new_value)
                upd['coefficient'] = (new_num / old_num) if (old_num not in (None, 0.0) and new_num is not None) else None
                upd['expression'] = f"{upd.get('expression', '')} | auto_constraint: {reason}".strip()
                self.logger.warning(
                    f"  [约束修正] {obj_type_u} ({obj_name_u}) {field_name}: {old_plan_val} -> {new_value} ({reason})"
                )
                return

            # 未在计划中但需要补充修正，追加一条自动修正更新
            base_obj = base_obj_lookup.get((obj_type_u, obj_name_u))
            if base_obj is None:
                return
            resolved = _find_field_name(base_obj, field_name)
            old_value = getattr(base_obj, resolved, None)
            old_num = _to_float(old_value)
            new_num = _to_float(new_value)
            coefficient = (new_num / old_num) if (old_num not in (None, 0.0) and new_num is not None) else None
            upd = {
                'type': obj_type_u,
                'name': obj_name_u,
                'field': resolved,
                'old_value': old_value,
                'value': new_value,
                'coefficient': coefficient,
                'expression': f"auto_constraint: {reason}"
            }
            target_updates.append(upd)
            updates_by_key[(obj_type_u, obj_name_u, resolved.upper())] = upd
            self.logger.warning(
                f"  [约束补充] {obj_type_u} ({obj_name_u}) {resolved}: {old_value} -> {new_value} ({reason})"
            )

        # 仅处理本次涉及的 WindowMaterial:Glazing 对象
        glazing_keys = set()
        for upd in target_updates:
            if str(upd.get('type', '')).upper() == 'WINDOWMATERIAL:GLAZING':
                glazing_keys.add((str(upd.get('type', '')).upper(), str(upd.get('name', '')).upper()))

        for obj_type_u, obj_name_u in glazing_keys:
            t_field = 'Solar_Transmittance_at_Normal_Incidence'
            rf_field = 'Front_Side_Solar_Reflectance_at_Normal_Incidence'
            rb_field = 'Back_Side_Solar_Reflectance_at_Normal_Incidence'

            t_val, t_mod = _get_value(obj_type_u, obj_name_u, t_field)
            rf_val, rf_mod = _get_value(obj_type_u, obj_name_u, rf_field)
            rb_val, rb_mod = _get_value(obj_type_u, obj_name_u, rb_field)

            t = _to_float(t_val)
            rf = _to_float(rf_val)
            rb = _to_float(rb_val)
            if t is None or rf is None or rb is None:
                continue

            # 基本区间约束 [0, 1]
            t2 = _clamp(t, 0.0, 1.0)
            rf2 = _clamp(rf, 0.0, 1.0)
            rb2 = _clamp(rb, 0.0, 1.0)
            if t2 != t:
                _set_value(obj_type_u, obj_name_u, t_field, t2, 'clamp_to_[0,1]')
            if rf2 != rf:
                _set_value(obj_type_u, obj_name_u, rf_field, rf2, 'clamp_to_[0,1]')
            if rb2 != rb:
                _set_value(obj_type_u, obj_name_u, rb_field, rb2, 'clamp_to_[0,1]')

            t, rf, rb = t2, rf2, rb2
            max_sum = 0.999

            # 组合约束：透射率 + 前/后表面反射率 <= 1
            # 若只有透射率被修改，优先下调透射率；否则下调对应反射率。
            if (t + rf) > max_sum:
                if t_mod and (not rf_mod):
                    new_t = _clamp(max_sum - rf, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, t_field, new_t, 'enforce T+FrontR<=1')
                    t = new_t
                else:
                    new_rf = _clamp(max_sum - t, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, rf_field, new_rf, 'enforce T+FrontR<=1')
                    rf = new_rf

            if (t + rb) > max_sum:
                if t_mod and (not rb_mod):
                    new_t = _clamp(max_sum - rb, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, t_field, new_t, 'enforce T+BackR<=1')
                    t = new_t
                else:
                    new_rb = _clamp(max_sum - t, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, rb_field, new_rb, 'enforce T+BackR<=1')
                    rb = new_rb

        return target_updates

    def _normalize_field_label(self, text):
        """标准化字段名/注释名，用于精确匹配（避免子串误匹配）。"""
        if text is None:
            return ""
        cleaned = str(text).upper()
        cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
        cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
        return cleaned

    def _get_object_identifier(self, obj):
        """获取对象实例标识，优先 Name，不存在时回退到常见主键字段。"""
        candidate_attrs = [
            "Name",
            "Zone_or_ZoneList_Name",
            "Zone_Name",
            "Space_Name",
            "AirLoop_Name",
            "Availability_Schedule_Name",
        ]

        for attr in candidate_attrs:
            try:
                val = getattr(obj, attr)
                if val is not None and str(val).strip() != "":
                    return str(val)
            except Exception:
                pass

        return ""
    
    def _check_modification_allowed(self, object_type, field_name):
        """
        检查是否允许修改指定字段（基于关键词规则）。
        
        对于某些字段，需要建议文本中包含特定关键词才能修改。
        例如：ZoneHVAC:IdealLoadsAirSystem的最大/最小供风温度阈值，
        只有建议中明确提到"阈值"时才能修改。
        
        Args:
            object_type: 对象类型（如"ZoneHVAC:IdealLoadsAirSystem"）
            field_name: 字段名（如"Minimum_Cooling_Supply_Air_Temperature"）
        
        Returns:
            bool: True表示允许修改，False表示不允许
        """
        # 标准化对象类型和字段名（不区分大小写）
        obj_type_upper = object_type.upper()
        field_upper = field_name.upper()
        
        # 检查是否有针对此对象类型的规则
        if obj_type_upper not in self.field_keyword_rules:
            return True  # 没有规则限制，允许修改
        
        obj_rules = self.field_keyword_rules[obj_type_upper]
        
        # 检查是否有针对此字段的规则
        matched_rule_keywords = None
        for rule_field, keywords in obj_rules.items():
            if rule_field.upper() == field_upper:
                matched_rule_keywords = keywords
                break
        
        # 没有针对此字段的规则，允许修改
        if matched_rule_keywords is None:
            return True
        
        # 有规则，检查上下文文本中是否包含至少一个必需关键词
        # 优先使用包含LLM建议内容的上下文，避免“建议里有阈值但仍被过滤”
        context_text = self.current_plan_context if self.current_plan_context else self.current_user_request
        context_text_lower = str(context_text).lower()
        for keyword in matched_rule_keywords:
            if str(keyword).lower() in context_text_lower:
                return True  # 找到关键词，允许修改
        
        # 没有找到任何必需关键词，不允许修改
        return False

    def _build_plan_context_text(self, plan):
        """构建字段关键词过滤所需的上下文文本。"""
        parts = [self.current_user_request or ""]
        if isinstance(plan, dict):
            reasoning = plan.get("reasoning", "")
            if reasoning:
                parts.append(str(reasoning))
            modifications = plan.get("modifications", [])
            if modifications:
                parts.append(json.dumps(modifications, ensure_ascii=False))
        return "\n".join([p for p in parts if p])
    
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
                obj_name = self._get_object_identifier(obj)
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
                    
                    # 检查参数修改的相关性与一致性
                    coherence_warnings = self._check_parameter_coherence(
                        target_obj_type, target_attr, old_value, final_value, coefficient
                    )
                    if coherence_warnings:
                        self.logger.warning(f"  ⚠️  参数相关性警告 ({target_obj_type}.{target_attr}): {coherence_warnings}")
                    
                    target_updates.append({
                        "type": target_obj_type,
                        "name": obj_name,
                        "field": target_attr,
                        "old_value": old_value,
                        "value": final_value,
                        "coefficient": coefficient,
                        "expression": value_expr
                    })

        if not target_updates:
            self.logger.warning("没有产生有效的修改计划，跳过保存")
            return
        
        # ========== 【关键】在应用前强制过滤相关参数反向修改 ==========
        target_updates = self._filter_conflicting_parameter_modifications(target_updates)

        # ========== 物理约束修正：防止生成非法IDF（如玻璃参数组合越界） ==========
        target_updates = self._enforce_physical_constraints(target_updates)
        
        if not target_updates:
            self.logger.warning("所有修改因参数冲突被过滤，跳过保存")
            return

        # ========== 【修改摘要】先于【计划修改】输出 ==========
        self.logger.info("\n" + "="*100)
        self.logger.info("【修改摘要】汇总 - 快速查看修改概览")
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
        self.logger.info("【计划修改】详情 - 逐一修改具体信息")
        self.logger.info("="*100)
        for update in target_updates:
            obj_type = update['type']
            obj_name = update['name']
            field = update['field']
            old_value = update['old_value']
            new_value = update['value']
            self.logger.info(f"  [计划修改] {obj_type} ({obj_name}): {field}")
            self.logger.info(f"    原值: {old_value} → 新值: {new_value}")
        self.logger.info("="*100)
        self.logger.info("开始执行IDF文本替换...")
        self.logger.info("="*100 + "\n")
        # ========== END 修改摘要和计划修改输出 ==========

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

        # 记录都有哪些计划修改
        planned_modifications_count = sum(len(v) for u in updates_map.values() for v in u.values())
        self.logger.info(f"\n计划修改总数: {planned_modifications_count}")
        executed_modifications = set()

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
                
                # 对象名识别：不同对象类型有不同的Name字段
                if "!- Name" in line:
                    val_part = line.split('!')[0].strip()
                    val_part = val_part.replace(',', '').replace(';', '').strip()
                    current_obj_name = val_part.upper()
                elif "!- Zone or ZoneList Name" in line and current_obj_type == "SIZING:ZONE":
                    # Sizing:Zone特殊处理：第一个字段就是Zone or ZoneList Name
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
                        matched_field_key = None
                        norm_comment = self._normalize_field_label(field_comment)
                        for target_field_key, target_val in target_fields.items():
                            norm_target = self._normalize_field_label(target_field_key)
                            if norm_target == norm_comment:
                                matched_val = target_val
                                matched_field_key = target_field_key
                                break
                        
                        if matched_val is not None:
                            # 关键词限制检查：某些字段需要建议中包含特定关键词才能修改
                            if not self._check_modification_allowed(current_obj_type, matched_field_key):
                                self.logger.info(
                                    f"  [跳过修改] {current_obj_type} ({current_obj_name}): {matched_field_key} "
                                    f"- 建议中未包含必需的关键词，不允许修改此字段"
                                )
                                new_lines.append(line)
                                continue
                            
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
                            exec_key = f"{current_obj_type}_{current_obj_name}_{matched_field_key}"
                            executed_modifications.add(exec_key)
                            self.logger.info(f"  [文本替换成功] {current_obj_type} ({current_obj_name}): {matched_field_key} -> {matched_val}")

            new_lines.append(line)

        # 检查是否有计划修改没有被执行
        self.logger.info(f"\n已执行修改数: {len(executed_modifications)}")
        if len(executed_modifications) < planned_modifications_count:
            unexecuted_count = planned_modifications_count - len(executed_modifications)
            self.logger.warning(f"⚠️ 警告：{unexecuted_count}个计划修改未在IDF中找到对应字段！")
            self.logger.warning("可能原因：")
            self.logger.warning("  1. LLM建议修改的字段在IDF中不存在")
            self.logger.warning("  2. 优化建议中提出的对象不符合当前支持的对象类型")
            self.logger.warning("  3. 字段名大小写或格式不匹配")

        with open(output_idf_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        self.logger.info(f"算例已保存至: {output_idf_path}")
        
        # ========== 本轮修改统计 ==========
        self._print_modification_statistics(target_updates)
        
        # ========== 更新字段修改频率统计 ==========
        self._update_field_modification_history(target_updates)

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
            f"- 单位面积总建筑能耗(EUI): {eui_saved_vs_baseline:+.2f} kWh/m² "
            f"({_safe_pct(eui_saved_vs_baseline, baseline_metrics['eui_kwh_per_m2']):+.2f}%)"
        )
        self.logger.info(
            f"- 制冷能耗: {cooling_saved_vs_baseline:+.2f} kWh/m² "
            f"({_safe_pct(cooling_saved_vs_baseline, baseline_metrics['total_cooling_kwh']):+.2f}%)"
        )
        self.logger.info(
            f"- 供暖能耗: {heating_saved_vs_baseline:+.2f} kWh/m² "
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
                f"- 单位面积总建筑能耗(EUI): {eui_saved_vs_prev:+.2f} kWh/m² "
                f"({_safe_pct(eui_saved_vs_prev, prev_metrics['eui_kwh_per_m2']):+.2f}%)"
            )
            self.logger.info(
                f"- 制冷能耗: {cooling_saved_vs_prev:+.2f} kWh/m² "
                f"({_safe_pct(cooling_saved_vs_prev, prev_metrics['total_cooling_kwh']):+.2f}%)"
            )
            self.logger.info(
                f"- 供暖能耗: {heating_saved_vs_prev:+.2f} kWh/m² "
                f"({_safe_pct(heating_saved_vs_prev, prev_metrics['total_heating_kwh']):+.2f}%)"
            )
    
    def run_optimization_loop(self, max_iterations=5):
        """运行并行工作流优化循环 - 5条工作流同时进行
        
        直接复制原有的单工作流逻辑 5 遍并行执行，不改变任何 prompt、代码逻辑、字段修改规则等。
        """
        self.logger.info(f"\n\n{'█'*80}")
        self.logger.info(f"启动{max_iterations}轮迭代优化 (并行{self.num_workflows}条工作流)")
        self.logger.info(f"{'█'*80}\n")
        
        # 并行调用 5 条工作流的优化循环
        with ThreadPoolExecutor(max_workers=self.num_workflows) as executor:
            futures = []
            for workflow_id in self.workflows.keys():
                future = executor.submit(
                    self._single_workflow_optimization_loop,
                    workflow_id,
                    max_iterations
                )
                futures.append(future)
            
            # 等待所有工作流完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"工作流优化出错: {e}")
        
        # 所有工作流完成后汇总报告
        self._print_parallel_final_report()
    
    def _single_workflow_optimization_loop(self, workflow_id, max_iterations):
        """单工作流优化循环 - 直接复制原有的单工作流逻辑
        
        参数:
            workflow_id: 工作流ID（如 "workflow_1"）
            max_iterations: 最大迭代次数
        """
        thread_handler = self._attach_workflow_thread_handler(workflow_id)
        try:
            self._log_context.workflow_id = workflow_id
            self.logger.info(f"\n\n{'█'*80}")
            self.logger.info(f"启动{max_iterations}轮迭代优化")
            self.logger.info(f"{'█'*80}\n")

            for iteration in range(1, max_iterations + 1):
                self.logger.info(f"\n\n{'═'*80}")
                self.logger.info(f"【第{iteration}轮/{max_iterations}】")
                self.logger.info(f"{'═'*80}\n")

                with self.workflows_lock:
                    current_idf_path = self.workflows[workflow_id]['current_idf_path']
                    self.current_idf_path = current_idf_path
                    self.current_workflow_id = workflow_id
                    self.iteration_history = self.workflows[workflow_id]['iteration_history']
                    self.best_metrics = self.workflows[workflow_id]['best_metrics']
                    self.best_iteration = self.workflows[workflow_id]['best_iteration']

                if iteration == 1:
                    sim_idf = current_idf_path
                    iter_name = "initial_baseline"
                else:
                    with self.workflows_lock:
                        current_metrics = self.workflows[workflow_id]['best_metrics']
                    if not current_metrics:
                        self.logger.error(f"第{iteration}轮无有效指标，中断优化")
                        break

                    # 生成优化建议（保持原逻辑）
                    with self.workflows_lock:
                        self.current_idf_path = self.workflows[workflow_id]['current_idf_path']
                        self.current_workflow_id = workflow_id
                        self.iteration_history = self.workflows[workflow_id]['iteration_history']
                    plan = self.generate_optimization_suggestions(current_metrics, iteration)
                    if not plan:
                        self.logger.warning(f"第{iteration}轮LLM无法生成建议，使用默认方案")
                        plan = self._get_default_suggestions(iteration)
                    if not plan:
                        self.logger.error(f"第{iteration}轮无有效计划，中断优化")
                        break

                    # 应用优化（保持并行安全）
                    with self.workflows_lock:
                        self.current_idf_path = self.workflows[workflow_id]['current_idf_path']
                        self.current_workflow_id = workflow_id
                        self.iteration_history = self.workflows[workflow_id]['iteration_history']
                        modified_idf_path = self.apply_optimization(plan, iteration)

                    if not modified_idf_path or not os.path.exists(modified_idf_path):
                        self.logger.warning(f"第{iteration}轮优化应用失败，使用当前IDF重新运行")
                        sim_idf = current_idf_path
                    else:
                        sim_idf = modified_idf_path
                        with self.workflows_lock:
                            self.workflows[workflow_id]['current_idf_path'] = modified_idf_path

                    iter_name = f"iteration_{iteration}"

                sim_dir = self.run_simulation(sim_idf, iter_name, workflow_id=workflow_id)
                if not sim_dir:
                    self.logger.error(f"第{iteration}轮模拟失败，中断优化")
                    break

                metrics = self.extract_metrics(sim_dir)
                if not metrics:
                    self.logger.error(f"第{iteration}轮无法提取能耗数据，中断优化")
                    break

                with self.workflows_lock:
                    self.workflows[workflow_id]['iteration_history'].append({
                        'iteration': iteration,
                        'metrics': metrics,
                        'idf_path': sim_idf,
                        'plan_description': '基准模拟' if iteration == 1 else '优化方案'
                    })
                    self.iteration_history = self.workflows[workflow_id]['iteration_history']
                    self.best_metrics = self.workflows[workflow_id]['best_metrics']
                    self.best_iteration = self.workflows[workflow_id]['best_iteration']

                self._print_iteration_savings(metrics, iteration)
                self.update_best_metrics(metrics, iteration)

                with self.workflows_lock:
                    self.workflows[workflow_id]['best_metrics'] = self.best_metrics
                    self.workflows[workflow_id]['best_iteration'] = self.best_iteration
            
        except Exception as e:
            self.logger.error(f"优化循环异常: {e}", exc_info=True)
        finally:
            if hasattr(self._log_context, 'workflow_id'):
                del self._log_context.workflow_id
            self._detach_workflow_thread_handler(thread_handler)

    def _print_parallel_final_report(self):
        """【并行工作流】汇总所有工作流的优化结果"""
        try:
            self.logger.info(f"\n\n{'='*80}")
            self.logger.info(f"【最终汇总报告 - {self.num_workflows}条并行工作流】")
            self.logger.info(f"{'='*80}\n")
            
            # 线程安全地读取所有工作流数据快照
            with self.workflows_lock:
                all_workflows = dict(self.workflows)

            # ---------- 随后为每个工作流打印完整的最优详情（也写入各自的工作流日志） ----------
            printed_workflow_optimal = set()
            global_best = None
            global_best_workflow = None
            global_best_energy = float('inf')

            for workflow_id in sorted(all_workflows.keys()):
                workflow_data = all_workflows[workflow_id]
                best_metrics = workflow_data.get('best_metrics')

                if best_metrics:
                    # 基本信息（同时写入汇总）
                    # 分隔符：每个工作流详情块前后清晰区分
                    self.logger.info(f"\n{'─'*80}")
                    self.logger.info(f"[{workflow_id}] 优化结果")
                    self.logger.info(f"{'─'*80}")
                    self.logger.info(f"  最优轮次: 第{workflow_data.get('best_iteration', 0)}轮")
                    self.logger.info(f"  总建筑能耗: {best_metrics.get('total_site_energy_kwh')} kWh")
                    self.logger.info(f"  单位面积能耗: {best_metrics.get('eui_kwh_per_m2')} kWh/m²")
                    self.logger.info(f"  制冷能耗: {best_metrics.get('total_cooling_kwh')} kWh/m²")
                    self.logger.info(f"  供暖能耗: {best_metrics.get('total_heating_kwh')} kWh/m²")

                    # 计算与基准的4个指标的变化 - 检查 iteration_history 是否为空
                    iteration_history = workflow_data.get('iteration_history', [])
                    wlogger = workflow_data.get('logger')
                    if iteration_history and len(iteration_history) > 0:
                        baseline_metrics = iteration_history[0].get('metrics', {})

                        # ---------- 打印各轮能耗对比表（既写汇总日志，也写工作流专属日志） ----------
                        header = "【各轮能耗对比】"
                        table_header = f"{'轮次':<8} {'总建筑能耗':<15} {'EUI':<15} {'冷却能耗':<15} {'供暖能耗':<15} {'状态'}"
                        self.logger.info(header)
                        if wlogger:
                            wlogger.info(header)
                        self.logger.info(table_header)
                        if wlogger:
                            wlogger.info(table_header)
                        self.logger.info("-" * 85)
                        if wlogger:
                            wlogger.info("-" * 85)

                        for item in iteration_history:
                            iteration = item.get('iteration')
                            m = item.get('metrics', {})
                            if iteration == 1:
                                status = '基准'
                            elif iteration == workflow_data.get('best_iteration'):
                                status = '最优 ✓'
                            else:
                                status = ''

                            # 基准值用于计算百分比变化
                            def _pct_str(curr, base):
                                try:
                                    base = float(base)
                                    curr = float(curr)
                                except Exception:
                                    return '(N/A)'
                                if base == 0:
                                    return '(N/A)'
                                pct = (curr - base) / base * 100
                                return f"({pct:+.1f}%)"

                            total_val = m.get('total_site_energy_kwh', 0)
                            eui_val = m.get('eui_kwh_per_m2', 0)
                            cool_val = m.get('total_cooling_kwh', 0)
                            heat_val = m.get('total_heating_kwh', 0)

                            total_pct = _pct_str(total_val, baseline_metrics.get('total_site_energy_kwh', 0))
                            eui_pct = _pct_str(eui_val, baseline_metrics.get('eui_kwh_per_m2', 0))
                            cool_pct = _pct_str(cool_val, baseline_metrics.get('total_cooling_kwh', 0))
                            heat_pct = _pct_str(heat_val, baseline_metrics.get('total_heating_kwh', 0))

                            total_str = f"{total_val:<12.2f} {total_pct:<8}"
                            eui_str = f"{eui_val:<12.2f} {eui_pct:<8}"
                            cool_str = f"{cool_val:<12.2f} {cool_pct:<8}"
                            heat_str = f"{heat_val:<12.2f} {heat_pct:<8}"

                            line = f"{iteration:<8} {total_str} {eui_str} {cool_str} {heat_str} {status}"
                            self.logger.info(line)
                            if wlogger:
                                wlogger.info(line)

                        # 在表格之后补充两行：最优来源与对应IDF路径（与单工作流格式一致）
                        best_iter = workflow_data.get('best_iteration', 0)
                        best_idf_path = None
                        if best_iter and iteration_history and 1 <= best_iter <= len(iteration_history):
                            best_idf_path = iteration_history[best_iter - 1].get('idf_path')

                        best_from_line = f"最优方案来自: 第{best_iter}轮优化" if best_iter else "最优方案来自: N/A"
                        idf_line = f"对应IDF: {best_idf_path or 'N/A'}"
                        self.logger.info("")
                        self.logger.info(best_from_line)
                        self.logger.info(idf_line)
                        if wlogger:
                            wlogger.info("")
                            wlogger.info(best_from_line)
                            wlogger.info(idf_line)

                        # 计算节能效果
                        total_energy_saved = baseline_metrics.get('total_site_energy_kwh', 0) - best_metrics.get('total_site_energy_kwh', 0)
                        total_energy_pct = (total_energy_saved / baseline_metrics.get('total_site_energy_kwh', 1) * 100) if baseline_metrics.get('total_site_energy_kwh', 0) > 0 else 0

                        eui_saved = baseline_metrics.get('eui_kwh_per_m2', 0) - best_metrics.get('eui_kwh_per_m2', 0)
                        eui_pct = (eui_saved / baseline_metrics.get('eui_kwh_per_m2', 1) * 100) if baseline_metrics.get('eui_kwh_per_m2', 0) > 0 else 0

                        cooling_saved = baseline_metrics.get('total_cooling_kwh', 0) - best_metrics.get('total_cooling_kwh', 0)
                        cooling_pct = (cooling_saved / baseline_metrics.get('total_cooling_kwh', 1) * 100) if baseline_metrics.get('total_cooling_kwh', 0) > 0 else 0

                        heating_saved = baseline_metrics.get('total_heating_kwh', 0) - best_metrics.get('total_heating_kwh', 0)
                        heating_pct = (heating_saved / baseline_metrics.get('total_heating_kwh', 1) * 100) if baseline_metrics.get('total_heating_kwh', 0) > 0 else 0

                        # 输出4个指标的节能百分比（也会写入工作流日志）
                        summary_lines = [
                            "  【优化效果】",
                            f"  总建筑能耗节能: {total_energy_saved:.2f} kWh ({total_energy_pct:.1f}%)",
                            f"  单位面积总建筑能耗改善: {eui_saved:.2f} kWh/m² ({eui_pct:.1f}%)",
                            f"  冷却能耗节能: {cooling_saved:.2f} kWh ({cooling_pct:.1f}%)",
                            f"  供暖能耗节能: {heating_saved:.2f} kWh ({heating_pct:.1f}%)",
                            f"  迭代次数: {len(iteration_history)}轮"
                        ]
                        for sl in summary_lines:
                            self.logger.info(sl)
                            if wlogger:
                                wlogger.info(sl)
                    else:
                        self.logger.warning(f"  ⚠️ {workflow_id} 无迭代历史记录")

                    # 记录全局最优候选
                    if best_metrics.get('total_site_energy_kwh', float('inf')) < global_best_energy:
                        global_best_energy = best_metrics.get('total_site_energy_kwh')
                        global_best = best_metrics
                        global_best_workflow = workflow_id

                    # ========== 打印该工作流的最优参数逐项对比（旧单工作流格式），并写入工作流专属日志和汇总日志 ==========
                    try:
                        best_iter = workflow_data.get('best_iteration', 0)
                        if best_iter and iteration_history and 1 <= best_iter <= len(iteration_history):
                            best_idf = iteration_history[best_iter - 1].get('idf_path')
                            lines = self._format_optimal_parameters(self.idf_path, best_idf)

                            # 写入汇总日志（带前缀）
                            try:
                                self._log_context.workflow_id = workflow_id
                                for l in lines:
                                    self.logger.info(l)
                            finally:
                                if hasattr(self._log_context, 'workflow_id'):
                                    del self._log_context.workflow_id

                            # 写入工作流专属日志（旧格式）
                            wlogger = workflow_data.get('logger')
                            if wlogger:
                                for l in lines:
                                    wlogger.info(l)

                            printed_workflow_optimal.add(workflow_id)
                        else:
                            self.logger.info(f"[{workflow_id}] 无有效最优轮次或最优IDF，跳过最优参数逐项对比输出")
                    except Exception as _e:
                        self.logger.warning(f"打印{workflow_id}最优参数逐项对比时出错: {_e}")
                else:
                    self.logger.info(f"\n[{workflow_id}] 无有效最优结果")

                    # ========== 打印该工作流的最优参数逐项对比（旧单工作流格式） ==========
                    try:
                        iteration_history = workflow_data.get('iteration_history', [])
                        best_iter = workflow_data.get('best_iteration', 0)
                        if best_iter and iteration_history and 1 <= best_iter <= len(iteration_history):
                            best_idf = iteration_history[best_iter - 1].get('idf_path')
                            # 生成文本行
                            lines = self._format_optimal_parameters(self.idf_path, best_idf)

                            # 写入汇总日志（带前缀）
                            try:
                                self._log_context.workflow_id = workflow_id
                                for l in lines:
                                    self.logger.info(l)
                            finally:
                                if hasattr(self._log_context, 'workflow_id'):
                                    del self._log_context.workflow_id

                            # 写入工作流专属日志（旧格式）
                            wlogger = workflow_data.get('logger')
                            if wlogger:
                                for l in lines:
                                    wlogger.info(l)
                        else:
                            self.logger.info(f"[{workflow_id}] 无有效最优轮次或最优IDF，跳过最优参数逐项对比输出")
                    except Exception as _e:
                        self.logger.warning(f"打印{workflow_id}最优参数逐项对比时出错: {_e}")
            
            # 打印全局最优结果
            # ---------- 在所有工作流详情输出完成后，生成并打印各工作流最优能耗对比表（汇总） ----------
            try:
                summary_rows = []
                for workflow_id in sorted(all_workflows.keys()):
                    wd = all_workflows[workflow_id]
                    bm = wd.get('best_metrics')
                    # 尝试获取基准值用于计算百分比
                    baseline = None
                    iteration_history = wd.get('iteration_history') or []
                    if iteration_history and len(iteration_history) > 0:
                        baseline = iteration_history[0].get('metrics', {})

                    if bm:
                        total = bm.get('total_site_energy_kwh', None)
                        eui = bm.get('eui_kwh_per_m2', None)
                        cooling = bm.get('total_cooling_kwh', None)
                        heating = bm.get('total_heating_kwh', None)

                        def _pct(curr, base):
                            try:
                                if base is None or float(base) == 0:
                                    return 0
                                return (float(curr) - float(base)) / float(base) * 100
                            except Exception:
                                return 0

                        total_pct = _pct(total, (baseline.get('total_site_energy_kwh') if baseline else None))
                        eui_pct = _pct(eui, (baseline.get('eui_kwh_per_m2') if baseline else None))
                        cooling_pct = _pct(cooling, (baseline.get('total_cooling_kwh') if baseline else None))
                        heating_pct = _pct(heating, (baseline.get('total_heating_kwh') if baseline else None))

                        summary_rows.append({
                            'workflow': workflow_id,
                            'best_iter': wd.get('best_iteration', 0),
                            'total': total,
                            'eui': eui,
                            'cooling': cooling,
                            'heating': heating,
                            'total_pct': total_pct,
                            'eui_pct': eui_pct,
                            'cooling_pct': cooling_pct,
                            'heating_pct': heating_pct
                        })
                    else:
                        summary_rows.append({
                            'workflow': workflow_id,
                            'best_iter': 0,
                            'total': None,
                            'eui': None,
                            'cooling': None,
                            'heating': None,
                            'total_pct': 0,
                            'eui_pct': 0,
                            'cooling_pct': 0,
                            'heating_pct': 0
                        })

                self.logger.info("\n【各工作流最优能耗对比】")
                self.logger.info(f"{'Workflow':<12} {'BestIter':<8} {'TotalEnergy(kWh)':<22} {'EUI(kWh/m2)':<18} {'Cooling(kWh)':<18} {'Heating(kWh)':<18}")
                self.logger.info('=' * 100)
                for row in summary_rows:
                    def _fmt(v):
                        try:
                            return float(v)
                        except Exception:
                            return None

                    def _fmt_val_pct(val, pct):
                        if val is None:
                            return 'N/A'
                        try:
                            return f"{val:.2f} ({pct:+.1f}%)"
                        except Exception:
                            return str(val)

                    total = _fmt(row['total'])
                    eui = _fmt(row['eui'])
                    cooling = _fmt(row['cooling'])
                    heating = _fmt(row['heating'])

                    total_display = _fmt_val_pct(total, row.get('total_pct', 0)) if total is not None else 'N/A'
                    eui_display = _fmt_val_pct(eui, row.get('eui_pct', 0)) if eui is not None else 'N/A'
                    cooling_display = _fmt_val_pct(cooling, row.get('cooling_pct', 0)) if cooling is not None else 'N/A'
                    heating_display = _fmt_val_pct(heating, row.get('heating_pct', 0)) if heating is not None else 'N/A'

                    self.logger.info(f"{row['workflow']:<12} {row['best_iter']:<8} {total_display:<28} {eui_display:<22} {cooling_display:<22} {heating_display:<22}")
                self.logger.info('\n')
            except Exception:
                pass
            if global_best and global_best_workflow:
                self.logger.info(f"\n\n{'─'*80}")
                self.logger.info(f"【🏆 全局最优方案】{global_best_workflow}")
                self.logger.info(f"{'─'*80}")
                self.logger.info(f"总建筑能耗: {global_best['total_site_energy_kwh']} kWh")
                self.logger.info(f"单位面积能耗: {global_best['eui_kwh_per_m2']} kWh/m²")
                self.logger.info(f"制冷能耗: {global_best['total_cooling_kwh']} kWh/m²")
                self.logger.info(f"供暖能耗: {global_best['total_heating_kwh']} kWh/m²")
                
                # 计算全局最优的4个指标与基准的变化 - 检查 iteration_history 是否为空
                if all_workflows[global_best_workflow]['iteration_history'] and len(all_workflows[global_best_workflow]['iteration_history']) > 0:
                    baseline_metrics = all_workflows[global_best_workflow]['iteration_history'][0]['metrics']
                    
                    # 计算4个指标的变化
                    total_energy_saved = baseline_metrics['total_site_energy_kwh'] - global_best['total_site_energy_kwh']
                    total_energy_pct = (total_energy_saved / baseline_metrics['total_site_energy_kwh'] * 100) if baseline_metrics['total_site_energy_kwh'] > 0 else 0
                    
                    eui_saved = baseline_metrics['eui_kwh_per_m2'] - global_best['eui_kwh_per_m2']
                    eui_pct = (eui_saved / baseline_metrics['eui_kwh_per_m2'] * 100) if baseline_metrics['eui_kwh_per_m2'] > 0 else 0
                    
                    cooling_saved = baseline_metrics['total_cooling_kwh'] - global_best['total_cooling_kwh']
                    cooling_pct = (cooling_saved / baseline_metrics['total_cooling_kwh'] * 100) if baseline_metrics['total_cooling_kwh'] > 0 else 0
                    
                    heating_saved = baseline_metrics['total_heating_kwh'] - global_best['total_heating_kwh']
                    heating_pct = (heating_saved / baseline_metrics['total_heating_kwh'] * 100) if baseline_metrics['total_heating_kwh'] > 0 else 0
                    
                    # 输出4个指标的节能百分比
                    self.logger.info(f"【较基准节能汇总】")
                    self.logger.info(f"  - 总建筑能耗: {total_energy_saved:+.2f} kWh ({total_energy_pct:+.1f}%)")
                    self.logger.info(f"  - 单位面积能耗: {eui_saved:+.2f} kWh/m² ({eui_pct:+.1f}%)")
                    self.logger.info(f"  - 制冷能耗: {cooling_saved:+.2f} kWh/m² ({cooling_pct:+.1f}%)")
                    self.logger.info(f"  - 供暖能耗: {heating_saved:+.2f} kWh/m² ({heating_pct:+.1f}%)")
                    
                    # 保存最优IDF路径 - 检查索引是否合法
                    iteration_history = all_workflows[global_best_workflow]['iteration_history']
                    best_iteration = all_workflows[global_best_workflow]['best_iteration']
                    if best_iteration > 0 and best_iteration <= len(iteration_history):
                        best_idf_path = iteration_history[best_iteration - 1]['idf_path']
                        self.logger.info(f"最优IDF文件: {best_idf_path}")
                    else:
                        self.logger.warning(f"⚠️ 无效的最优轮次索引: {best_iteration}")
                else:
                    self.logger.warning(f"⚠️ 全局最优方案所在工作流无迭代历史记录")

                # ========== 打印全局最优参数逐项对比（格式同单工作流） ==========
                try:
                    if global_best and global_best_workflow:
                        gh_iteration_history = all_workflows[global_best_workflow].get('iteration_history', [])
                        gh_best_iter = all_workflows[global_best_workflow].get('best_iteration', 0)
                        if gh_best_iter and gh_iteration_history and 1 <= gh_best_iter <= len(gh_iteration_history):
                            gh_best_idf = gh_iteration_history[gh_best_iter - 1].get('idf_path')
                            gh_lines = self._format_optimal_parameters(self.idf_path, gh_best_idf)
                            # 写到汇总日志（无需额外前缀，因为已在全局上下文）
                            for ln in gh_lines:
                                self.logger.info(ln)
                            # 同时写入该工作流的专属日志（如果存在且尚未为该工作流写过相同最优输出）
                            wlogger = all_workflows[global_best_workflow].get('logger')
                            if wlogger:
                                if global_best_workflow not in printed_workflow_optimal:
                                    for ln in gh_lines:
                                        wlogger.info(ln)
                                    printed_workflow_optimal.add(global_best_workflow)
                                else:
                                    # 如果已在先前为该工作流写过最优详情，则跳过写入以避免重复
                                    self.logger.debug(f"已为{global_best_workflow}写入最优详情，跳过重复写入到其工作流日志")
                        else:
                            self.logger.info("全局最优没有有效IDF路径，跳过详细参数对比输出")
                except Exception as _e:
                    self.logger.warning(f"打印全局最优参数逐项对比时出错: {_e}")
            
            # Token使用统计
            self.logger.info(f"\n{'─'*80}")
            self.logger.info(f"【Token使用统计】")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"总Token消耗: {self.total_tokens_used}")
            self.logger.info(f"LLM调用次数: {self.llm_calls_count}")
            if self.llm_calls_count > 0:
                avg_tokens = self.total_tokens_used / self.llm_calls_count
                self.logger.info(f"平均每次Token数: {avg_tokens:.2f}")

            # 生成并行可视化曲线：每个工作流单独图 + 所有工作流汇总图
            self._generate_parallel_energy_plots(all_workflows)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"【并行优化循环完成】")
            self.logger.info(f"{'='*80}\n")
            
        except Exception as e:
            self.logger.error(f"最终报告汇总异常: {e}", exc_info=True)

    def _generate_parallel_energy_plots(self, all_workflows):
        """生成并行工作流能耗变化曲线。

        输出内容：
        1) 每个工作流单独图：供冷实线、供暖虚线
        2) 所有工作流汇总图：不同工作流使用不同颜色，供冷实线、供暖虚线
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            plot_dir = "optimization_plot_并行2"
            os.makedirs(plot_dir, exist_ok=True)

            # 尽量设置中文字体，找不到则使用默认字体继续绘图
            try:
                import matplotlib.font_manager as fm
                font_candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
                font_name = None
                for candidate in font_candidates:
                    try:
                        fm.findfont(candidate, fallback_to_default=False)
                        font_name = candidate
                        break
                    except Exception:
                        continue
                if font_name:
                    matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # ---------- 1) 每个工作流单独曲线 ----------
            valid_workflows = []
            for workflow_id in sorted(all_workflows.keys()):
                history = all_workflows[workflow_id].get('iteration_history', [])
                if not history:
                    self.logger.warning(f"[{workflow_id}] 无迭代历史，跳过单工作流曲线生成")
                    continue

                x = [item.get('iteration', idx + 1) for idx, item in enumerate(history)]
                cooling = [item['metrics']['total_cooling_kwh'] for item in history]
                heating = [item['metrics']['total_heating_kwh'] for item in history]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x, cooling, color='tab:blue', linestyle='-', marker='o', linewidth=2,
                        label='Cooling (solid)')
                ax.plot(x, heating, color='tab:orange', linestyle='--', marker='s', linewidth=2,
                        label='Heating (dashed)')

                ax.set_title(f"{workflow_id} Cooling/Heating Energy Curve")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Energy (kWh/m2)")
                ax.grid(True, alpha=0.3)
                ax.legend()

                workflow_plot_path = os.path.join(plot_dir, f"{workflow_id}_cooling_heating_curve_{timestamp}.png")
                fig.savefig(workflow_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

                self.logger.info(f"[{workflow_id}] 已保存单工作流能耗曲线: {workflow_plot_path}")
                valid_workflows.append((workflow_id, x, cooling, heating))

            # ---------- 2) 所有工作流汇总曲线 ----------
            if valid_workflows:
                fig, ax = plt.subplots(figsize=(12, 7))
                cmap = plt.get_cmap('tab10')

                for idx, (workflow_id, x, cooling, heating) in enumerate(valid_workflows):
                    color = cmap(idx % 10)
                    # 供冷：实线；供暖：虚线；同一workflow同色
                    ax.plot(x, cooling, color=color, linestyle='-', marker='o', linewidth=2,
                            label=f"{workflow_id} Cooling")
                    ax.plot(x, heating, color=color, linestyle='--', marker='s', linewidth=2,
                            label=f"{workflow_id} Heating")

                ax.set_title("All Workflows Cooling/Heating Energy Curves")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Energy (kWh/m2)")
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2, fontsize=9)

                summary_plot_path = os.path.join(plot_dir, f"all_workflows_cooling_heating_curve_{timestamp}.png")
                fig.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

                self.logger.info(f"已保存并行汇总能耗曲线: {summary_plot_path}")
            else:
                self.logger.warning("无有效工作流历史，未生成汇总能耗曲线")

        except Exception as e:
            self.logger.warning(f"生成并行能耗曲线失败: {e}")
    
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
    
    def _format_optimal_parameters(self, baseline_idf_path, optimal_idf_path):
        """比较基准IDF与最优IDF，生成逐项参数修改的文本行列表（不直接写日志）。

        返回: list[str]，每一行为一条待写入日志的文本
        """
        lines = []
        if not os.path.exists(optimal_idf_path):
            lines.append(f"最优IDF文件不存在: {optimal_idf_path}")
            return lines

        try:
            baseline_idf = IDF(baseline_idf_path)
            optimal_idf = IDF(optimal_idf_path)

            modifications = []
            modified_objects = set()

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

                    for field in optimal_obj.fieldnames:
                        try:
                            baseline_val = getattr(baseline_obj, field, None)
                            optimal_val = getattr(optimal_obj, field, None)
                            if baseline_val != optimal_val:
                                if field.lower() not in ['name', 'type']:
                                    try:
                                        baseline_num = float(baseline_val) if baseline_val is not None and str(baseline_val) != '' else 0
                                        optimal_num = float(optimal_val) if optimal_val is not None and str(optimal_val) != '' else 0
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
                        except Exception:
                            continue

            # format lines similar to 原始单工作流格式
            if modifications:
                lines.append(f"【修改对象类型】({len(modified_objects)}种)")
                lines.append(', '.join(sorted(modified_objects)))
                lines.append("")
                lines.append(f"【参数修改详情】(共{len(modifications)}项修改)")

                current_obj_type = None
                def _fmt_num(n):
                    try:
                        n = float(n)
                    except Exception:
                        return str(n)
                    an = abs(n)
                    if an == 0:
                        return "0"
                    if an < 1e-4:
                        return f"{n:.8f}"
                    if an < 1e-2:
                        return f"{n:.6f}"
                    if an < 1:
                        return f"{n:.4f}"
                    if an < 1000:
                        return f"{n:.2f}"
                    return f"{n:.2f}"

                for mod in sorted(modifications, key=lambda x: (x['object_type'], x['object_name'])):
                    if mod['object_type'] != current_obj_type:
                        current_obj_type = mod['object_type']
                        lines.append(f"\n━━ {current_obj_type} ━━")

                    obj_name = mod['object_name']
                    field = mod['field']
                    baseline = mod['baseline_value']
                    optimal = mod['optimal_value']
                    change_pct = mod['change_pct']

                    if isinstance(change_pct, str):
                        change_str = f"{baseline} → {optimal}"
                        lines.append(f"  • {obj_name}")
                        lines.append(f"    └─ {field}: {change_str}")
                    else:
                        # 使用更高精度输出，避免非常小的数被截断为0
                        baseline_str = _fmt_num(baseline)
                        optimal_str = _fmt_num(optimal)
                        change_str = f"{baseline_str} → {optimal_str}"
                        pct_str = f"({change_pct:+.2f}%)" if change_pct != 0 else ""
                        lines.append(f"  • {obj_name}")
                        lines.append(f"    └─ {field}: {change_str} {pct_str}")
            else:
                lines.append("未检测到参数修改（可能原始文件和优化文件相同）")

            return lines
        except Exception as e:
            return [f"解析最优方案修改详情时出错: {e}"]
    #                 
    #                 if isinstance(change_pct, str):
    #                     change_str = f"{baseline} → {optimal}"
    #                     self.logger.info(f"  • {obj_name}")
    #                     self.logger.info(f"    └─ {field}: {change_str}")
    #                 else:
    #                     change_str = f"{baseline:.4f} → {optimal:.4f}"
    #                     pct_str = f"({change_pct:+.2f}%)" if change_pct != 0 else ""
    #                     self.logger.info(f"  • {obj_name}")
    #                     self.logger.info(f"    └─ {field}: {change_str} {pct_str}")
    #         else:
    #             self.logger.info("未检测到参数修改（可能原始文件和优化文件相同）")
    #         
    #         self.logger.info(f"\n{'═'*80}\n")
    #         
    #     except Exception as e:
    #         self.logger.error(f"解析最优方案修改详情时出错: {e}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
    
    # def _print_final_report(self):
    #     """打印最终优化报告"""
    #     self.logger.info(f"\n\n{'█'*80}")
    #     self.logger.info(f"优化完成总结报告")
    #     self.logger.info(f"{'█'*80}\n")
        
        # 先打印最优参数修改细节
        #self._extract_and_print_optimal_parameters()
        
        #if not self.iteration_history:
            #self.logger.info("无优化数据")
            #return
        
        #initial_metrics = self.iteration_history[0]['metrics']
        
        #self.logger.info("【各轮能耗对比】\n")
        #self.logger.info(f"{'轮次':<8} {'总建筑能耗':<15} {'EUI':<15} {'冷却能耗':<15} {'供暖能耗':<15} {'状态'}")
        #self.logger.info("-" * 85)
        
        #for item in self.iteration_history:
            #iteration = item['iteration']
            #m = item['metrics']
            
            #if iteration == 1:
                #status = "基准"
            #elif iteration == self.best_iteration:
                #status = "最优 ✓"
            #else:
                #status = ""
            
            #self.logger.info(
                #f"{iteration:<8} {m['total_site_energy_kwh']:<15.2f} "
                #f"{m['eui_kwh_per_m2']:<15.2f} {m['total_cooling_kwh']:<15.2f} "
                #f"{m['total_heating_kwh']:<15.2f} {status}"
            #)
        
        # 计算最优值相比初始值的节能效果
        #if self.best_metrics:
            #total_savings = initial_metrics['total_site_energy_kwh'] - self.best_metrics['total_site_energy_kwh']
            #total_savings_pct = (total_savings / initial_metrics['total_site_energy_kwh'] * 100)
            
            #eui_savings = initial_metrics['eui_kwh_per_m2'] - self.best_metrics['eui_kwh_per_m2']
            #eui_savings_pct = (eui_savings / initial_metrics['eui_kwh_per_m2'] * 100)
            
            #cool_savings = initial_metrics['total_cooling_kwh'] - self.best_metrics['total_cooling_kwh']
            #cool_savings_pct = (cool_savings / initial_metrics['total_cooling_kwh'] * 100) if initial_metrics['total_cooling_kwh'] > 0 else 0
            
            #heat_savings = initial_metrics['total_heating_kwh'] - self.best_metrics['total_heating_kwh']
            #heat_savings_pct = (heat_savings / initial_metrics['total_heating_kwh'] * 100) if initial_metrics['total_heating_kwh'] > 0 else 0
            
            #self.logger.info(f"\n【优化效果】(第{self.best_iteration}轮最优)\n")
            #self.logger.info(f"总建筑能耗节能: {total_savings:.2f} kWh ({total_savings_pct:.1f}%)")
            #self.logger.info(f"单位面积总建筑能耗改善: {eui_savings:.2f} kWh/m² ({eui_savings_pct:.1f}%)")
            #self.logger.info(f"冷却能耗节能: {cool_savings:.2f} kWh ({cool_savings_pct:.1f}%)")
            #self.logger.info(f"供暖能耗节能: {heat_savings:.2f} kWh ({heat_savings_pct:.1f}%)")
            
            #self.logger.info(f"\n【最优参数】\n")
            #self.logger.info(f"最优方案来自: 第{self.best_iteration}轮优化")
            #self.logger.info(f"对应IDF: {self.iteration_history[self.best_iteration-1]['idf_path']}")
        
        # Token使用统计
        #self.logger.info(f"\n{'='*80}")
        #self.logger.info(f"【Token使用统计】")
        #self.logger.info(f"{'='*80}")
        #self.logger.info(f"LLM调用次数: {self.llm_calls_count}")
        #self.logger.info(f"总计消耗Token: {self.total_tokens_used:,}")
        #self.logger.info(f"  - Input Tokens: {getattr(self, 'total_input_tokens', 0):,}")
        #self.logger.info(f"  - Output Tokens: {getattr(self, 'total_output_tokens', 0):,}")
        #self.logger.info(f"  - Cached Input Tokens: {getattr(self, 'total_cached_input_tokens', 0):,}")

        # 变化曲线可视化
        #try:
            #import matplotlib.pyplot as plt
            #import matplotlib
            #import os
            # 设置中文字体（优先使用 SimHei、Microsoft YaHei）
            # 中英文混合字体，保证“²”等符号和中文都能显示
            #import matplotlib.font_manager as fm
            #font_candidates = ["Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans", "SimHei", "STHeiti", "Arial"]
            #font_path = None
            #for font_name in font_candidates:
                #font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                #for f in font_list:
                    #if font_name.lower() in fm.FontProperties(fname=f).get_name().lower():
                        #font_path = f
                        #break
                #if font_path:
                    #break
            #if font_path:
                #font_prop = fm.FontProperties(fname=font_path)
                #matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
                #matplotlib.rcParams['axes.unicode_minus'] = False
            #else:
                #font_prop = None
                #self.logger.warning("未找到理想的中英文混合字体，可能仍有部分符号无法显示。建议安装 Microsoft YaHei 或 Arial Unicode MS 字体。")
            #plot_dir = "optimization_plot_并行"
            #os.makedirs(plot_dir, exist_ok=True)
            #cooling = [item['metrics']['total_cooling_kwh'] for item in self.iteration_history]
            #heating = [item['metrics']['total_heating_kwh'] for item in self.iteration_history]
            #x = list(range(1, len(cooling)+1))
            #plt.figure(figsize=(8,5))
            #plt.plot(x, cooling, marker='o', label='冷却能耗 (kWh/m²)')
            #plt.plot(x, heating, marker='o', label='供暖能耗 (kWh/m²)')
            #if font_prop:
                #plt.xlabel('优化轮次', fontproperties=font_prop)
                #plt.ylabel('能耗 (kWh/m²)', fontproperties=font_prop)
                #plt.title('供冷/供暖能耗优化变化曲线', fontproperties=font_prop)
                #plt.legend(prop=font_prop)
            #else:
                #plt.xlabel('优化轮次')
                #plt.ylabel('能耗 (kWh/m²)')
                #plt.title('供冷/供暖能耗优化变化曲线')
                #plt.legend()
            #plt.grid(True)
            #from datetime import datetime
            #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            #plot_path = os.path.join(plot_dir, f'cooling_heating_curve_{timestamp}.png')
            #plt.savefig(plot_path, bbox_inches='tight')
            #plt.close()
            #self.logger.info(f"已保存供冷/供暖能耗变化曲线: {plot_path}")
        #except Exception as e:
            #self.logger.warning(f"保存能耗变化曲线失败: {e}")
        
        # 计算API剩余tokens（假设API总配额，这里需要根据实际情况调整）
        # GPT-5.2的常见配额范围，这里以一个示例值展示
        # 注意：实际配额需要从API账户查询，这里仅作演示
        #try:
            #with open("api_key.txt", 'r') as f:
                #api_key = f.read().strip()
            
            # 这里使用一个假设的总配额值（用户需根据实际情况调整）
            # 不同API key有不同的配额限制，可以从OpenAI dashboard查看
            # 这里假设总配额为 10,000,000 tokens（仅为示例）
            #assumed_total_quota = 10_000_000
            #remaining_tokens = assumed_total_quota - self.total_tokens_used
            
            #self.logger.info(f"估算剩余Token: {remaining_tokens:,} (基于假设总配额 {assumed_total_quota:,})")
            #self.logger.info(f"")
            #self.logger.info(f"⚠️  注意：剩余Token为估算值，实际配额请登录OpenAI账户查看")
            #self.logger.info(f"    不同API key配额不同，上述计算基于假设值 {assumed_total_quota:,} tokens")
        #except Exception as e:
            #self.logger.warning(f"无法读取API key文件: {e}")
        
        #self.logger.info(f"{'='*80}\n")
        
        #self.logger.info(f"\n{'█'*80}\n")
    
    # ========== 未使用方法注释结束 ==========


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
