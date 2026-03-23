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
from time import perf_counter
from pathlib import Path
from collections import defaultdict
from eppy.modeleditor import IDF
from openai import OpenAI
from knowledge_base import EnergyPlusKnowledgeBase

# 强制使用无界面绘图后端，避免Tkinter在多线程退出时触发Tcl_AsyncDelete崩溃
os.environ.setdefault("MPLBACKEND", "Agg")


class EnergyPlusOptimizer:
    """EnergyPlus 5轮并行工作流迭代自动优化系统"""
    
    def __init__(
        self,
        idf_path,
        idd_path,
        api_key_path,
        epw_path="weather.epw",
        log_dir="optimization_logs_并行",
        optimization_dir="optimization_results_并行",
        plot_dir="optimization_plot_并行",
        num_workflows=1
    ):
        self.idf_path = idf_path
        self.idd_path = idd_path
        self.epw_path = epw_path
        self.log_dir = log_dir
        self.optimization_dir = optimization_dir
        self.plot_dir = plot_dir
        # 线程级日志上下文：仅用于汇总日志添加[workflow_x]前缀
        self._log_context = threading.local()
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.optimization_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
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

        # 早停配置：每条工作流独立判定是否收敛
        self.early_stop_enabled = True
        self.early_stop_target_total_saving_pct = 60.0      # 达到目标节能率可提前停止
        self.early_stop_min_iterations = 4                  # 至少完成基准+3轮后再判定
        self.early_stop_convergence_patience = 2            # 连续N次增益极小判定收敛
        self.early_stop_min_delta_pct = 2                   # 节能率变化阈值（百分点）
        self.max_iterations_cap = 20                        # 自动模式下的最大安全上限
        
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
                'suggestion_object_frequency': {},
                'suggestion_field_frequency': {},
                'workflow_total_duration_sec': 0.0,
                'llm_total_duration_sec': 0.0,
                'sim_total_duration_sec': 0.0,
                'llm_call_records': [],
                'sim_call_records': [],
                'llm_token_records': [],
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
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_input_tokens = 0

        # 全局并行耗时统计
        self.parallel_total_duration_sec = 0.0
        
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
                record.workflow_prefix = f"{self_outer._format_scope_tag(workflow_id)} "
                return True

        self_outer = self
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

    def _format_scope_tag(self, workflow_id=None):
        """统一日志作用域标记：有工作流时输出【工作流N】，否则输出【总】。"""
        if not workflow_id:
            return "【总】"

        wid = str(workflow_id)
        match = re.match(r"^workflow_(\d+)$", wid, flags=re.IGNORECASE)
        if match:
            return f"【工作流{match.group(1)}】"
        return f"【{wid}】"
    
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

    def _record_workflow_timing(self, workflow_id, phase, duration_sec, call_label=None, extra=None):
        """记录工作流级耗时统计，确保并行场景下不串数据。"""
        if not workflow_id or workflow_id not in self.workflows:
            return

        duration_value = max(0.0, float(duration_sec or 0.0))
        with self.workflows_lock:
            wf = self.workflows.get(workflow_id)
            if not wf:
                return

            if phase == "llm":
                wf['llm_total_duration_sec'] = float(wf.get('llm_total_duration_sec', 0.0)) + duration_value
                wf.setdefault('llm_call_records', []).append({
                    'call_label': call_label or f"llm_call_{len(wf.get('llm_call_records', [])) + 1}",
                    'duration_sec': duration_value,
                    'extra': extra or {}
                })
            elif phase == "simulation":
                wf['sim_total_duration_sec'] = float(wf.get('sim_total_duration_sec', 0.0)) + duration_value
                wf.setdefault('sim_call_records', []).append({
                    'call_label': call_label or f"simulation_call_{len(wf.get('sim_call_records', [])) + 1}",
                    'duration_sec': duration_value,
                    'extra': extra or {}
                })
    
    def run_simulation(self, idf_path, iteration_name, workflow_id=None):
        """运行EnergyPlus模拟"""
        self.logger.info(f"\n【模拟】{iteration_name}")
        self.logger.info("-" * 80)

        resolved_workflow_id = workflow_id or getattr(self._log_context, 'workflow_id', None)
        sim_start = perf_counter()
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
        finally:
            sim_elapsed = perf_counter() - sim_start
            self._record_workflow_timing(
                resolved_workflow_id,
                "simulation",
                sim_elapsed,
                call_label=iteration_name,
                extra={
                    'idf_path': idf_path
                }
            )
            if resolved_workflow_id:
                self.logger.info(
                    f"【计时】[{resolved_workflow_id}] 模拟 {iteration_name} 耗时: {sim_elapsed:.2f}s"
                )
    
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
        self.current_iteration = iteration
        
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
        
        self._log_plan_reasoning(
            stage="最终采用方案",
            plan=plan,
            level="info"
        )
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
        """动态生成优化方向提示（去除固定对象-字段绑定，鼓励LLM自由组合）。"""
        has_heating_cooling_conflict = (
            delta_cooling is not None and delta_heating is not None and delta_cooling > 0 and delta_heating > 0
        )

        direction_atoms = [
            {
                "name": "内热源控制",
                "keywords": ["PEOPLE", "LIGHTS", "ELECTRICEQUIPMENT", "WATTS", "DENSITY"],
                "description": "可在人员、照明、设备等内部热源字段中灵活选取，组合降低冷负荷并兼顾全年能耗",
                "tags": ["balanced"]
            },
            {
                "name": "围护保温",
                "keywords": ["MATERIAL", "CONDUCTIVITY", "RESISTANCE", "UFACTOR"],
                "description": "可从墙体/屋面/窗体相关热工字段中自由组合，提升保温隔热性能",
                "tags": ["balanced"]
            },
            {
                "name": "渗透与通风",
                "keywords": ["INFILTRATION", "OUTDOORAIR", "VENTILATION", "FLOW"],
                "description": "可在渗透和新风相关字段中灵活选取，并根据目标负荷动态调节",
                "tags": ["balanced"]
            },
            {
                "name": "HVAC运行效率",
                "keywords": ["HEATRECOVERY", "EFFECTIVENESS", "COP", "EER", "TEMPERATURE", "HVAC"],
                "description": "可组合供风温度、热回收、效率参数等字段，避免仅依赖单一温度字段",
                "tags": ["balanced"]
            },
            {
                "name": "窗体光学参数",
                "keywords": ["WINDOW", "GLAZING", "SHGC", "SOLAR", "VISIBLE", "EMISSIVITY"],
                "description": "可在太阳透射、反射、可见光及辐射参数间灵活配对，兼顾夏冬季负荷",
                "tags": ["seasonal"]
            },
            {
                "name": "运行设定与调度",
                "keywords": ["SCHEDULE", "SETPOINT", "CONTROL", "AVAILABILITY"],
                "description": "可结合设定点与时段调度字段组合优化，不局限固定对象",
                "tags": ["balanced"]
            },
        ]

        # 基于历史实际修改字段统计“方向使用频率”，优先推荐低频方向。
        usage_scores = {atom["name"]: 0 for atom in direction_atoms}
        for field_key, count in (self.field_modification_history or {}).items():
            key_upper = str(field_key).upper()
            for atom in direction_atoms:
                if any(token in key_upper for token in atom["keywords"]):
                    usage_scores[atom["name"]] += int(count)

        # 冲突场景下仅“弱偏好”balanced方向，不做硬禁止，保持灵活组合。
        def _rank(atom):
            usage = usage_scores.get(atom["name"], 0)
            balanced_penalty = 0 if (not has_heating_cooling_conflict or "balanced" in atom.get("tags", [])) else 1
            return (balanced_penalty, usage, atom["name"])

        sorted_atoms = sorted(direction_atoms, key=_rank)
        focus_count = min(4, len(sorted_atoms))
        focus_atoms = sorted_atoms[:focus_count]

        if has_heating_cooling_conflict:
            directions_text = """【建议优化方向（冲突负荷场景）】
当前供暖与制冷负荷均需改善。
请优先选择对全年能耗更稳健的字段组合，但不要被固定模板束缚。
"""
        else:
            directions_text = "【建议优化方向】本轮可从以下方向灵活组合字段：\n"

        directions_text += "- 说明：以下方向仅作引导，不做对象-字段死绑定；请基于候选对象字段自由排列组合。\n"
        for atom in focus_atoms:
            usage = usage_scores.get(atom["name"], 0)
            usage_tag = "低频优先" if usage == 0 else f"历史使用{usage}次"
            sample_tokens = "、".join(atom["keywords"][:4])
            directions_text += f"  • {atom['name']}（{usage_tag}，关键词示例：{sample_tokens}）：{atom['description']}\n"

        if usage_scores:
            most_used_name, most_used_count = max(usage_scores.items(), key=lambda x: x[1])
            if most_used_count >= 3:
                directions_text += f"\n⚠️ 历史上“{most_used_name}”使用较多（{most_used_count}次），本轮可适当扩展到其他方向。"

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

        candidates_count = len(kb_context.get('candidates', [])) if isinstance(kb_context, dict) else 0
        min_categories = min(4, max(3, candidates_count)) if candidates_count > 0 else 4
        max_modifications = min(6, max(4, candidates_count if candidates_count > 0 else 6))
        min_modifications = max(4, min(5, max_modifications))

        # 每个工作流“第一轮优化建议（含其全部重试）”不启用低频/高频约束
        has_iteration_history = len(getattr(self, 'iteration_history', []) or []) > 1
        has_field_history = bool(getattr(self, 'field_modification_history', {})) or bool(getattr(self, 'last_round_fields', set()))
        enable_novelty_constraints = has_iteration_history and has_field_history
        base_novelty_directive = self._build_novelty_directive(None) if enable_novelty_constraints else ""
        
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
{('优先避免重复上轮字段及历史高频字段，尽量采用历史低频或未出现的对象字段组合。' if enable_novelty_constraints else '首轮优化阶段请优先保证方案可执行与物理合理，不强制低频/高频约束。')}

{base_novelty_directive}

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

3. **从至少{min_categories}个不同类别中选择优化方向（强制要求）**
   - 不允许只优化一个类别（如只改Sizing:Zone）
   - 必须涉及多个维度（HVAC + 围护 + 内热源等）
    - 温度相关场景默认优先考虑“设计供风温度”（Sizing:Zone）
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
        
        try:
            rejected_reasoning_seen = set()
            plan = self._request_plan_from_llm(system_prompt, user_prompt)
            if not plan:
                return None

            max_repair_attempts = max(2, min(3, len(kb_context.get('candidates', [])) // 3 if isinstance(kb_context, dict) else 2))

            has_history_for_novelty = bool(getattr(self, 'field_modification_history', {})) or bool(getattr(self, 'last_round_fields', set()))

            # 每个工作流首轮优化（含重试）不做低频补充，避免把探索阶段过早约束
            if enable_novelty_constraints and has_history_for_novelty:
                initial_augmented_plan = self._force_plan_diversity(
                    plan,
                    kb_context,
                    target_min_categories=min_categories,
                    enforce_novelty=True
                )
                if initial_augmented_plan:
                    plan = initial_augmented_plan

            category_count = self._count_plan_categories(plan)
            category_attempt = 0
            while category_count < min_categories and category_attempt < max_repair_attempts:
                category_attempt += 1
                start_reason = f"当前覆盖{category_count}/{min_categories}，对象类别不足"
                self._log_repair_step("类别覆盖校验", category_attempt, "start", reason=start_reason)
                self._log_rejected_plan_reasoning(
                    "类别覆盖触发纠偏",
                    plan,
                    rejected_reasoning_seen,
                    reason=f"对象类别覆盖不足：当前{category_count}，要求至少{min_categories}",
                    attempt_label=f"第{category_attempt}次纠偏",
                    include_reasoning=True,
                    include_reason=False,
                    stage_suffix=f"第{category_attempt}次纠偏前失败候选（类别覆盖）"
                )
                novelty_directive = self._build_novelty_directive(plan) if enable_novelty_constraints else ""
                anti_repeat_directive = self._build_retry_anti_repeat_directive(plan)
                repair_prompt = user_prompt + f"""

【纠偏重生成（必须遵守）】
你上一版仅覆盖{category_count}个优化方面，未达到要求。
请重新生成JSON，并满足：
1) modifications 至少覆盖{min_categories}个不同对象类别（优化方面）
2) {('必须优先使用“过去较少出现或未出现”的对象+字段组合，减少与上一轮重复' if enable_novelty_constraints else '优先扩大对象与字段组合的覆盖，不必受低频/高频历史约束')}
3) 禁止重复沿用本轮已经过度重复的对象字段组合
4) 若提到“阈值/上下限/限制/最大/最小”，可修改IdealLoads阈值字段；否则优先修改Sizing:Zone设计温度字段
5) 不得修改特殊关键字字段（autosize/autocalculate等）

{novelty_directive}
{anti_repeat_directive}
"""
                repaired_plan = self._request_plan_from_llm(system_prompt, repair_prompt, temperature=min(0.35, 0.12 + 0.06 * category_attempt))
                if repaired_plan:
                    self._log_repair_step("类别覆盖校验", category_attempt, "end", "已收到LLM返回")
                    repaired_augmented_plan = self._force_plan_diversity(
                        repaired_plan,
                        kb_context,
                        target_min_categories=min_categories,
                        enforce_novelty=True
                    ) if enable_novelty_constraints else None
                    candidate_plan = repaired_augmented_plan if repaired_augmented_plan else repaired_plan
                    repaired_count = self._count_plan_categories(candidate_plan)
                    if repaired_count >= category_count:
                        plan = candidate_plan
                        category_count = repaired_count
                        self._log_repair_step("类别覆盖校验", category_attempt, "passed", result=f"覆盖提升至{repaired_count}/{min_categories}")
                    else:
                        self._log_repair_step("类别覆盖校验", category_attempt, "failed", reason=f"重生成后覆盖{repaired_count}/{min_categories}")
                        self._log_rejected_plan_reasoning(
                            "类别覆盖纠偏后仍不足",
                            candidate_plan,
                            rejected_reasoning_seen,
                            reason=f"重生成后类别仍不足：当前{repaired_count}，目标至少{min_categories}",
                            attempt_label=f"第{category_attempt}次纠偏",
                            stage_suffix=f"第{category_attempt}次纠偏返回候选（类别覆盖）",
                            include_reason=False
                        )
                else:
                    self._log_repair_step("类别覆盖校验", category_attempt, "end", "LLM未返回有效方案")

            if category_count < min_categories:
                self.logger.warning(
                    f"LLM纠偏后类别覆盖仍不足（{category_count}/{min_categories}），启动自动字段补全提升覆盖度"
                )
                forced_category_plan = self._force_plan_diversity(
                    plan,
                    kb_context,
                    target_min_categories=min_categories,
                    enforce_novelty=True
                )
                if forced_category_plan:
                    forced_category_count = self._count_plan_categories(forced_category_plan)
                    if forced_category_count > category_count:
                        plan = forced_category_plan
                        category_count = forced_category_count
                        self.logger.info(f"✓ 自动补全后类别覆盖提升至{category_count}个优化方面")

            if category_count < min_categories:
                self.logger.warning(
                    f"最终类别覆盖仍为{category_count}/{min_categories}。已达到纠偏与补全上限，将使用当前最优可行方案继续运行"
                )

            temp_issue = self._get_temperature_mapping_issue(plan, user_request)
            if temp_issue:
                self.logger.warning(f"温度字段映射检查未通过：{temp_issue}，触发一次纠偏重生成")
                self._log_repair_step("温度映射校验", 1, "start", reason=temp_issue)
                self._log_rejected_plan_reasoning(
                    "温度映射触发纠偏",
                    plan,
                    rejected_reasoning_seen,
                    reason=temp_issue,
                    attempt_label="第1次纠偏",
                    include_reasoning=True,
                    include_reason=False,
                    stage_suffix="第1次纠偏前失败候选（温度映射）"
                )
                novelty_directive = self._build_novelty_directive(plan) if enable_novelty_constraints else ""
                anti_repeat_directive = self._build_retry_anti_repeat_directive(plan)
                temp_repair_prompt = user_prompt + f"""

【温度映射纠偏（必须遵守）】
{temp_issue}

规则：
1) 如果文本没有“阈值/上下限/限制/最大/最小”等语义，禁止修改 IdealLoads 的
   Minimum_Cooling_Supply_Air_Temperature / Maximum_Heating_Supply_Air_Temperature。
2) 此时应优先修改 Sizing:Zone 的
   Zone_Cooling_Design_Supply_Air_Temperature / Zone_Heating_Design_Supply_Air_Temperature。
3) 如果文本出现阈值语义，允许修改 IdealLoads 阈值字段；并鼓励同时保留设计温度优化。
4) {('必须优先采用历史低频/未出现字段，避免重复上一轮字段。' if enable_novelty_constraints else '可自由组合对象字段，优先保证物理合理与可执行。')}

{novelty_directive}
{anti_repeat_directive}
请重新生成严格JSON。
"""
                temp_repaired_plan = self._request_plan_from_llm(system_prompt, temp_repair_prompt, temperature=0.18)
                if temp_repaired_plan:
                    self._log_repair_step("温度映射校验", 1, "end", "已收到LLM返回")
                    repaired_issue = self._get_temperature_mapping_issue(temp_repaired_plan, user_request)
                    if not repaired_issue:
                        plan = temp_repaired_plan
                        self._log_repair_step("温度映射校验", 1, "passed", result="返回方案满足映射规则")
                    else:
                        self._log_repair_step("温度映射校验", 1, "failed", reason=repaired_issue)
                        self._log_rejected_plan_reasoning(
                            "温度映射纠偏后仍不满足",
                            temp_repaired_plan,
                            rejected_reasoning_seen,
                            reason=repaired_issue,
                            attempt_label="第1次纠偏",
                            stage_suffix="第1次纠偏返回候选（温度映射）",
                            include_reason=False
                        )
                        self.logger.warning(f"温度字段映射纠偏后仍未完全满足：{repaired_issue}，将继续执行当前最优可行方案")
                else:
                    self._log_repair_step("温度映射校验", 1, "end", "LLM未返回有效方案")
            
            # ========== 字段多样性检查 ==========
            is_diverse, diversity_issue = self._check_field_diversity(plan)
            diversity_attempt = 0
            while not is_diverse and diversity_issue and diversity_attempt < max_repair_attempts:
                diversity_attempt += 1
                self._log_repair_step("字段多样性校验", diversity_attempt, "start", reason=diversity_issue)
                self._log_rejected_plan_reasoning(
                    "字段多样性触发纠偏",
                    plan,
                    rejected_reasoning_seen,
                    reason=diversity_issue,
                    attempt_label=f"第{diversity_attempt}次纠偏",
                    include_reasoning=True,
                    include_reason=False,
                    stage_suffix=f"第{diversity_attempt}次纠偏前失败候选（字段多样性）"
                )
                novelty_directive = self._build_novelty_directive(plan) if enable_novelty_constraints else ""
                anti_repeat_directive = self._build_retry_anti_repeat_directive(plan)
                diversity_repair_prompt = user_prompt + f"""

【字段多样性纠偏（必须遵守）】
{diversity_issue}

要求：
1) 避免过度集中于少数高频字段，应优先选择历史中出现次数更少的对象字段
2) 与上一轮重复字段比例必须<=50%，尽量使用上轮未出现字段
3) 至少覆盖多个对象类别（建议>=3）
4) 参考【历史字段修改频率统计】，优先使用低频或未出现字段，避免高频字段
5) 不得复用当前这版中已判定重复过高的对象字段组合

{novelty_directive}
{anti_repeat_directive}
请重新生成包含更多样化字段的JSON方案。
"""
                diversity_repaired_plan = self._request_plan_from_llm(system_prompt, diversity_repair_prompt, temperature=min(0.4, 0.16 + 0.08 * diversity_attempt))
                if diversity_repaired_plan:
                    self._log_repair_step("字段多样性校验", diversity_attempt, "end", "已收到LLM返回")
                    repaired_augmented_plan = self._force_plan_diversity(
                        diversity_repaired_plan,
                        kb_context,
                        target_min_categories=min_categories,
                        enforce_novelty=True
                    ) if enable_novelty_constraints else None
                    candidate_plan = repaired_augmented_plan if repaired_augmented_plan else diversity_repaired_plan
                    repaired_is_diverse, repaired_issue = self._check_field_diversity(candidate_plan)
                    if repaired_is_diverse:
                        plan = candidate_plan
                        self._log_repair_step("字段多样性校验", diversity_attempt, "passed", result="返回方案通过多样性校验")
                        self.logger.info("✓ 多样性纠偏成功，已采用新方案")
                        is_diverse = True
                        diversity_issue = None
                    else:
                        self._log_repair_step("字段多样性校验", diversity_attempt, "failed", reason=repaired_issue)
                        self._log_rejected_plan_reasoning(
                            "多样性纠偏后仍不满足",
                            candidate_plan,
                            rejected_reasoning_seen,
                            reason=repaired_issue,
                            attempt_label=f"第{diversity_attempt}次纠偏",
                            stage_suffix=f"第{diversity_attempt}次纠偏返回候选（字段多样性）",
                            include_reason=False
                        )
                        plan = candidate_plan
                        is_diverse = False
                        diversity_issue = repaired_issue
                else:
                    self._log_repair_step("字段多样性校验", diversity_attempt, "end", "LLM未返回有效方案")

            if not is_diverse and diversity_issue:
                # 多样性纠偏LLM仍未完全满足时，执行系统自动多样性重排
                self.logger.warning("多样性纠偏后仍未完全满足，启动自动多样性重排补救")
                diversity_forced_plan = self._force_plan_diversity(
                    plan,
                    kb_context,
                    target_min_categories=min_categories,
                    enforce_novelty=True
                ) if enable_novelty_constraints else None
                if diversity_forced_plan:
                    forced_is_diverse, forced_issue = self._check_field_diversity(diversity_forced_plan)
                    if forced_is_diverse:
                        plan = diversity_forced_plan
                        self.logger.info("✓ 自动多样性重排已完成，已采用重排方案")
                    else:
                        if diversity_forced_plan:
                            plan = diversity_forced_plan
                        self.logger.warning(f"自动多样性重排后仍未完全满足：{forced_issue}，将继续执行当前最优可行方案")

            return plan
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return None

    def _force_plan_diversity(self, origin_plan, kb_context, target_min_categories=4, enforce_novelty=True):
        """强制执行多样性重排：当LLM纠偏仍失败时的兜底方案。
        
        策略：
        1. 收集原方案中已使用字段，标记为已占用
        2. 从知识库候选对象中补充“历史低频/未出现”的对象字段
        3. 尽可能将对象类别覆盖补齐到 target_min_categories
        4. 使用保守且物理合理的表达式（避免反优化）
        """
        if not isinstance(origin_plan, dict):
            return None
        
        try:
            used_fields = set()
            obj_field_count = {}
            
            for mod in origin_plan.get('modifications', []):
                obj_type = str(mod.get('object_type', '')).upper().strip()
                for field_name in mod.get('fields', {}).keys():
                    used_fields.add(f"{obj_type}.{str(field_name).upper().strip()}")
                    obj_field_count[obj_type] = obj_field_count.get(obj_type, 0) + 1
            
            origin_categories = set(obj_field_count.keys())
            origin_fields = self._extract_plan_field_keys(origin_plan)
            
            forced_plan = {
                "reasoning": origin_plan.get("reasoning", ""),
                "confidence": origin_plan.get("confidence", "medium"),
                "modifications": list(origin_plan.get("modifications", []))
            }

            candidates = []
            if isinstance(kb_context, dict):
                candidates = kb_context.get('candidates', []) or []

            field_history = {
                str(k).upper(): int(v)
                for k, v in self.field_modification_history.items()
            } if hasattr(self, 'field_modification_history') else {}
            last_round_fields = set(getattr(self, 'last_round_fields', set()) or set())

            # 收集可补充候选（仅数值且非特殊关键字）
            supplement_pool = []
            for cand in candidates:
                obj_type = str(cand.get('object_type', '')).upper().strip()
                if not obj_type:
                    continue
                for field_info in cand.get('field_semantics', []):
                    field_name = str(field_info.get('field_name', '')).strip()
                    if not field_name:
                        continue
                    current_val = cand.get('current_sample_values', {}).get(field_name, '')
                    if self._is_special_value(current_val) or (not self._is_numeric_value(current_val)):
                        continue

                    field_key = f"{obj_type}.{field_name.upper()}"
                    if field_key in used_fields:
                        continue

                    freq = field_history.get(field_key, 0)
                    is_new_object = 0 if obj_type in origin_categories else 1
                    is_unseen_field = 1 if field_key not in field_history else 0
                    not_in_last_round = 1 if field_key not in last_round_fields else 0

                    # 按“历史出现少 + 本轮未用 + 上轮未用 + 新对象类别”综合排序
                    sort_key = (
                        -is_unseen_field,
                        -not_in_last_round,
                        -is_new_object,
                        freq,
                        obj_type,
                        field_name.upper()
                    )
                    supplement_pool.append((sort_key, obj_type, field_name, field_key))

            supplement_pool.sort(key=lambda x: x[0])

            target_new_categories = max(0, int(target_min_categories) - len(origin_categories))
            max_additions = max(2, min(10, target_new_categories + 4))

            # 若与上一轮重复率偏高，动态增加补充项以提升通过概率
            if enforce_novelty and last_round_fields and origin_fields:
                overlap_count = len(origin_fields & last_round_fields)
                # 目标：overlap / total <= 0.5 => total >= 2 * overlap
                need_total = overlap_count * 2
                need_extra = max(0, need_total - len(origin_fields))
                if need_extra > 0:
                    max_additions = min(12, max(max_additions, need_extra + 1))

            # 灵活策略：以低频/未出现字段为主，但允许少量历史字段保留稳定性
            novelty_quota = max(1, int(max_additions * 0.7))
            novelty_added = 0

            added_count = 0
            covered_categories = set(origin_categories)
            for _, target_type, field_name, field_key in supplement_pool:
                if added_count >= max_additions:
                    break

                # enforce_novelty=True 时，优先跳过上轮字段；若池子太小则后续自然会补到
                if (
                    enforce_novelty
                    and field_key in last_round_fields
                    and len(supplement_pool) > max_additions
                    and novelty_added < novelty_quota
                ):
                    continue

                expr = self._build_balanced_expression_for_field(target_type, field_name)
                forced_plan['modifications'].append({
                    "object_type": target_type,
                    "name_filter": None,
                    "apply_to_all": True,
                    "reasoning": f"补充{target_type}的{field_name}优化以提升多样性与覆盖度",
                    "fields": {field_name: expr}
                })
                used_fields.add(field_key)
                covered_categories.add(target_type)
                added_count += 1
                if field_key not in last_round_fields:
                    novelty_added += 1

                if len(covered_categories) >= int(target_min_categories) and added_count >= 2:
                    break

            if added_count == 0:
                # 这是可预期场景：候选可能已被用尽或均不可修改，不应视为错误告警
                total_candidates = len(candidates)
                self.logger.info(
                    f"自动多样性重排未补充新字段：候选对象数={total_candidates}，"
                    f"可能原因包括字段已使用、字段非数值或字段为特殊关键字（如autosize）"
                )
                return None

            if len(covered_categories) > len(origin_categories):
                self.logger.info(
                    f"自动多样性重排：补充了{added_count}项字段（低频优先，含少量稳定字段）；对象类别覆盖由{len(origin_categories)}提升到{len(covered_categories)}"
                )
                if len(origin_categories) == 0:
                    self.logger.info("说明：原方案缺少有效对象类别（通常是modifications为空或字段无效），重排后已补入可执行对象类别")
            else:
                self.logger.info(
                    f"自动多样性重排：补充了{added_count}项字段（低频优先，含少量稳定字段）；对象类别保持{len(covered_categories)}（本次主要用于降低字段重复）"
                )
            return self._deduplicate_plan_modifications(forced_plan, context_tag="自动多样性重排")
        
        except Exception as e:
            self.logger.error(f"自动多样性重排失败: {e}")
            return None

    def _log_rejected_plan_reasoning(
        self,
        title,
        candidate_plan,
        dedup_cache=None,
        reason=None,
        attempt_label=None,
        include_reasoning=True,
        stage_suffix=None,
        include_reason=True
    ):
        """在告警场景打印不满足要求方案的原因；可按需附带推理文本。"""
        reasoning = ""
        if isinstance(candidate_plan, dict):
            reasoning = str(candidate_plan.get("reasoning", "") or "").strip()

        field_keys = self._extract_plan_field_keys(candidate_plan)
        fingerprint = f"{reasoning}|{'|'.join(sorted(field_keys))}"
        if dedup_cache is not None:
            if fingerprint in dedup_cache:
                return
            dedup_cache.add(fingerprint)

        if include_reason:
            reason_text = str(reason or "未提供失败原因").strip()
            label_prefix = f"[{attempt_label}] " if attempt_label else ""
            self.logger.warning(f"【{title}】{label_prefix}原因：{reason_text}")

        if include_reasoning:
            stage = stage_suffix if stage_suffix else f"不可取候选方案{f' - {attempt_label}' if attempt_label else ''}"
            self._log_plan_reasoning(
                stage=stage,
                plan=candidate_plan,
                level="warning"
            )

    def _log_repair_step(self, check_name, attempt, phase, result=None, reason=None):
        """统一记录纠偏步骤，保证“启动-结束-结果”语义一致。"""
        if phase == "start":
            tail = f"：{reason}" if reason else ""
            self.logger.info(f"【{check_name}】启动第{attempt}次纠偏{tail}")
            return

        if phase == "failed":
            tail = f"：{reason}" if reason else ""
            self.logger.warning(f"【{check_name}】第{attempt}次纠偏结果未通过{tail}")
            return

        if phase == "passed":
            tail = f"：{result}" if result else ""
            self.logger.info(f"【{check_name}】第{attempt}次纠偏结果通过{tail}")
            return

        tail = f"：{result}" if result else ""
        self.logger.info(f"【{check_name}】第{attempt}次纠偏结束{tail}")

    def _format_reasoning_numbered(self, reasoning_text):
        """把长推理文本格式化为序号列表，便于阅读与排查。"""
        text = str(reasoning_text or "").strip()
        if not text:
            return "(无推理文本)"

        raw_parts = re.split(r"[\n\r]+|[。；;]", text)
        parts = []
        for part in raw_parts:
            p = str(part).strip()
            if not p:
                continue
            p = re.sub(r"^\d+[\.|、|\)]\s*", "", p)
            parts.append(p)

        if not parts:
            return text

        def _ensure_period(s):
            """若句末没有标点，补充句号。"""
            if s and s[-1] not in "。！？!?":
                return s + "。"
            return s

        return "\n".join([f"{idx}. {_ensure_period(item)}" for idx, item in enumerate(parts, start=1)])

    def _log_plan_reasoning(self, stage, plan, level="info"):
        """统一输出推理日志：标识来源阶段，并用序号格式化。"""
        plan = self._deduplicate_plan_modifications(plan, context_tag=f"日志:{stage}")

        reasoning = ""
        if isinstance(plan, dict):
            reasoning = str(plan.get("reasoning", "") or "").strip()

        # 在输出推理前，计算并记录建议的丰富度与相似度指标，便于直观对比
        try:
            mods = []
            if isinstance(plan, dict):
                mods = plan.get('modifications', []) or []
            unique_fields = set()
            for m in mods:
                obj_type = str(m.get('object_type', '')).upper().strip()
                fields = m.get('fields', {}) if isinstance(m.get('fields', {}), dict) else {}
                for f_name in fields.keys():
                    unique_fields.add(f"{obj_type}.{str(f_name).upper().strip()}")

            # 丰富度（统一口径）：对象.字段 去重后的字段数量
            num_unique_fields = len(unique_fields)
            # 对象类别数
            num_categories = len({str(m.get('object_type', '')).upper().strip() for m in mods if m.get('object_type')})

            # 相似度（统一口径）：
            # 分子=与上一轮重复的对象.字段数量（去重）
            # 分母=本轮对象.字段数量（去重）
            try:
                last_round = set([f.upper() for f in (getattr(self, 'last_round_fields', set()) or set())])
            except Exception:
                last_round = set()
            overlap_ratio = 0.0
            overlap_count = 0
            if num_unique_fields > 0 and last_round:
                overlap_count = len(unique_fields & last_round)
                overlap_ratio = overlap_count / float(num_unique_fields)

            metrics_line = (
                f"【建议指标】丰富度: 字段类别 {num_unique_fields}项（对象.字段去重）；"
                f"对象类别: {num_categories}个；相似度: 与上一轮字段重复率 {overlap_ratio:.1%} ({overlap_count}/{num_unique_fields})"
            )
            # 先以 info 级别打印指标，确保可见性（后续的reasoning仍按level输出）
            self.logger.info(metrics_line)
        except Exception as _e:
            self.logger.debug(f"计算建议指标失败: {_e}")

        header = f"【LLM推理过程 - {stage}】"
        body = self._format_reasoning_numbered(reasoning)

        if str(level).lower() == "warning":
            self.logger.warning(f"{header}\n{body}")
        else:
            self.logger.info(f"{header}\n{body}")

    def _extract_plan_field_keys(self, plan):
        """提取计划中的对象.字段键集合，用于去重与提示增强。"""
        keys = set()
        if not isinstance(plan, dict):
            return keys

        for mod in plan.get("modifications", []):
            obj_type = str(mod.get("object_type", "")).upper().strip()
            fields = mod.get("fields", {})
            if not obj_type or not isinstance(fields, dict):
                continue
            for field_name in fields.keys():
                keys.add(f"{obj_type}.{str(field_name).upper().strip()}")

        return keys

    def _deduplicate_plan_modifications(self, plan, context_tag=""):
        """按对象分组合并并去重重复字段，保证同轮建议字段唯一。"""
        if not isinstance(plan, dict):
            return plan

        modifications = plan.get("modifications", [])
        if not isinstance(modifications, list) or not modifications:
            return plan

        bucket_order = []
        bucket_map = {}
        total_before = 0

        for mod in modifications:
            if not isinstance(mod, dict):
                continue
            object_type = str(mod.get("object_type", "")).strip()
            if not object_type:
                continue

            name_filter = mod.get("name_filter", None)
            apply_to_all = bool(mod.get("apply_to_all", True))
            bucket_key = (
                object_type.upper(),
                str(name_filter).upper() if name_filter is not None else None,
                apply_to_all,
            )

            if bucket_key not in bucket_map:
                bucket_map[bucket_key] = {
                    "object_type": object_type,
                    "name_filter": name_filter,
                    "apply_to_all": apply_to_all,
                    "fields": {},
                }
                bucket_order.append(bucket_key)

            fields = mod.get("fields", {})
            if not isinstance(fields, dict):
                continue

            for field_name, expr in fields.items():
                total_before += 1
                field_norm = str(field_name).upper().strip()
                # 后出现的同字段覆盖前者，避免重复建议
                bucket_map[bucket_key]["fields"][field_norm] = (str(field_name).strip(), expr)

        merged_mods = []
        total_after = 0
        for bucket_key in bucket_order:
            bucket = bucket_map[bucket_key]
            raw_fields = bucket.pop("fields", {})
            if not raw_fields:
                continue

            bucket["fields"] = {orig_name: expr for _, (orig_name, expr) in raw_fields.items()}
            total_after += len(bucket["fields"])
            merged_mods.append(bucket)

        if not merged_mods:
            return plan

        new_plan = dict(plan)
        new_plan["modifications"] = merged_mods

        removed = max(0, total_before - total_after)
        if removed > 0:
            tag = f"[{context_tag}] " if context_tag else ""
            self.logger.info(f"{tag}建议去重：移除{removed}项重复字段，保留{total_after}项字段")

        return new_plan

    def _build_novelty_directive(self, rejected_plan=None, max_items=10):
        """构建给LLM的低频/未出现字段约束，减少重复提案。"""
        lines = ["【低频与新颖性硬约束（必须遵守）】"]

        last_round = sorted(list(getattr(self, "last_round_fields", set()) or set()))
        if last_round:
            lines.append("1) 以下为上一轮已使用字段，除非必要请避免复用：")
            for key in last_round[:max_items]:
                lines.append(f"   - {key}")

        if self.field_modification_history:
            sorted_freq = sorted(self.field_modification_history.items(), key=lambda x: x[1], reverse=True)
            high_freq = [str(k).upper() for k, _ in sorted_freq[:max_items]]
            if high_freq:
                lines.append("2) 以下为历史高频字段，本轮应尽量避免：")
                for key in high_freq:
                    lines.append(f"   - {key}")

        rejected_keys = sorted(list(self._extract_plan_field_keys(rejected_plan))) if rejected_plan else []
        if rejected_keys:
            lines.append("3) 以下为当前被判定不合格方案涉及字段，重生成时应优先避免原样重复（可保留少量已验证字段）：")
            for key in rejected_keys[:max_items]:
                lines.append(f"   - {key}")

        lines.append("4) 请优先从候选中选择历史低频或未出现字段，并尽量与上一轮至少50%不同。")
        lines.append("5) 若可行字段不足，可保留少量高价值历史字段，同时扩大对象类型覆盖，避免机械重复。")
        return "\n".join(lines)

    def _build_retry_anti_repeat_directive(self, rejected_plan=None, max_items=12):
        """构建重试时的反重复约束，明确禁止原样返回同一字段集合。"""
        lines = ["【重试反重复约束（必须遵守）】"]

        previous_fields = sorted(list(getattr(self, "last_round_fields", set()) or set()))
        if previous_fields:
            lines.append("1) 以下是上一轮已实际修改字段，本轮与其重合比例必须<=50%，禁止原样照搬：")
            for key in previous_fields[:max_items]:
                lines.append(f"   - {key}")

        rejected_fields = sorted(list(self._extract_plan_field_keys(rejected_plan))) if rejected_plan else []
        if rejected_fields:
            lines.append("2) 以下是刚刚被判失败的字段集合，下一次重试不得与其完全相同：")
            for key in rejected_fields[:max_items]:
                lines.append(f"   - {key}")

        lines.append("3) 若仍选择上述部分字段，必须同时加入足够多的新字段，使总重合比例降到50%以下。")
        lines.append("4) 不要只改措辞；reasoning 和 modifications 都必须体现新的字段组合。")
        return "\n".join(lines)

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

    def _request_plan_from_llm(self, system_prompt, user_prompt, temperature=0.0):
        """统一的LLM请求入口，负责调用与token日志和详细token统计。"""
        with self.workflows_lock:
            self.llm_calls_count += 1
            call_index = self.llm_calls_count

        workflow_id = getattr(self._log_context, 'workflow_id', None) or getattr(self, 'current_workflow_id', None)
        iteration_in_context = getattr(self._log_context, 'iteration', None)
        llm_start = perf_counter()
        self.logger.info(f"\n🤖 【调用LLM】第{call_index}次调用 - 模型: gpt-5.4")

        try:
            response = self.client.chat.completions.create(
                model="gpt-5.4",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            plan = json.loads(response.choices[0].message.content)
            plan = self._deduplicate_plan_modifications(plan, context_tag=f"LLM返回#{call_index}")
            usage = response.usage
        finally:
            llm_elapsed = perf_counter() - llm_start
            self._record_workflow_timing(
                workflow_id,
                "llm",
                llm_elapsed,
                call_label=f"llm_call_{call_index}",
                extra={
                    'global_call_index': call_index,
                    'temperature': temperature,
                    'iteration': iteration_in_context
                }
            )
            if workflow_id:
                iter_label = f"第{iteration_in_context}轮" if iteration_in_context else "未知轮次"
                self.logger.info(
                    f"【计时】[{workflow_id}] LLM调用 #{call_index} ({iter_label}) 耗时: {llm_elapsed:.2f}s"
                )

        # 详细token统计
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        cached_input_tokens = getattr(usage, 'cached_prompt_tokens', 0) if hasattr(usage, 'cached_prompt_tokens') else 0
        total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)

        # 分别累计
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_input_tokens += cached_input_tokens
        self.total_tokens_used = getattr(self, 'total_tokens_used', 0) + total_tokens

        # 记录到对应工作流，便于按轮次导出 token 明细
        if workflow_id and workflow_id in self.workflows:
            with self.workflows_lock:
                wf = self.workflows.get(workflow_id)
                if wf is not None:
                    wf.setdefault('llm_token_records', []).append({
                        'call_label': f"llm_call_{call_index}",
                        'iteration': iteration_in_context,
                        'input_tokens': int(input_tokens or 0),
                        'output_tokens': int(output_tokens or 0),
                        'cached_input_tokens': int(cached_input_tokens or 0),
                        'total_tokens': int(total_tokens or 0),
                        'temperature': temperature,
                    })

        self.logger.info("✓ LLM分析完成")
        self.logger.info(f"  - Input Tokens: {input_tokens:,}")
        self.logger.info(f"  - Output Tokens: {output_tokens:,}")
        self.logger.info(f"  - Cached Input Tokens: {cached_input_tokens:,}")
        self.logger.info(f"  - Temperature: {temperature}")
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

    def _update_suggestion_frequency_for_workflow(self, workflow_id, plan):
        """统计每轮最终执行建议中的对象/字段出现频率（按工作流独立记录）。"""
        if not workflow_id or workflow_id not in self.workflows:
            return
        if not isinstance(plan, dict):
            return

        modifications = plan.get('modifications', []) or []
        if not isinstance(modifications, list) or not modifications:
            return

        with self.workflows_lock:
            wf = self.workflows.get(workflow_id)
            if not wf:
                return

            obj_freq = wf.get('suggestion_object_frequency')
            field_freq = wf.get('suggestion_field_frequency')
            if not isinstance(obj_freq, dict):
                obj_freq = {}
            if not isinstance(field_freq, dict):
                field_freq = {}

            for mod in modifications:
                if not isinstance(mod, dict):
                    continue
                obj_type = str(mod.get('object_type', '')).strip()
                if not obj_type:
                    continue
                obj_key = obj_type.upper()
                obj_freq[obj_key] = int(obj_freq.get(obj_key, 0)) + 1

                fields = mod.get('fields', {})
                if not isinstance(fields, dict):
                    continue
                for field_name in fields.keys():
                    field_key = f"{obj_key}.{str(field_name).strip().upper()}"
                    field_freq[field_key] = int(field_freq.get(field_key, 0)) + 1

            wf['suggestion_object_frequency'] = obj_freq
            wf['suggestion_field_frequency'] = field_freq
    
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
        
        # 检查1：是否过度集中在高频字段（柔性提示，不再作为硬失败）
        # 说明：只要跨轮重复率已经达标（上面的硬约束），高频集中不应阻断流程。
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
            
            # 若本轮80%以上字段为历史高频，仅给出提示，不触发失败。
            if len(current_fields) > 0 and overlap_count / len(current_fields) >= 0.8:
                self.logger.info(
                    f"【字段多样性提示】本轮字段较集中于历史高频字段（{overlap_count}/{len(current_fields)}），"
                    f"但跨轮重复率已通过时允许继续执行"
                )
                return True, None
        
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
        """保留输入修改项，不再对字段组合做硬绑定过滤。

        说明：
        - 字段组合由 LLM 与候选字段动态决定，避免固定搭配导致建议重复。
        - 反优化方向由 `_enforce_no_reverse_energy_updates` 负责修正，不在此处做组合层面的硬约束。
        """
        if not target_updates:
            return target_updates

        return target_updates

    def _check_parameter_coherence(self, object_type, field_name, old_value, new_value, coefficient):
        """参数相关性检查（轻量模式）。

        当前不再对字段之间做硬绑定约束，保持 LLM 字段组合灵活性；
        反优化方向由后续统一物理约束函数修正。
        """
        return None

    def _enforce_no_reverse_energy_updates(self, target_updates):
        """强制纠正会导致反优化方向的数值修改，确保关键负荷参数不被增大。

        规则（遵循物理节能常识）：
        - 不允许增大：人员/照明/设备功率密度、渗透率、导热系数、SHGC、新风量参数
        - 若检测到增大，自动改为“小幅下降”而非直接丢弃，减少空计划风险
        """
        if not target_updates:
            return target_updates

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        def _norm(s):
            return str(s or "").upper().replace(" ", "").replace("_", "")

        def _is_forbidden_increase(obj_type, field_name):
            obj_u = str(obj_type or "").upper()
            fld_n = _norm(field_name)

            # 新风两参数：必须下降或保持，不允许上升
            if "OUTDOORAIR" in obj_u or "DESIGNSPECIFICATION:OUTDOORAIR" in obj_u:
                if "PERPERSON" in fld_n or ("PERZONE" in fld_n and "AREA" in fld_n):
                    return True

            # 人员密度/照明/设备功率不应上升
            if "PEOPLE" in obj_u and "PEOPLE" in fld_n:
                return True
            if "LIGHTS" in obj_u and ("WATTS" in fld_n or "LIGHTINGLEVEL" in fld_n):
                return True
            if ("ELECTRICEQUIPMENT" in obj_u or "EQUIPMENT" in obj_u) and ("WATTS" in fld_n or "DESIGNLEVEL" in fld_n or "POWER" in fld_n):
                return True

            # 围护与渗透关键参数不应上升
            if "INFILTRATION" in obj_u and ("FLOW" in fld_n or "AIRCHANGES" in fld_n):
                return True
            if "MATERIAL" in obj_u and "CONDUCTIVITY" in fld_n:
                return True
            if ("WINDOW" in obj_u or "GLAZING" in obj_u) and ("SHGC" in fld_n or "SOLARHEATGAIN" in fld_n):
                return True

            return False

        corrected = 0
        for upd in target_updates:
            coef = upd.get('coefficient')
            old_num = _to_float(upd.get('old_value'))
            new_num = _to_float(upd.get('value'))

            if coef is None or old_num is None or new_num is None:
                continue
            if coef <= 1.0:
                continue
            if not _is_forbidden_increase(upd.get('type', ''), upd.get('field', '')):
                continue

            # 动态回调：增幅越大，回调下降幅度越大（2%~15%）
            proposed_drop = min(0.15, max(0.02, (coef - 1.0) * 0.5))
            fixed_new = old_num * (1.0 - proposed_drop)
            upd['value'] = round(fixed_new, 6)
            upd['coefficient'] = (upd['value'] / old_num) if old_num != 0 else None
            upd['expression'] = f"{upd.get('expression', '')} | auto_non_reverse_fix".strip()
            corrected += 1

            self.logger.info(
                f"  [物理约束修正] {upd.get('type')} ({upd.get('name')}): {upd.get('field')} "
                f"检测到增大({coef:.1%})，已自动修正为下降({proposed_drop:.1%})"
            )

        if corrected > 0:
            self.logger.info(f"[物理约束修正汇总] 共修正 {corrected} 项潜在反优化增大修改")

        return target_updates

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
            # 针对玻璃材料执行完整物理约束，避免 GetMaterialData 报错导致模拟终止。
            field_defs = {
                'SOLAR_T': 'Solar_Transmittance_at_Normal_Incidence',
                'SOLAR_RF': 'Front_Side_Solar_Reflectance_at_Normal_Incidence',
                'SOLAR_RB': 'Back_Side_Solar_Reflectance_at_Normal_Incidence',
                'VIS_T': 'Visible_Transmittance_at_Normal_Incidence',
                'VIS_RF': 'Front_Side_Visible_Reflectance_at_Normal_Incidence',
                'VIS_RB': 'Back_Side_Visible_Reflectance_at_Normal_Incidence',
                'IR_T': 'Infrared_Transmittance_at_Normal_Incidence',
                'IR_EF': 'Front_Side_Infrared_Hemispherical_Emissivity',
                'IR_EB': 'Back_Side_Infrared_Hemispherical_Emissivity',
                'DIRT': 'Dirt_Correction_Factor_for_Solar_and_Visible_Transmittance',
                'THICKNESS': 'Thickness',
                'COND': 'Conductivity',
            }

            values = {}
            modified = {}
            for key, f_name in field_defs.items():
                val, was_modified = _get_value(obj_type_u, obj_name_u, f_name)
                values[key] = _to_float(val)
                modified[key] = was_modified

            # 如果关键光学字段读取失败，则跳过该对象修正，避免误改。
            if values['SOLAR_T'] is None or values['SOLAR_RF'] is None or values['SOLAR_RB'] is None:
                continue

            # 单字段取值约束
            for key in ['SOLAR_T', 'SOLAR_RF', 'SOLAR_RB', 'VIS_T', 'VIS_RF', 'VIS_RB', 'IR_T', 'IR_EF', 'IR_EB', 'DIRT']:
                if values[key] is None:
                    continue
                clamped = _clamp(values[key], 0.0, 1.0)
                if clamped != values[key]:
                    _set_value(obj_type_u, obj_name_u, field_defs[key], clamped, f'{field_defs[key]}_clamp_[0,1]')
                    values[key] = clamped

            # 厚度和导热系数必须为正值
            if values['THICKNESS'] is not None and values['THICKNESS'] <= 0:
                new_thickness = 0.001
                _set_value(obj_type_u, obj_name_u, field_defs['THICKNESS'], new_thickness, 'Thickness_must_be_positive')
                values['THICKNESS'] = new_thickness

            if values['COND'] is not None and values['COND'] <= 0:
                new_cond = 0.01
                _set_value(obj_type_u, obj_name_u, field_defs['COND'], new_cond, 'Conductivity_must_be_positive')
                values['COND'] = new_cond

            max_sum = 0.999

            def _enforce_pair_sum(lhs_key, rhs_key, reason):
                lhs = values.get(lhs_key)
                rhs = values.get(rhs_key)
                if lhs is None or rhs is None:
                    return
                if (lhs + rhs) <= max_sum:
                    return

                # 若 lhs 是本次主动改动而 rhs 不是，则优先调 lhs；否则调 rhs。
                if modified.get(lhs_key, False) and (not modified.get(rhs_key, False)):
                    new_lhs = _clamp(max_sum - rhs, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, field_defs[lhs_key], new_lhs, reason)
                    values[lhs_key] = new_lhs
                else:
                    new_rhs = _clamp(max_sum - lhs, 0.0, 1.0)
                    _set_value(obj_type_u, obj_name_u, field_defs[rhs_key], new_rhs, reason)
                    values[rhs_key] = new_rhs

            # Solar 组合约束
            _enforce_pair_sum('SOLAR_T', 'SOLAR_RF', 'enforce Solar_T+Front_Solar_R<=1')
            _enforce_pair_sum('SOLAR_T', 'SOLAR_RB', 'enforce Solar_T+Back_Solar_R<=1')
            # Visible 组合约束（本次故障根因）
            _enforce_pair_sum('VIS_T', 'VIS_RF', 'enforce Visible_T+Front_Visible_R<=1')
            _enforce_pair_sum('VIS_T', 'VIS_RB', 'enforce Visible_T+Back_Visible_R<=1')
            # Infrared 组合约束
            _enforce_pair_sum('IR_T', 'IR_EF', 'enforce IR_T+Front_IR_Emissivity<=1')
            _enforce_pair_sum('IR_T', 'IR_EB', 'enforce IR_T+Back_IR_Emissivity<=1')

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
            
            safe_globals = {"__builtins__": {}}
            safe_locals = {
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow
            }
            result = eval(expr, safe_globals, safe_locals)
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
        
        # ========== 【关键】先执行物理方向强约束，防止关键负荷参数被增大 ==========
        target_updates = self._enforce_no_reverse_energy_updates(target_updates)

        # ========== 再过滤相关参数反向修改 ==========
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
            self.logger.info(f"【最优值】第{iteration}轮为新的最优方案")
        else:
            if metrics['total_site_energy_kwh'] < self.best_metrics['total_site_energy_kwh']:
                energy_saved = self.best_metrics['total_site_energy_kwh'] - metrics['total_site_energy_kwh']
                save_pct = (energy_saved / self.best_metrics['total_site_energy_kwh'] * 100)
                
                self.best_metrics = metrics
                self.best_iteration = iteration
                
                self.logger.info(f"【最优值更新】第{iteration}轮为新的最优方案！")
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
    
    def _should_early_stop_workflow(self, workflow_id):
        """判断单条工作流是否应提前停止（达到目标或节能幅度收敛）。"""
        if not self.early_stop_enabled:
            return False, ""

        with self.workflows_lock:
            wf = self.workflows.get(workflow_id, {})
            iteration_history = list(wf.get('iteration_history', []) or [])

        if len(iteration_history) < max(2, int(self.early_stop_min_iterations)):
            return False, ""

        baseline = iteration_history[0].get('metrics', {}) if iteration_history else {}
        latest = iteration_history[-1].get('metrics', {}) if iteration_history else {}

        try:
            baseline_total = float(baseline.get('total_site_energy_kwh', 0) or 0)
            latest_total = float(latest.get('total_site_energy_kwh', 0) or 0)
        except Exception:
            return False, ""

        if baseline_total <= 0:
            return False, ""

        latest_saving_pct = (baseline_total - latest_total) / baseline_total * 100.0
        if latest_saving_pct >= float(self.early_stop_target_total_saving_pct):
            return True, f"已达到目标节能率 {latest_saving_pct:.2f}% (阈值 {self.early_stop_target_total_saving_pct:.2f}%)"

        saving_series = []
        for item in iteration_history[1:]:
            m = item.get('metrics', {})
            try:
                curr_total = float(m.get('total_site_energy_kwh', 0) or 0)
            except Exception:
                continue
            saving_series.append((baseline_total - curr_total) / baseline_total * 100.0)

        patience = max(1, int(self.early_stop_convergence_patience))
        if len(saving_series) < patience + 1:
            return False, ""

        recent = saving_series[-(patience + 1):]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        abs_deltas = [abs(d) for d in deltas]
        threshold = float(self.early_stop_min_delta_pct)
        if abs_deltas and all(delta <= threshold for delta in abs_deltas):
            return True, f"节能幅度收敛：最近{patience}轮节能率绝对变化均≤{threshold:.2f}个百分点"

        return False, ""

    def run_optimization_loop(self, max_iterations=None):
        """运行并行工作流优化循环 - 5条工作流同时进行
        
        直接复制原有的单工作流逻辑 5 遍并行执行，不改变任何 prompt、代码逻辑、字段修改规则等。
        """
        if max_iterations is None:
            max_iterations = int(self.max_iterations_cap)

        self.logger.info(f"\n{'█'*80}")
        self.logger.info(f"启动{max_iterations}轮迭代优化 (并行{self.num_workflows}条工作流)")
        self.logger.info("执行模型说明：工作流在线程池并行推进；日志会按完成先后交错显示，不代表串行执行")
        self.logger.info(
            f"早停策略：目标节能率≥{self.early_stop_target_total_saving_pct:.2f}% 或最近"
            f"{self.early_stop_convergence_patience}轮节能率变化≤{self.early_stop_min_delta_pct:.2f}个百分点"
        )
        self.logger.info(f"{'█'*80}\n")
        
        parallel_start = perf_counter()

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

            self.parallel_total_duration_sec = perf_counter() - parallel_start
            self.logger.info(f"【计时】【总】并行流程总耗时(墙钟时间): {self.parallel_total_duration_sec:.2f}s")
        
        # 所有工作流完成后汇总报告
        self._print_parallel_final_report()
    
    def _single_workflow_optimization_loop(self, workflow_id, max_iterations):
        """单工作流优化循环 - 直接复制原有的单工作流逻辑
        
        参数:
            workflow_id: 工作流ID（如 "workflow_1"）
            max_iterations: 最大迭代次数
        """
        thread_handler = self._attach_workflow_thread_handler(workflow_id)
        workflow_start = perf_counter()
        try:
            self._log_context.workflow_id = workflow_id
            self.logger.info(f"\n{'█'*80}")
            self.logger.info(f"启动{max_iterations}轮迭代优化")
            self.logger.info(f"{'█'*80}\n")

            for iteration in range(1, max_iterations + 1):
                self.logger.info(f"\n{'═'*80}")
                self.logger.info(f"【第{iteration}轮/{max_iterations}】")
                self.logger.info(f"{'═'*80}\n")
                self._log_context.iteration = iteration

                with self.workflows_lock:
                    current_idf_path = self.workflows[workflow_id]['current_idf_path']
                    self.current_idf_path = current_idf_path
                    self.current_workflow_id = workflow_id
                    self.iteration_history = self.workflows[workflow_id]['iteration_history']
                    self.best_metrics = self.workflows[workflow_id]['best_metrics']
                    self.best_iteration = self.workflows[workflow_id]['best_iteration']
                    self.field_modification_history = dict(self.workflows[workflow_id].get('field_modification_history', {}))
                    self.last_round_fields = set(self.workflows[workflow_id].get('last_round_fields', set()))

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
                        self.field_modification_history = dict(self.workflows[workflow_id].get('field_modification_history', {}))
                        self.last_round_fields = set(self.workflows[workflow_id].get('last_round_fields', set()))
                    plan = self.generate_optimization_suggestions(current_metrics, iteration)
                    if not plan:
                        self.logger.warning(f"第{iteration}轮LLM无法生成建议，使用默认方案")
                        plan = self._get_default_suggestions(iteration)
                    if not plan:
                        self.logger.error(f"第{iteration}轮无有效计划，中断优化")
                        break

                    # 应用优化（保持并行安全）
                    candidate_modified_idf = None
                    # 仅统计每轮最终执行的建议（中间重试不统计）
                    self._update_suggestion_frequency_for_workflow(workflow_id, plan)
                    with self.workflows_lock:
                        self.current_idf_path = self.workflows[workflow_id]['current_idf_path']
                        self.current_workflow_id = workflow_id
                        self.iteration_history = self.workflows[workflow_id]['iteration_history']
                        self.field_modification_history = dict(self.workflows[workflow_id].get('field_modification_history', {}))
                        self.last_round_fields = set(self.workflows[workflow_id].get('last_round_fields', set()))
                        modified_idf_path = self.apply_optimization(plan, iteration)
                        self.workflows[workflow_id]['field_modification_history'] = dict(self.field_modification_history)
                        self.workflows[workflow_id]['last_round_fields'] = set(self.last_round_fields)

                    if not modified_idf_path or not os.path.exists(modified_idf_path):
                        self.logger.warning(f"第{iteration}轮优化应用失败，使用当前IDF重新运行")
                        sim_idf = current_idf_path
                    else:
                        sim_idf = modified_idf_path
                        candidate_modified_idf = modified_idf_path

                    iter_name = f"iteration_{iteration}"

                sim_dir = self.run_simulation(sim_idf, iter_name, workflow_id=workflow_id)
                if not sim_dir:
                    self.logger.error(f"第{iteration}轮模拟失败，中断优化")
                    break

                metrics = self.extract_metrics(sim_dir)
                if not metrics:
                    self.logger.error(f"第{iteration}轮无法提取能耗数据，中断优化")
                    break

                # 仅在新IDF模拟与提取都成功后，才把它确认为工作流当前IDF
                if iteration > 1 and 'candidate_modified_idf' in locals() and candidate_modified_idf and os.path.exists(candidate_modified_idf):
                    with self.workflows_lock:
                        self.workflows[workflow_id]['current_idf_path'] = candidate_modified_idf

                with self.workflows_lock:
                    self.workflows[workflow_id]['iteration_history'].append({
                        'iteration': iteration,
                        'metrics': metrics,
                        'idf_path': sim_idf,
                        'plan_description': '基准模拟' if iteration == 1 else '优化方案',
                        'is_early_stop_trigger_iteration': False,
                        'early_stop_reason': ''
                    })
                    self.iteration_history = self.workflows[workflow_id]['iteration_history']
                    self.best_metrics = self.workflows[workflow_id]['best_metrics']
                    self.best_iteration = self.workflows[workflow_id]['best_iteration']

                self._print_iteration_savings(metrics, iteration)
                self.update_best_metrics(metrics, iteration)

                with self.workflows_lock:
                    self.workflows[workflow_id]['best_metrics'] = self.best_metrics
                    self.workflows[workflow_id]['best_iteration'] = self.best_iteration
                    self.workflows[workflow_id]['field_modification_history'] = dict(self.field_modification_history)
                    self.workflows[workflow_id]['last_round_fields'] = set(self.last_round_fields)

                should_stop, stop_reason = self._should_early_stop_workflow(workflow_id)
                if should_stop:
                    self.logger.info(f"【早停】第{iteration}轮后停止该工作流：{stop_reason}")
                    with self.workflows_lock:
                        wf = self.workflows.get(workflow_id, {})
                        history = wf.get('iteration_history', [])
                        if history:
                            history[-1]['is_early_stop_trigger_iteration'] = True
                            history[-1]['early_stop_reason'] = stop_reason
                    break
            
        except Exception as e:
            self.logger.error(f"优化循环异常: {e}", exc_info=True)
        finally:
            workflow_elapsed = perf_counter() - workflow_start
            suggestion_obj_freq = {}
            suggestion_field_freq = {}
            with self.workflows_lock:
                if workflow_id in self.workflows:
                    self.workflows[workflow_id]['workflow_total_duration_sec'] = workflow_elapsed
                    suggestion_obj_freq = dict(self.workflows[workflow_id].get('suggestion_object_frequency', {}) or {})
                    suggestion_field_freq = dict(self.workflows[workflow_id].get('suggestion_field_frequency', {}) or {})

            self.logger.info(f"【计时】[{workflow_id}] 工作流总耗时: {workflow_elapsed:.2f}s")

            # 输出每个工作流最终执行建议中的对象/字段出现频率（仅输出>0的项）
            if suggestion_obj_freq:
                self.logger.info("【优化建议对象出现频率（仅最终执行建议）】")
                for obj_key, count in sorted(suggestion_obj_freq.items(), key=lambda x: (-x[1], x[0])):
                    self.logger.info(f"  - {obj_key}: {count}次")
            else:
                self.logger.info("【优化建议对象出现频率（仅最终执行建议）】暂无记录")

            if suggestion_field_freq:
                self.logger.info("【优化建议字段出现频率（仅最终执行建议）】")
                for field_key, count in sorted(suggestion_field_freq.items(), key=lambda x: (-x[1], x[0])):
                    self.logger.info(f"  - {field_key}: {count}次")
            else:
                self.logger.info("【优化建议字段出现频率（仅最终执行建议）】暂无记录")

            if hasattr(self._log_context, 'workflow_id'):
                del self._log_context.workflow_id
            if hasattr(self._log_context, 'iteration'):
                del self._log_context.iteration
            self._detach_workflow_thread_handler(thread_handler)

    def _print_parallel_final_report(self):
        """【并行工作流】汇总所有工作流的优化结果"""
        try:
            if hasattr(self._log_context, 'workflow_id'):
                del self._log_context.workflow_id
            self.logger.info(f"\n{'='*80}")
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
                self._log_context.workflow_id = workflow_id
                try:
                    workflow_data = all_workflows[workflow_id]
                    best_metrics = workflow_data.get('best_metrics')
                    iteration_history = workflow_data.get('iteration_history', [])
                    wlogger = workflow_data.get('logger')

                    if best_metrics:
                        self.logger.info(f"{'─'*80}")
                        self.logger.info(f"[{workflow_id}] 优化结果")
                        self.logger.info(f"{'─'*80}")
                        self.logger.info(f"  最优轮次: 第{workflow_data.get('best_iteration', 0)}轮")
                        self.logger.info(f"  总建筑能耗: {best_metrics.get('total_site_energy_kwh')} kWh")
                        self.logger.info(f"  单位面积能耗: {best_metrics.get('eui_kwh_per_m2')} kWh/m²")
                        self.logger.info(f"  制冷能耗: {best_metrics.get('total_cooling_kwh')} kWh/m²")
                        self.logger.info(f"  供暖能耗: {best_metrics.get('total_heating_kwh')} kWh/m²")

                        if iteration_history and len(iteration_history) > 0:
                            baseline_metrics = iteration_history[0].get('metrics', {})

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

                            total_energy_saved = baseline_metrics.get('total_site_energy_kwh', 0) - best_metrics.get('total_site_energy_kwh', 0)
                            total_energy_pct = (total_energy_saved / baseline_metrics.get('total_site_energy_kwh', 1) * 100) if baseline_metrics.get('total_site_energy_kwh', 0) > 0 else 0
                            eui_saved = baseline_metrics.get('eui_kwh_per_m2', 0) - best_metrics.get('eui_kwh_per_m2', 0)
                            eui_pct = (eui_saved / baseline_metrics.get('eui_kwh_per_m2', 1) * 100) if baseline_metrics.get('eui_kwh_per_m2', 0) > 0 else 0
                            cooling_saved = baseline_metrics.get('total_cooling_kwh', 0) - best_metrics.get('total_cooling_kwh', 0)
                            cooling_pct = (cooling_saved / baseline_metrics.get('total_cooling_kwh', 1) * 100) if baseline_metrics.get('total_cooling_kwh', 0) > 0 else 0
                            heating_saved = baseline_metrics.get('total_heating_kwh', 0) - best_metrics.get('total_heating_kwh', 0)
                            heating_pct = (heating_saved / baseline_metrics.get('total_heating_kwh', 1) * 100) if baseline_metrics.get('total_heating_kwh', 0) > 0 else 0

                            summary_lines = [
                                "  【优化效果】",
                                f"  总建筑能耗节能: {total_energy_saved:.2f} kWh ({total_energy_pct:.1f}%)",
                                f"  单位面积总建筑能耗改善: {eui_saved:.2f} kWh/m² ({eui_pct:.1f}%)",
                                f"  冷却能耗节能: {cooling_saved:.2f} kWh/m² ({cooling_pct:.1f}%)",
                                f"  供暖能耗节能: {heating_saved:.2f} kWh/m² ({heating_pct:.1f}%)",
                                f"  迭代次数: {len(iteration_history)}轮"
                            ]
                            for sl in summary_lines:
                                self.logger.info(sl)
                                if wlogger:
                                    wlogger.info(sl)

                            workflow_total = float(workflow_data.get('workflow_total_duration_sec', 0.0) or 0.0)
                            llm_total = float(workflow_data.get('llm_total_duration_sec', 0.0) or 0.0)
                            sim_total = float(workflow_data.get('sim_total_duration_sec', 0.0) or 0.0)
                            llm_records = workflow_data.get('llm_call_records', []) or []
                            sim_records = workflow_data.get('sim_call_records', []) or []

                            timing_header_lines = [
                                "  【工作流耗时统计】",
                                f"  工作流总时长: {workflow_total:.2f}s",
                                f"  LLM总耗时: {llm_total:.2f}s (调用{len(llm_records)}次)",
                                f"  模拟总耗时: {sim_total:.2f}s (调用{len(sim_records)}次)"
                            ]
                            for line in timing_header_lines:
                                self.logger.info(line)
                                if wlogger:
                                    wlogger.info(line)

                            if llm_records:
                                self.logger.info("  LLM各次耗时明细:")
                                if wlogger:
                                    wlogger.info("  LLM各次耗时明细:")
                                for idx, rec in enumerate(llm_records, start=1):
                                    extra = rec.get('extra', {}) or {}
                                    iter_text = f"第{extra.get('iteration')}轮" if extra.get('iteration') else "未知轮次"
                                    global_idx = extra.get('global_call_index', '?')
                                    temp_val = extra.get('temperature', '?')
                                    detail = (
                                        f"    {idx}. {rec.get('call_label')} "
                                        f"(全局#{global_idx}, {iter_text}, temperature={temp_val}) "
                                        f"-> {float(rec.get('duration_sec', 0.0)):.2f}s"
                                    )
                                    self.logger.info(detail)
                                    if wlogger:
                                        wlogger.info(detail)

                            if sim_records:
                                self.logger.info("  模拟各次耗时明细:")
                                if wlogger:
                                    wlogger.info("  模拟各次耗时明细:")
                                for idx, rec in enumerate(sim_records, start=1):
                                    detail = (
                                        f"    {idx}. {rec.get('call_label')} "
                                        f"-> {float(rec.get('duration_sec', 0.0)):.2f}s"
                                    )
                                    self.logger.info(detail)
                                    if wlogger:
                                        wlogger.info(detail)
                        else:
                            self.logger.warning(f"  ⚠️ {workflow_id} 无迭代历史记录")

                        if best_metrics.get('total_site_energy_kwh', float('inf')) < global_best_energy:
                            global_best_energy = best_metrics.get('total_site_energy_kwh')
                            global_best = best_metrics
                            global_best_workflow = workflow_id

                        try:
                            best_iter = workflow_data.get('best_iteration', 0)
                            if best_iter and iteration_history and 1 <= best_iter <= len(iteration_history):
                                best_idf = iteration_history[best_iter - 1].get('idf_path')
                                lines = self._format_optimal_parameters(self.idf_path, best_idf)
                                for l in lines:
                                    self.logger.info(l)
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
                        try:
                            best_iter = workflow_data.get('best_iteration', 0)
                            if best_iter and iteration_history and 1 <= best_iter <= len(iteration_history):
                                best_idf = iteration_history[best_iter - 1].get('idf_path')
                                lines = self._format_optimal_parameters(self.idf_path, best_idf)
                                for l in lines:
                                    self.logger.info(l)
                                if wlogger:
                                    for l in lines:
                                        wlogger.info(l)
                            else:
                                self.logger.info(f"[{workflow_id}] 无有效最优轮次或最优IDF，跳过最优参数逐项对比输出")
                        except Exception as _e:
                            self.logger.warning(f"打印{workflow_id}最优参数逐项对比时出错: {_e}")
                finally:
                    if hasattr(self._log_context, 'workflow_id'):
                        del self._log_context.workflow_id

            if hasattr(self._log_context, 'workflow_id'):
                del self._log_context.workflow_id
            
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
                self.logger.info(f"\n{'─'*80}")
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
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"【Token使用统计】")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"总Token消耗: {self.total_tokens_used}")
            self.logger.info(f"LLM调用次数: {self.llm_calls_count}")
            if self.llm_calls_count > 0:
                avg_tokens = self.total_tokens_used / self.llm_calls_count
                self.logger.info(f"平均每次Token数: {avg_tokens:.2f}")

            # 全局耗时统计（并行总时长 + 各工作流累计LLM/模拟时长）
            total_llm_duration = 0.0
            total_sim_duration = 0.0
            for _wid, _wf in all_workflows.items():
                total_llm_duration += float(_wf.get('llm_total_duration_sec', 0.0) or 0.0)
                total_sim_duration += float(_wf.get('sim_total_duration_sec', 0.0) or 0.0)

            self.logger.info(f"{'─'*80}")
            self.logger.info("【全局耗时统计】")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"并行流程总耗时: {float(getattr(self, 'parallel_total_duration_sec', 0.0) or 0.0):.2f}s")
            self.logger.info(f"LLM总耗时(各工作流累计): {total_llm_duration:.2f}s")
            self.logger.info(f"模拟总耗时(各工作流累计): {total_sim_duration:.2f}s")

            # 生成并行可视化曲线：每个工作流单独图 + 所有工作流汇总图
            self._generate_parallel_energy_plots(all_workflows)
            
            self.logger.info(f"{'='*80}")
            self.logger.info(f"【并行优化循环完成】")
            self.logger.info(f"{'='*80}\n")
            
        except Exception as e:
            self.logger.error(f"最终报告汇总异常: {e}", exc_info=True)

    def _generate_parallel_energy_plots(self, all_workflows):
        """生成并行工作流能耗变化曲线。

        输出内容：
        1) 每个工作流单独图：供冷实线、供暖虚线
        2) 所有工作流汇总图：不同工作流使用不同颜色，供冷实线、供暖虚线
        3) 每个工作流单独图：总建筑能耗曲线
        4) 所有工作流汇总图：总建筑能耗曲线（颜色区分工作流）
        """
        if threading.current_thread() is not threading.main_thread():
            self.logger.warning("当前非主线程，跳过绘图以避免GUI后端线程冲突")
            return

        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt

            plot_dir = self.plot_dir
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
                total_site = [item['metrics']['total_site_energy_kwh'] for item in history]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x, cooling, color='tab:blue', linestyle='-', marker='o', linewidth=2,
                        label='Cooling (solid)')
                ax.plot(x, heating, color='tab:orange', linestyle='--', marker='s', linewidth=2,
                        label='Heating (dashed)')

                ax.set_title(f"{workflow_id} Cooling/Heating Energy Curve")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Energy (kWh/m2)")
                ax.grid(True, alpha=0.3)
                # 显式设置横轴刻度为连续整数，确保每个迭代都标号
                try:
                    if x:
                        xticks = list(range(min(x), max(x) + 1))
                        ax.set_xticks(xticks)
                except Exception:
                    pass
                ax.legend()

                workflow_plot_path = os.path.join(plot_dir, f"{workflow_id}_cooling_heating_curve_{timestamp}.png")
                fig.savefig(workflow_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

                self.logger.info(f"[{workflow_id}] 已保存单工作流能耗曲线: {workflow_plot_path}")

                # 单工作流总建筑能耗曲线
                fig_total, ax_total = plt.subplots(figsize=(10, 6))
                ax_total.plot(x, total_site, color='tab:green', linestyle='-', marker='o', linewidth=2,
                              label='Total Site Energy')
                ax_total.set_title(f"{workflow_id} Total Site Energy Curve")
                ax_total.set_xlabel("Iteration")
                ax_total.set_ylabel("Energy (kWh)")
                ax_total.grid(True, alpha=0.3)
                # 显式设置横轴刻度为连续整数，确保每个迭代都标号
                try:
                    if x:
                        xticks = list(range(min(x), max(x) + 1))
                        ax_total.set_xticks(xticks)
                except Exception:
                    pass
                ax_total.legend()

                workflow_total_plot_path = os.path.join(
                    plot_dir,
                    f"{workflow_id}_total_site_energy_curve_{timestamp}.png"
                )
                fig_total.savefig(workflow_total_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig_total)

                self.logger.info(f"[{workflow_id}] 已保存单工作流总建筑能耗曲线: {workflow_total_plot_path}")
                valid_workflows.append((workflow_id, x, cooling, heating, total_site))

            # ---------- 2) 所有工作流汇总曲线 ----------
            if valid_workflows:
                fig, ax = plt.subplots(figsize=(12, 7))
                cmap = plt.get_cmap('tab10')

                # 计算全局横轴范围并显示每个整数刻度
                try:
                    global_max_x = max([max(x) for (_, x, _, _, _) in valid_workflows])
                    global_xticks = list(range(1, global_max_x + 1))
                except Exception:
                    global_xticks = None

                for idx, (workflow_id, x, cooling, heating, total_site) in enumerate(valid_workflows):
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
                if global_xticks:
                    try:
                        ax.set_xticks(global_xticks)
                    except Exception:
                        pass
                ax.legend(ncol=2, fontsize=9)

                summary_plot_path = os.path.join(plot_dir, f"all_workflows_cooling_heating_curve_{timestamp}.png")
                fig.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

                self.logger.info(f"已保存并行汇总能耗曲线: {summary_plot_path}")

                # ---------- 3) 所有工作流汇总总建筑能耗曲线 ----------
                fig_total_sum, ax_total_sum = plt.subplots(figsize=(12, 7))
                for idx, (workflow_id, x, cooling, heating, total_site) in enumerate(valid_workflows):
                    color = cmap(idx % 10)
                    ax_total_sum.plot(
                        x,
                        total_site,
                        color=color,
                        linestyle='-',
                        marker='o',
                        linewidth=2,
                        label=f"{workflow_id} Total Site"
                    )

                ax_total_sum.set_title("All Workflows Total Site Energy Curves")
                ax_total_sum.set_xlabel("Iteration")
                ax_total_sum.set_ylabel("Energy (kWh)")
                ax_total_sum.grid(True, alpha=0.3)
                if global_xticks:
                    try:
                        ax_total_sum.set_xticks(global_xticks)
                    except Exception:
                        pass
                ax_total_sum.legend(ncol=2, fontsize=9)

                total_summary_plot_path = os.path.join(
                    plot_dir,
                    f"all_workflows_total_site_energy_curve_{timestamp}.png"
                )
                fig_total_sum.savefig(total_summary_plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig_total_sum)

                self.logger.info(f"已保存并行汇总总建筑能耗曲线: {total_summary_plot_path}")
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
    #     self.logger.info(f"\n{'█'*80}")
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
        #self.logger.info(f"{'='*80}")
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
        # GPT-5.4的常见配额范围，这里以一个示例值展示
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


def _extract_iteration_from_label(call_label):
    """从 simulation call_label 解析迭代轮次。"""
    label = str(call_label or "").strip().lower()
    if label == "initial_baseline":
        return 1
    match = re.search(r"iteration_(\d+)", label)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _safe_pct(numerator, denominator):
    try:
        den = float(denominator or 0.0)
        num = float(numerator or 0.0)
        if den == 0:
            return 0.0
        return num / den * 100.0
    except Exception:
        return 0.0


def _collect_run_export_rows(city, run_tag, optimizer, run_status="success", error_message=""):
    """提取单次运行的导出数据（按工作流）。"""
    rows = {
        'run_summary': [],
        'round_details': [],
        'early_stop': [],
        'object_freq': [],
        'field_freq': [],
    }

    if optimizer is None:
        return rows

    with optimizer.workflows_lock:
        workflows_snapshot = dict(optimizer.workflows)

    for workflow_id, wf in workflows_snapshot.items():
        iteration_history = list(wf.get('iteration_history', []) or [])
        best_metrics = wf.get('best_metrics') or {}
        best_iteration = int(wf.get('best_iteration', 0) or 0)
        workflow_duration = float(wf.get('workflow_total_duration_sec', 0.0) or 0.0)
        llm_duration_total = float(wf.get('llm_total_duration_sec', 0.0) or 0.0)
        sim_duration_total = float(wf.get('sim_total_duration_sec', 0.0) or 0.0)

        baseline_metrics = iteration_history[0].get('metrics', {}) if iteration_history else {}
        baseline_total = float(baseline_metrics.get('total_site_energy_kwh', 0.0) or 0.0)

        best_total = float(best_metrics.get('total_site_energy_kwh', 0.0) or 0.0) if best_metrics else 0.0
        best_saving_pct = _safe_pct(baseline_total - best_total, baseline_total)

        per_round_saving_pct = []
        for item in iteration_history:
            m = item.get('metrics', {})
            current_total = float(m.get('total_site_energy_kwh', 0.0) or 0.0)
            per_round_saving_pct.append(_safe_pct(baseline_total - current_total, baseline_total))
        avg_saving_pct = sum(per_round_saving_pct) / len(per_round_saving_pct) if per_round_saving_pct else 0.0

        llm_records = list(wf.get('llm_call_records', []) or [])
        sim_records = list(wf.get('sim_call_records', []) or [])
        token_records = list(wf.get('llm_token_records', []) or [])

        total_tokens = sum(int(r.get('total_tokens', 0) or 0) for r in token_records)
        input_tokens = sum(int(r.get('input_tokens', 0) or 0) for r in token_records)
        output_tokens = sum(int(r.get('output_tokens', 0) or 0) for r in token_records)
        cached_input_tokens = sum(int(r.get('cached_input_tokens', 0) or 0) for r in token_records)

        avg_round_duration_sec = workflow_duration / len(iteration_history) if iteration_history else 0.0
        early_stop_iteration = len(iteration_history)
        early_stop_reason = ''
        if iteration_history:
            for item in reversed(iteration_history):
                if item.get('is_early_stop_trigger_iteration'):
                    early_stop_iteration = int(item.get('iteration', early_stop_iteration) or early_stop_iteration)
                    early_stop_reason = str(item.get('early_stop_reason', '') or '')
                    break

        rows['run_summary'].append({
            'city': city,
            'run_tag': run_tag,
            'workflow_id': workflow_id,
            'run_status': run_status,
            'error_message': str(error_message or ''),
            'iterations_executed': len(iteration_history),
            'best_iteration': best_iteration,
            'best_saving_pct_total_site': round(best_saving_pct, 4),
            'avg_saving_pct_total_site': round(avg_saving_pct, 4),
            'avg_round_duration_sec': round(avg_round_duration_sec, 4),
            'workflow_total_duration_sec': round(workflow_duration, 4),
            'llm_total_duration_sec': round(llm_duration_total, 4),
            'sim_total_duration_sec': round(sim_duration_total, 4),
            'llm_calls_count': len(llm_records),
            'sim_calls_count': len(sim_records),
            'total_tokens': total_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_input_tokens': cached_input_tokens,
        })

        rows['early_stop'].append({
            'city': city,
            'run_tag': run_tag,
            'workflow_id': workflow_id,
            'run_status': run_status,
            'error_message': str(error_message or ''),
            'early_stop_iteration': early_stop_iteration,
            'iterations_executed': len(iteration_history),
            'early_stop_reason': early_stop_reason,
        })

        llm_duration_by_iter = defaultdict(float)
        for rec in llm_records:
            iter_no = rec.get('extra', {}).get('iteration')
            if iter_no is None:
                continue
            try:
                llm_duration_by_iter[int(iter_no)] += float(rec.get('duration_sec', 0.0) or 0.0)
            except Exception:
                continue

        sim_duration_by_iter = defaultdict(float)
        for rec in sim_records:
            iter_no = _extract_iteration_from_label(rec.get('call_label', ''))
            if iter_no is None:
                continue
            sim_duration_by_iter[int(iter_no)] += float(rec.get('duration_sec', 0.0) or 0.0)

        tokens_by_iter = defaultdict(lambda: {'input': 0, 'output': 0, 'cached': 0, 'total': 0, 'calls': 0})
        for rec in token_records:
            iter_no = rec.get('iteration')
            if iter_no is None:
                continue
            try:
                i = int(iter_no)
            except Exception:
                continue
            tokens_by_iter[i]['input'] += int(rec.get('input_tokens', 0) or 0)
            tokens_by_iter[i]['output'] += int(rec.get('output_tokens', 0) or 0)
            tokens_by_iter[i]['cached'] += int(rec.get('cached_input_tokens', 0) or 0)
            tokens_by_iter[i]['total'] += int(rec.get('total_tokens', 0) or 0)
            tokens_by_iter[i]['calls'] += 1

        for item in iteration_history:
            iteration = int(item.get('iteration', 0) or 0)
            metrics = item.get('metrics', {})
            current_total = float(metrics.get('total_site_energy_kwh', 0.0) or 0.0)
            saving_pct_total = _safe_pct(baseline_total - current_total, baseline_total)
            iter_llm_sec = float(llm_duration_by_iter.get(iteration, 0.0) or 0.0)
            iter_sim_sec = float(sim_duration_by_iter.get(iteration, 0.0) or 0.0)
            iter_total_sec = iter_llm_sec + iter_sim_sec
            token_info = tokens_by_iter.get(iteration, {'input': 0, 'output': 0, 'cached': 0, 'total': 0, 'calls': 0})

            rows['round_details'].append({
                'city': city,
                'run_tag': run_tag,
                'workflow_id': workflow_id,
                'run_status': run_status,
                'error_message': str(error_message or ''),
                'iteration': iteration,
                'total_site_energy_kwh': float(metrics.get('total_site_energy_kwh', 0.0) or 0.0),
                'eui_kwh_per_m2': float(metrics.get('eui_kwh_per_m2', 0.0) or 0.0),
                'total_cooling_kwh': float(metrics.get('total_cooling_kwh', 0.0) or 0.0),
                'total_heating_kwh': float(metrics.get('total_heating_kwh', 0.0) or 0.0),
                'saving_pct_total_site_vs_baseline': round(saving_pct_total, 4),
                'iteration_llm_duration_sec': round(iter_llm_sec, 4),
                'iteration_sim_duration_sec': round(iter_sim_sec, 4),
                'iteration_total_duration_sec': round(iter_total_sec, 4),
                'iteration_llm_call_count': int(token_info.get('calls', 0) or 0),
                'iteration_input_tokens': int(token_info.get('input', 0) or 0),
                'iteration_output_tokens': int(token_info.get('output', 0) or 0),
                'iteration_cached_input_tokens': int(token_info.get('cached', 0) or 0),
                'iteration_total_tokens': int(token_info.get('total', 0) or 0),
                'is_early_stop_trigger_iteration': bool(item.get('is_early_stop_trigger_iteration', False)),
                'early_stop_reason': str(item.get('early_stop_reason', '') or ''),
            })

        obj_freq = wf.get('suggestion_object_frequency', {}) or {}
        for obj_key, count in sorted(obj_freq.items(), key=lambda x: (-x[1], x[0])):
            rows['object_freq'].append({
                'city': city,
                'run_tag': run_tag,
                'workflow_id': workflow_id,
                'run_status': run_status,
                'error_message': str(error_message or ''),
                'object_type': obj_key,
                'frequency': int(count or 0),
            })

        field_freq = wf.get('suggestion_field_frequency', {}) or {}
        for field_key, count in sorted(field_freq.items(), key=lambda x: (-x[1], x[0])):
            rows['field_freq'].append({
                'city': city,
                'run_tag': run_tag,
                'workflow_id': workflow_id,
                'run_status': run_status,
                'error_message': str(error_message or ''),
                'object_field': field_key,
                'frequency': int(count or 0),
            })

    return rows


def _write_rows_to_sheet(workbook, sheet_name, rows, headers):
    ws = workbook.create_sheet(title=sheet_name)
    ws.append(headers)
    for row in rows:
        ws.append([row.get(col, "") for col in headers])


def _export_city_xlsx(city_root, city, city_rows):
    """导出城市级xlsx：汇总、轮次明细、早停、对象频率、字段频率。"""
    try:
        from openpyxl import Workbook
    except Exception as e:
        raise RuntimeError(f"导出xlsx失败，缺少openpyxl依赖: {e}")

    summary_rows = city_rows.get('run_summary', [])
    detail_rows = city_rows.get('round_details', [])
    early_rows = city_rows.get('early_stop', [])
    object_rows = city_rows.get('object_freq', [])
    field_rows = city_rows.get('field_freq', [])

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    summary_headers = [
        'city', 'run_tag', 'workflow_id', 'run_status', 'error_message',
        'iterations_executed', 'best_iteration',
        'best_saving_pct_total_site', 'avg_saving_pct_total_site', 'avg_round_duration_sec',
        'workflow_total_duration_sec', 'llm_total_duration_sec', 'sim_total_duration_sec',
        'llm_calls_count', 'sim_calls_count', 'total_tokens', 'input_tokens',
        'output_tokens', 'cached_input_tokens'
    ]
    detail_headers = [
        'city', 'run_tag', 'workflow_id', 'run_status', 'error_message', 'iteration',
        'total_site_energy_kwh', 'eui_kwh_per_m2', 'total_cooling_kwh', 'total_heating_kwh',
        'saving_pct_total_site_vs_baseline',
        'iteration_llm_duration_sec', 'iteration_sim_duration_sec', 'iteration_total_duration_sec',
        'iteration_llm_call_count', 'iteration_input_tokens', 'iteration_output_tokens',
        'iteration_cached_input_tokens', 'iteration_total_tokens',
        'is_early_stop_trigger_iteration', 'early_stop_reason'
    ]
    early_headers = [
        'city', 'run_tag', 'workflow_id', 'run_status', 'error_message',
        'early_stop_iteration', 'iterations_executed', 'early_stop_reason'
    ]
    object_headers = ['city', 'run_tag', 'workflow_id', 'run_status', 'error_message', 'object_type', 'frequency']
    field_headers = ['city', 'run_tag', 'workflow_id', 'run_status', 'error_message', 'object_field', 'frequency']

    _write_rows_to_sheet(wb, 'run_summary', summary_rows, summary_headers)
    _write_rows_to_sheet(wb, 'round_details', detail_rows, detail_headers)
    _write_rows_to_sheet(wb, 'early_stop', early_rows, early_headers)
    _write_rows_to_sheet(wb, 'object_frequency', object_rows, object_headers)
    _write_rows_to_sheet(wb, 'field_frequency', field_rows, field_headers)

    # 城市层聚合统计
    city_agg_rows = []
    if summary_rows:
        best_values = [float(r.get('best_saving_pct_total_site', 0.0) or 0.0) for r in summary_rows]
        avg_values = [float(r.get('avg_saving_pct_total_site', 0.0) or 0.0) for r in summary_rows]
        avg_time_values = [float(r.get('avg_round_duration_sec', 0.0) or 0.0) for r in summary_rows]
        token_values = [int(r.get('total_tokens', 0) or 0) for r in summary_rows]
        city_agg_rows.append({
            'city': city,
            'run_workflow_count': len(summary_rows),
            'best_saving_pct_max': round(max(best_values), 4) if best_values else 0.0,
            'best_saving_pct_mean': round(sum(best_values) / len(best_values), 4) if best_values else 0.0,
            'avg_saving_pct_mean': round(sum(avg_values) / len(avg_values), 4) if avg_values else 0.0,
            'avg_round_duration_sec_mean': round(sum(avg_time_values) / len(avg_time_values), 4) if avg_time_values else 0.0,
            'tokens_total': sum(token_values),
            'tokens_mean_per_run_workflow': round(sum(token_values) / len(token_values), 2) if token_values else 0.0,
        })
    city_agg_headers = [
        'city', 'run_workflow_count', 'best_saving_pct_max', 'best_saving_pct_mean',
        'avg_saving_pct_mean', 'avg_round_duration_sec_mean', 'tokens_total', 'tokens_mean_per_run_workflow'
    ]
    _write_rows_to_sheet(wb, 'city_aggregate', city_agg_rows, city_agg_headers)

    output_path = os.path.join(city_root, f"{city}_统计汇总.xlsx")
    wb.save(output_path)
    return output_path


if __name__ == "__main__":
    # 配置文件路径
    IDF_PATH = "in.idf"
    IDD_PATH = "Energy+.idd"
    API_KEY_PATH = "api_key.txt"
    WEATHER_DIR = "weather"

    # 仅运行指定四个城市，避免API费用过高
    TARGET_CITIES = ["Beijing", "Guangzhou", "Shanghai", "Wuhan"]
    RUNS_PER_CITY = 10

    # 所有城市结果统一收敛到该目录下
    ROOT_OUTPUT_DIR = "各城市迭代结果"
    
    try:
        # 建立城市名到EPW路径的映射
        city_epw_map = {}
        for city in TARGET_CITIES:
            epw_path = os.path.join(WEATHER_DIR, f"{city}.epw")
            if os.path.exists(epw_path):
                city_epw_map[city] = epw_path
            else:
                print(f"⚠ 跳过城市 {city}：未找到气象文件 {epw_path}")

        if not city_epw_map:
            raise FileNotFoundError("未找到任何目标城市的EPW文件，请检查weather目录")

        total_runs = len(city_epw_map) * RUNS_PER_CITY
        completed_runs = 0

        for city, epw_path in city_epw_map.items():
            city_root = os.path.join(ROOT_OUTPUT_DIR, city)
            os.makedirs(city_root, exist_ok=True)
            city_rows = {
                'run_summary': [],
                'round_details': [],
                'early_stop': [],
                'object_freq': [],
                'field_freq': [],
            }

            print("\n" + "=" * 100)
            print(f"开始城市批量模拟: {city} | EPW: {epw_path}")
            print("=" * 100)

            for run_idx in range(1, RUNS_PER_CITY + 1):
                run_tag = f"run_{run_idx:02d}"
                run_root = os.path.join(city_root, run_tag)

                # 每次运行的日志、结果、图像独立归档，避免相互覆盖
                log_dir = os.path.join(run_root, f"optimization_logs_并行_{city}_{run_tag}")
                optimization_dir = os.path.join(run_root, f"optimization_results_并行_{city}_{run_tag}")
                plot_dir = os.path.join(run_root, f"optimization_plot_并行_{city}_{run_tag}")

                os.makedirs(run_root, exist_ok=True)

                print("\n" + "-" * 100)
                print(f"[{city}] 第{run_idx}/{RUNS_PER_CITY}次自动运行开始")
                print(f"输出目录: {run_root}")
                print("-" * 100)

                try:
                    optimizer = None
                    optimizer = EnergyPlusOptimizer(
                        idf_path=IDF_PATH,
                        idd_path=IDD_PATH,
                        api_key_path=API_KEY_PATH,
                        epw_path=epw_path,
                        log_dir=log_dir,
                        optimization_dir=optimization_dir,
                        plot_dir=plot_dir
                    )

                    # 保持原逻辑：每次运行内仍由早停机制决定提前结束
                    optimizer.run_optimization_loop()

                    run_rows = _collect_run_export_rows(city, run_tag, optimizer, run_status="success", error_message="")
                    for key in city_rows.keys():
                        city_rows[key].extend(run_rows.get(key, []))

                    completed_runs += 1
                    print(f"✓ [{city}] 第{run_idx}/{RUNS_PER_CITY}次运行完成")
                    print(f"✓ 总进度: {completed_runs}/{total_runs}")

                except Exception as run_err:
                    run_rows = _collect_run_export_rows(
                        city,
                        run_tag,
                        optimizer if 'optimizer' in locals() else None,
                        run_status="failed",
                        error_message=str(run_err)
                    )
                    for key in city_rows.keys():
                        city_rows[key].extend(run_rows.get(key, []))

                    completed_runs += 1
                    print(f"✗ [{city}] 第{run_idx}/{RUNS_PER_CITY}次运行失败: {run_err}")
                    print(f"✗ 总进度: {completed_runs}/{total_runs}")

            # 每个城市运行结束后导出该城市xlsx
            try:
                xlsx_path = _export_city_xlsx(city_root, city, city_rows)
                print(f"✓ [{city}] 统计表导出完成: {xlsx_path}")
            except Exception as export_err:
                print(f"✗ [{city}] 统计表导出失败: {export_err}")

        print("\n" + "=" * 100)
        print(f"全部批量任务结束，共执行 {completed_runs}/{total_runs} 次")
        print(f"结果总目录: {ROOT_OUTPUT_DIR}")
        print("=" * 100)

    except Exception as e:
        print(f"✗ 批量优化过程异常: {e}")
        import traceback
        traceback.print_exc()
