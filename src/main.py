import os
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import hashlib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import sys
import traceback
import time
import threading
import signal
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint import MemorySaver  # 暂时禁用此导入

# 定义状态类型
class GraphState(TypedDict):
    """图状态类型"""
    account_id: str
    account_data: Dict[str, Any]
    analysis_result: str
    error: str
    dates: List[str]
    readings: List[float]
    amounts: List[float]
    daily_avgs: List[float]
    stats: Dict[str, Any]
    success: bool

def load_config():
    """加载配置"""
    load_dotenv()
    
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return {
        "db1_connection_string": os.getenv("DB1_CONNECTION_STRING"),
        "db2_connection_string": os.getenv("DB2_CONNECTION_STRING"),
        "db1_mongo_uri": os.getenv("DB1_MONGO_URI"),
        "db2_mongo_uri": os.getenv("DB2_MONGO_URI"),
        "db1_database": os.getenv("DB1_DATABASE"),
        "db2_database": os.getenv("DB2_DATABASE"),
        "db1_collection": os.getenv("DB1_COLLECTION"),
        "db2_collection": os.getenv("DB2_COLLECTION"),
        "model_type": "siliconflow",  # 使用硅基流动平台
        "llm_api_key": os.getenv("DEEPSEEK_API_KEY", "sk-qsybuxxdlcvuhmtmbollzxzxvkwzqzxbkmbockxpujpcjyfk"),
        "llm_api_base": os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1"),
        "max_workers": int(os.getenv("MAX_WORKERS", "5")),  # 减少工作线程数
        "batch_size": int(os.getenv("BATCH_SIZE", "5")),    # 减少批处理大小
        "use_local_storage": os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true",
        "account_data_db": os.path.join(root_dir, "data/account_data1.db"),  # 修改为account_data1.db
        "historical_data_db": os.path.join(root_dir, "data/account_data2.db"),  # 添加account_data2.db
        "analysis_results_db": os.path.join(root_dir, "data/analysis_results.db"),  # 添加分析结果数据库
        "root_dir": root_dir,
        "use_mock_analysis": os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true",  # 添加模拟分析开关
        "reports_dir": os.path.join(root_dir, "reports"),
        # MCP服务器配置
        "use_mcp_server": os.getenv("USE_MCP_SERVER", "false").lower() == "true",
        "mcp_server_host": os.getenv("MCP_SERVER_HOST", "localhost"),
        "mcp_server_port": int(os.getenv("MCP_SERVER_PORT", "5000")),
        "mcp_server_path": os.getenv("MCP_SERVER_PATH", "/api/data"),
        "mcp_auth_token": os.getenv("MCP_AUTH_TOKEN", "")
    }

class DeepSeekR1:
    """DeepSeek-R1 模型的包装器 (硅基流动平台)"""
    
    def __init__(self, api_key=None, api_base="https://api.siliconflow.cn/v1", temperature=0.1):
        """初始化DeepSeek-R1模型"""
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("请提供DeepSeek API密钥")
            
        self.api_base = api_base
        self.temperature = temperature
        self.model_name = "deepseek-ai/DeepSeek-R1"  # 直接使用这个模型名称
        logger.info(f"初始化DeepSeek-R1模型，API基础URL: {self.api_base}，模型: {self.model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, messages):
        """生成回复"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备请求体，优化参数配置
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 300,
            "stream": False,  # 禁用流式响应
            "tool_choice": "none",  # 禁用工具调用
            "response_format": {"type": "text"},  # 强制文本格式
            "seed": 42,  # 使用固定种子以提高一致性
            # 关键参数：禁用思考过程/推理过程
            "tool_config": {"none": {"reasoning": "never"}}  # 禁用思考过程
        }
        
        logger.info(f"发送请求到DeepSeek API...")
        
        # 开始计时
        start_time = time.time()
        
        # 创建一个事件来控制计时器线程
        stop_event = threading.Event()
        
        def print_elapsed_time():
            # 减少输出频率以降低系统负担
            start_time = time.time()
            for i in range(120):  # 增加等待时间到120秒
                if stop_event.is_set():
                    return
                elapsed = time.time() - start_time
                # 修改这里，添加当前的时间显示
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\r等待API响应... {elapsed:.1f}秒 | 当前时间: {current_time}", end="", flush=True)
                time.sleep(2.0)  # 每2秒更新一次
        
        # 创建并启动计时器线程
        timer_thread = threading.Thread(target=print_elapsed_time)
        timer_thread.daemon = True
        timer_thread.start()
        
        try:
            # 使用较长的超时时间，确保API有足够时间响应
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=90  # 大幅增加超时时间到90秒
            )
            
            # 计算总用时
            total_time = time.time() - start_time
            
            # 停止计时器线程
            stop_event.set()
            timer_thread.join(timeout=0.5)  # 等待线程结束，最多0.5秒
            
            # 清除计时器输出行
            print(f"\r完成! 用时: {total_time:.0f}秒{' ' * 20}")
            
            response.raise_for_status()
            
            result = response.json()
            
            # 检查是否有content字段
            content = result["choices"][0]["message"].get("content", "")
            
            # 如果content为空，检查是否有reasoning_content字段
            if (not content or content.strip() == "") and "reasoning_content" in result["choices"][0]["message"]:
                content = result["choices"][0]["message"]["reasoning_content"]
            
            # 如果内容仍然为空，抛出异常触发重试
            if not content or content.strip() == "":
                raise Exception("API返回空内容")
            
            return content
            
        except requests.exceptions.Timeout:
            # 超时处理
            stop_event.set()
            timer_thread.join(timeout=0.5)
            print(f"\rAPI请求超时! {' ' * 30}")
            logger.warning("API请求超时")
            raise Exception("API超时")
            
        except Exception as e:
            # 确保计时器线程停止
            stop_event.set()
            timer_thread.join(timeout=0.5)
            
            # 清除计时器输出并打印换行
            print(f"\r请求失败: {type(e).__name__}{' ' * 30}")
            
            # 抛出异常让外部捕获
            raise Exception(f"API错误: {type(e).__name__}")

class LocalStorageManager:
    """本地存储管理器，用于替代Redis"""
    
    def __init__(self, storage_file="processed_accounts.db"):
        """初始化本地存储"""
        # 确保目录存在
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
        
        self.storage_file = storage_file
        self.conn = sqlite3.connect(storage_file)
        self.create_tables()
        
    def create_tables(self):
        """创建必要的表"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_accounts (
            key TEXT PRIMARY KEY,
            data TEXT,
            timestamp TEXT
        )
        ''')
        self.conn.commit()
        
    def set(self, key, value, ex=None):
        """存储键值对"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # 确保值是字符串，如果是列表或字典则转换为JSON字符串
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        
        cursor.execute(
            "INSERT OR REPLACE INTO processed_accounts (key, data, timestamp) VALUES (?, ?, ?)",
            (key, value, timestamp)
        )
        self.conn.commit()
        return True
        
    def get(self, key):
        """获取键对应的值"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM processed_accounts WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result:
            try:
                return json.loads(result[0])
            except:
                return result[0]
        return None
        
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

class AccountDataAnalyzer:
    """账户数据分析器"""
    
    def __init__(self, db_path):
        """初始化分析器"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect_db(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"成功连接到数据库: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def get_transaction_data(self):
        """获取用水数据"""
        try:
            if not self.conn:
                self.connect_db()
            
            # 首先获取所有表名
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()
            logger.info(f"数据库中的表: {[table[0] for table in tables]}")
            
            if not tables:
                raise Exception("数据库中没有表")
            
            # 使用meter_readings表
            table_name = "meter_readings"
            logger.info(f"使用表: {table_name}")
            
            # 获取表结构
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = self.cursor.fetchall()
            logger.info(f"{table_name}表结构:")
            for col in columns:
                logger.info(f"列名: {col[1]}, 类型: {col[2]}")
            
            # 获取所有账户ID
            self.cursor.execute(f"SELECT DISTINCT account_id FROM {table_name} ORDER BY account_id")
            account_ids = [row[0] for row in self.cursor.fetchall()]
            
            # 限制账户数量
            account_ids = account_ids[:10]  # 获取前10个账户的数据
            
            result = []
            for account_id in account_ids:
                # 获取账户的用水数据
                self.cursor.execute(
                    f"""
                    SELECT 
                        reading_time,
                        current_usage,
                        daily_average
                    FROM {table_name} 
                    WHERE account_id = ? 
                    ORDER BY reading_time
                    """, 
                    (account_id,)
                )
                rows = self.cursor.fetchall()
                
                dates = []
                amounts = []
                daily_avgs = []
                for row in rows:
                    dates.append(row[0])  # reading_time
                    amounts.append(row[1])  # current_usage
                    daily_avgs.append(row[2])  # daily_average
                
                if dates:  # 只有当有有效数据时才添加
                    account_data = {
                        "account_id": account_id,
                        "dates": dates,
                        "amount": amounts,
                        "daily_average": daily_avgs
                    }
                    result.append(account_data)
            
            logger.info(f"成功获取{len(result)}个账户的用水数据")
            return result
            
        except sqlite3.Error as e:
            logger.error(f"获取用水数据失败: {str(e)}")
            raise Exception(f"数据获取失败: {str(e)}")
    
    def prepare_data_for_llm(self):
        """准备用于大模型分析的数据"""
        # 获取用水数据
        water_usage_data = self.get_transaction_data()
        
        # 准备完整数据
        result = {
            "database_name": os.path.basename(self.db_path),
            "analysis_target": "用水量异常分析",
            "water_usage_data": water_usage_data
        }
            
        return result

class LLMAnalyzer:
    """大模型分析器，使用DeepSeek-R1模型分析数据"""
    
    def __init__(self, config):
        """初始化大模型分析器"""
        self.config = config
        self.model = DeepSeekR1(
            api_key=config.get("llm_api_key"),
            api_base=config.get("llm_api_base"),
            temperature=0.1  # 降低温度以获得更确定性的回答
        )
        logger.info("初始化大模型分析器完成")
    
    def analyze_data(self, data, custom_prompt=None):
        """使用大模型分析数据"""
        try:
            # 准备提示
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._create_prompt(data)
            
            # 准备消息，使用更强制性的系统提示，以字典形式
            messages = [
                {"role": "system", "content": """直接生成分析结果，不要思考过程:
你的任务是直接填写用水量分析模板，绝对禁止发送任何思考过程。
1. 不要写"分析"、"根据"、"经过"、"我认为"等词语
2. 不要说明你如何得出结论
3. 不要解释任何内容，只填入模板中的空白处
4. 你必须直接使用系统给你的模板，填入结果并返回
5. 不要重述问题或数据，只输出填好的模板
6. 使用提供的填写提示直接填入对应位置"""},
                {"role": "user", "content": prompt}
            ]
            
            # 调用大模型API
            logger.info("调用大模型分析...")
            
            try:
                analysis = self.model.generate(messages)
            except Exception as e:
                # API调用失败，返回超时信息
                logger.warning(f"API调用失败: {str(e)}")
                return {
                    "success": False,
                    "error": "API调用失败",
                    "analysis": self._generate_mock_analysis(data)
                }
            
            # 后处理：移除任何分析过程，只保留结论
            import re
            
            # 首先尝试提取```块内的内容
            code_pattern = r"```(?:markdown)?\s*([\s\S]*?)\s*```"
            code_matches = re.findall(code_pattern, analysis)
            if code_matches:
                analysis = code_matches[0].strip()
                logger.info("成功提取代码块中的内容")
            else:
                # 尝试查找账户关键发现段落
                account_id = data["water_usage_data"][0]["account_id"]
                finding_pattern = r"\*\*账户\s*" + str(account_id) + r"\s*关键发现\*\*([\s\S]*)"
                finding_matches = re.search(finding_pattern, analysis)
                if finding_matches:
                    # 如果找到了关键发现段落，加上前面可能的漏水警告
                    warning_pattern = r"(⚠️\s*漏水风险警告)"
                    warning_match = re.search(warning_pattern, analysis)
                    warning_text = warning_match.group(1) + "\n\n" if warning_match else ""
                    
                    analysis = warning_text + "**账户 " + str(account_id) + " 关键发现**" + finding_matches.group(1)
                    logger.info("成功提取关键发现段落")
                elif "关键发现" in analysis:
                    # 如果找到了关键发现但没有账户ID，尝试提取关键发现段落
                    finding_pattern = r"(?:\*\*)?关键发现(?:\*\*)?([\s\S]*)"
                    finding_matches = re.search(finding_pattern, analysis)
                    if finding_matches:
                        # 同样检查漏水警告
                        warning_pattern = r"(⚠️\s*漏水风险警告)"
                        warning_match = re.search(warning_pattern, analysis)
                        warning_text = warning_match.group(1) + "\n\n" if warning_match else ""
                        
                        analysis = warning_text + "**账户 " + str(account_id) + " 关键发现**" + finding_matches.group(1)
                        logger.info("成功提取关键发现内容")
            
            # 如果还是非常长，或者包含明显的分析过程词汇，则进行简单截断
            if len(analysis.split()) > 100 or "分析" in analysis or "根据" in analysis:
                logger.warning("对分析结果进行后处理...")
                lines = analysis.split('\n')
                filtered_lines = []
                skip_line = False
                
                for line in lines:
                    # 跳过包含分析过程关键词的行
                    if any(word in line.lower() for word in ["分析", "根据", "通过", "我", "认为", "观察", "基于"]):
                        skip_line = True
                        continue
                    
                    # 如果之前跳过了行，现在遇到了新的段落标记，恢复添加
                    if skip_line and (line.strip().startswith("**") or line.strip().startswith("#")):
                        skip_line = False
                    
                    if not skip_line:
                        filtered_lines.append(line)
                
                analysis = "\n".join(filtered_lines)
                
                # 如果处理后内容为空，则返回超时信息
                if not analysis.strip():
                    logger.warning("处理后内容为空，返回API超时信息")
                    return {
                        "success": False,
                        "error": "处理后内容为空",
                    "analysis": self._generate_mock_analysis(data)
                }
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.warning(f"分析过程中出错: {type(e).__name__}")
            
            # 生成超时通知信息
            return {
                "success": False,
                "error": str(e),
                "analysis": self._generate_mock_analysis(data)
            }
    
    def _create_prompt(self, data):
        """创建分析提示"""
        # 获取用水数据
        db_name = data.get("database_name", "未知数据库")
        analysis_target = data.get("analysis_target", "未知分析目标")
        water_usage_data = data.get("water_usage_data", [])
        
        # 确保有数据
        if not water_usage_data:
            return "没有可分析的用水数据"
        
        # 获取账户数据（只有一个账户）
        account_data = water_usage_data[0]
        account_id = account_data.get("account_id", "未知")
        dates = account_data.get("dates", [])
        readings = account_data.get("current_reading", [])
        amounts = account_data.get("amount", [])
        daily_avgs = account_data.get("daily_average", [])
        
        # 计算一些基本统计数据
        if amounts:
            amounts_float = [float(amount) for amount in amounts]
            avg_amount = sum(amounts_float) / len(amounts_float)
            max_amount = max(amounts_float)
            max_date = dates[amounts_float.index(max_amount)]
            min_amount = min(amounts_float)
            time_span = f"{dates[0]} 至 {dates[-1]}"
            
            # 检测是否存在高用量和异常增长
            has_zero_usage = any(float(amount) == 0 for amount in amounts)
            has_high_usage = any(float(amount) > 5 for amount in amounts)
            
            # 计算最大日均用水量
            max_daily_avg = max([float(avg) for avg in daily_avgs]) if daily_avgs else 0
            has_high_daily = max_daily_avg > 3
            
            # 计算是否有风险
            has_leak_risk = has_high_daily or has_high_usage
        else:
            avg_amount = 0
            max_amount = 0
            max_date = "未知"
            min_amount = 0
            time_span = "未知"
            has_zero_usage = False
            has_high_usage = False
            has_leak_risk = False
        
        # 添加water_meter_reading信息到提示中
        reading_info = f"\n   最新读数 {readings[-1]}，初始读数 {readings[0]}" if readings and len(readings) > 0 else ""
        
        # 超级精简的提示，直接提供几乎所有分析结果，让模型只需要格式化
        prompt = f"""
直接使用下面的模板，只填入结果，不要任何分析过程。

模板结构:
{("⚠️ 漏水风险警告" if has_leak_risk else "")}

**账户 {account_id} 关键发现**
**漏水可能性**: _____（只填入：无/低/中/高）
1. **记录数量和时间跨度**: {len(dates)}条记录，覆盖{time_span}
2. **异常用水**: _____（只填入："存在高用水量"或"无明显异常用水"）
3. **漏水风险**: _____（只填入相应风险描述）{reading_info}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

填写提示:
- 漏水可能性: {risk_level if has_leak_risk else "由你判断"}
- 异常用水: {f"存在{max_amount}方的高用水量（{max_date}）" if has_high_usage else "无明显异常用水"}
- 漏水风险: {f"日均用水量达{max_daily_avg:.1f}方，超过安全阈值3方" if has_high_daily else "无明显漏水风险"}

只需返回填好的模板，不要任何思考过程。不要解释你如何得出结论。不要重述任何数据或问题。
"""
        
        return prompt
    
    def _generate_mock_analysis(self, data):
        """生成超时通知（当API调用超时时使用）"""
        # 获取账户ID
        water_usage_data = data.get("water_usage_data", [])
        
        if not water_usage_data:
            return "# AI分析超时\n\n很抱歉，AI分析请求超时，无法提供完整的分析报告。"
        
        account_data = water_usage_data[0]
        account_id = account_data.get("account_id", "未知")
        
        return f"""# 账户 {account_id} 分析失败

## ⚠️ AI分析请求超时

很抱歉, AI分析请求超时, 无法提供完整的水用量分析报告.

请考虑以下可能的解决方案:
1. 稍后再次尝试分析
2. 检查API密钥和网络连接是否正常
3. 如果问题持续出现, 请联系系统管理员

**注意**: 由于分析未完成, 无法提供漏水可能性评估(无/低/中/高)

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
            """

def test_model_list():
    """测试可用的模型列表"""
    # 从环境变量加载API密钥
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1")
    
    # 获取模型列表
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("获取可用模型列表...")
        response = requests.get(
            f"{api_base}/models", 
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        models = result.get("data", [])
        
        # 筛选DeepSeek模型
        deepseek_models = [model for model in models if "deepseek" in model.get("id", "").lower()]
        
        if deepseek_models:
            logger.info(f"找到 {len(deepseek_models)} 个DeepSeek模型:")
            for model in deepseek_models:
                logger.info(f"- {model.get('id')}")
            
            # 返回第一个DeepSeek模型ID
            return deepseek_models[0].get("id")
        else:
            logger.warning("没有找到可用的DeepSeek模型")
            return None
            
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return None

def process_historical_data(history_rows):
    """处理历史抄表数据并计算日均用水量
    
    参数:
        history_rows: 从数据库获取的历史抄表记录
        
    返回:
        包含历史日期、读数和日均用水量的字典
    """
    from datetime import datetime
    
    # 初始化历史数据变量
    historical_dates = []
    historical_readings = []
    historical_daily_avgs = []
    historical_daily_avg = 0
    
    if len(history_rows) >= 2:
        # 将历史数据按日期排序
        history_rows_sorted = sorted(history_rows, key=lambda x: datetime.strptime(x[0], "%Y/%m/%d"))
        
        print(f"  历史数据排序后:")
        for index, row in enumerate(history_rows_sorted):
            print(f"    {index+1}. {row[0]} - {row[1]}方")
            historical_dates.append(row[0])
            historical_readings.append(row[1])
        
        for i in range(1, len(history_rows_sorted)):
            # 计算两次抄表之间的天数
            prev_date = datetime.strptime(history_rows_sorted[i-1][0], "%Y/%m/%d")
            curr_date = datetime.strptime(history_rows_sorted[i][0], "%Y/%m/%d")
            days_diff = (curr_date - prev_date).days
            
            # 确保天数不为零，避免除零错误
            if days_diff <= 0:
                print(f"  警告: 日期差异异常 {prev_date.date()} -> {curr_date.date()}, 跳过此计算")
                continue
            
            # 计算用水量差值
            prev_reading = history_rows_sorted[i-1][1]
            curr_reading = history_rows_sorted[i][1]
            usage = curr_reading - prev_reading
            
            # 计算日均用水量
            daily_avg = usage / days_diff
            
            # 只添加合理的值（正值）
            if daily_avg > 0:
                historical_daily_avgs.append(daily_avg)
                # 打印日志，帮助调试计算过程
                print(f"  历史: {prev_date.date()} -> {curr_date.date()}, {days_diff}天, 用水量: {usage}方, 日均: {daily_avg:.4f}方/日")
        
        # 计算平均日均用水量，只使用正值
        if historical_daily_avgs:
            historical_daily_avg = sum(historical_daily_avgs) / len(historical_daily_avgs)
            print(f"  历史平均日用水量: {historical_daily_avg:.4f}方/日")
        else:
            print(f"  没有找到有效的历史日均用水量数据")
    
    return {
        "historical_dates": historical_dates,
        "historical_readings": historical_readings,
        "historical_daily_avgs": historical_daily_avgs,
        "historical_daily_avg": historical_daily_avg
    }

def fetch_account_data(state: GraphState) -> GraphState:
    """从数据库获取账户数据"""
    config = load_config()
    account_id = state["account_id"]
    
    try:
        # 连接数据库1 (account_data1.db)
        conn1 = sqlite3.connect(config["account_data_db"])
        cursor1 = conn1.cursor()
        
        # 获取账户数据
        cursor1.execute("""
            SELECT 
                reading_time,
                current_reading,  -- 实际抄表数
                current_usage     -- 实际用水量
            FROM meter_readings 
            WHERE account_id = ?
            ORDER BY reading_time
        """, (account_id,))
        rows1 = cursor1.fetchall()
        
        # 解析数据
        dates = []
        readings = []
        amounts = []
        daily_avgs = []
        
        for row in rows1:
            dates.append(row[0])
            readings.append(row[1])
            amounts.append(row[2])
            
        # 关闭连接
        conn1.close()
        
        # 连接数据库2 (account_data2.db)
        conn2 = sqlite3.connect(config["historical_data_db"])
        cursor2 = conn2.cursor()
        
        # 获取历史抄表数据
        cursor2.execute("""
            SELECT 
                reading_time,
                reading_value    -- 日均用水量
            FROM historical_readings 
            WHERE account_id = ?
            ORDER BY reading_time
        """, (account_id,))
        
        rows2 = cursor2.fetchall()
        
        # 处理历史数据
        historical_data = process_historical_data(rows2)
        
        # 关闭连接
        conn2.close()
        
        if not dates:
            print(f"警告：账户 {account_id} 在数据库1中没有找到数据")
            return {
                **state,
                "success": False,
                "error": f"未找到账户数据"
            }
        
        return {
            **state,
            "dates": dates,
            "readings": readings,
            "amounts": amounts,
            "daily_avgs": daily_avgs,
            "historical_dates": historical_data["historical_dates"],
            "historical_readings": historical_data["historical_readings"],
            "historical_daily_avgs": historical_data["historical_daily_avgs"],
            "historical_daily_avg": historical_data["historical_daily_avg"],
            "success": True
        }
    except Exception as e:
        print(f"获取账户 {account_id} 数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "success": False,
            "error": f"数据库访问错误: {str(e)}"
        }

def calculate_statistics(state: GraphState) -> GraphState:
    """计算统计数据"""
    if not state["success"]:
        return state
    
    try:
        dates = state["dates"]
        amounts = state["amounts"]  # current_usage: 该日期的抄表用水量
        readings = state["readings"]  # current_reading: 抄的水表止度
        daily_avgs = state["daily_avgs"]
        historical_daily_avgs = state.get("historical_daily_avgs", [])
        historical_daily_avg = state.get("historical_daily_avg", 0)
        
        if not amounts:
            raise ValueError("没有用水数据")
            
        # 转换为浮点数
        amounts_float = [float(amount) for amount in amounts]
        
        # 计算基本统计数据
        stats = {
            "avg_amount": sum(amounts_float) / len(amounts_float),
            "max_amount": max(amounts_float),
            "max_date": dates[amounts_float.index(max(amounts_float))],
            "min_amount": min(amounts_float),
            "time_span": f"{dates[0]} 至 {dates[-1]}",
            "record_count": len(dates),
            "has_zero_usage": any(float(amount) == 0 for amount in amounts),
            # 单次用水超过5方作为漏水一个判断条件 - 使用current_usage判断
            "has_high_usage": any(float(amount) > 5 for amount in amounts),
        }
        
        # 计算日均用水
        if daily_avgs:
            daily_avgs_float = [float(avg) for avg in daily_avgs]
            stats["max_daily_avg"] = max(daily_avgs_float)
            
            # 检查是否存在高日均用水
            stats["has_high_daily"] = stats["max_daily_avg"] > 3
            
            # 检查近期是否有两次漏水情况（超过3方的日均用水被视为漏水）
            recent_daily_avgs = daily_avgs_float[-5:] if len(daily_avgs_float) > 5 else daily_avgs_float
            leak_count = sum(1 for avg in recent_daily_avgs if avg > 3)
            stats["has_multiple_leaks"] = leak_count >= 2
        else:
            stats["max_daily_avg"] = 0
            stats["has_high_daily"] = False
            stats["has_multiple_leaks"] = False
        
        # 使用已处理的历史数据日均用水量
        if historical_daily_avg > 0:
            stats["historical_avg_daily"] = historical_daily_avg
            
            # 计算最大历史日均用水量
            if historical_daily_avgs:
                stats["historical_max_daily"] = max(historical_daily_avgs)
            else:
                stats["historical_max_daily"] = historical_daily_avg
            
            # 计算历史数据与当前数据的差异比例（使用current_usage）
            if stats["avg_amount"] > 0:
                stats["usage_increase_ratio"] = stats["avg_amount"] / historical_daily_avg
                
                # 检查是否连续超过历史平均用水量一倍 - 使用current_usage判断
                if len(amounts_float) >= 3:
                    recent_amounts = amounts_float[-3:]
                    recent_ratios = [amount / historical_daily_avg for amount in recent_amounts]
                    stats["has_continuous_high_usage"] = all(ratio > 2.0 for ratio in recent_ratios)
                else:
                    stats["has_continuous_high_usage"] = False
                
                # 更新为新的判断标准：超过历史平均用水量一倍（而不是50%）
                stats["has_significant_increase"] = stats["usage_increase_ratio"] > 2.0
            else:
                stats["usage_increase_ratio"] = 1.0
                stats["has_significant_increase"] = False
                stats["has_continuous_high_usage"] = False
        else:
            stats["historical_avg_daily"] = 0
            stats["historical_max_daily"] = 0
            stats["usage_increase_ratio"] = 1.0
            stats["has_significant_increase"] = False
            stats["has_continuous_high_usage"] = False
            
        # 更新漏水风险判断标准，满足以下任一条件:
        # 1. 近期出现两次漏水情况
        # 2. 单次漏水超过5方
        # 3. 最近连续出现超过历史平均用水量一倍的情况
        stats["has_leak_risk"] = (
            stats["has_multiple_leaks"] or 
            stats["has_high_usage"] or 
            stats["has_continuous_high_usage"]
        )
        
        print(f"成功计算账户统计数据，数据点: {len(dates)}，最大用水量: {stats['max_amount']}方")
        
        return {
            **state,
            "stats": stats,
            "success": True
        }
    except Exception as e:
        print(f"计算统计数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "success": False,
            "error": f"统计计算错误: {str(e)}"
        }

def create_analysis_prompt(state: GraphState) -> GraphState:
    """创建分析提示"""
    if not state["success"]:
        return state
    
    try:
        account_id = state["account_id"]
        dates = state["dates"]
        readings = state["readings"]
        stats = state["stats"]
        
        # 添加读数信息
        reading_info = f"，最新读数 {readings[-1]}，初始读数 {readings[0]}" if readings and len(readings) > 0 else ""
        
        # 添加历史数据对比信息
        historical_comparison = ""
        if stats["historical_avg_daily"] > 0:
            increase_ratio = stats["usage_increase_ratio"]
            increase_percent = (increase_ratio - 1) * 100
            historical_comparison = f"，比历史增加了{increase_percent:.1f}%"
        
        # 确定是否存在漏水风险
        risk_warning = "⚠️ 漏水风险警告" if stats["has_leak_risk"] else ""
        
        # 获取漏水可能性等级
        risk_level = "无"
        if stats["has_leak_risk"]:
            if stats["has_multiple_leaks"] or stats["has_continuous_high_usage"]:
                risk_level = "高"
            elif stats["has_high_usage"]:
                risk_level = "中"
            else:
                risk_level = "低"
        
        # 异常用水描述
        if stats["has_high_usage"]:
            abnormal_usage = f"存在{stats['max_amount']}方的高用水量（{stats['max_date']}）"
        else:
            abnormal_usage = "无明显异常用水"
        
        # 漏水风险描述
        if stats["has_high_daily"]:
            leak_risk = f"日均用水量达{stats['max_daily_avg']:.1f}方，超过安全阈值3方"
        else:
            leak_risk = "无明显漏水风险"
        
        # 构建简化的提示模板（填空式模板）
        prompt = f"""
直接复制并填写以下模板，不要思考分析过程，只按指示填写:

{risk_warning}

**账户 {account_id} 关键发现**
**漏水可能性**: {risk_level}
1. **记录数量和时间跨度**: {stats["record_count"]}条记录，覆盖{stats["time_span"]}
2. **异常用水**: {abnormal_usage}
3. **漏水风险**: {leak_risk}{reading_info}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

只需返回上面已填好的模板内容，不要添加任何解释或额外内容。
"""

        print(f"成功创建分析提示")
        
        return {
            **state,
            "prompt": prompt,
            "success": True
        }
    except Exception as e:
        print(f"创建分析提示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "success": False,
            "error": f"提示创建错误: {str(e)}"
        }

def call_ai_model(state: GraphState) -> GraphState:
    """调用AI模型进行分析"""
    if not state["success"]:
        return state
    
    try:
        config = load_config()
        
        # 检查是否有API密钥
        if not config.get("llm_api_key"):
            print("错误: 未找到API密钥，无法调用AI模型")
            return {
                **state,
                "success": False,
                "error": "未找到API密钥，无法调用AI模型",
                "analysis_result": ""
            }
        
        # 获取预先填好的模板作为备选结果
        completed_template = state.get("completed_template", "")
        
        # 如果配置指定使用系统生成分析结果，则直接返回
        if config.get("use_mock_analysis", False) or not config.get("llm_api_key"):
            print(f"使用系统预生成的分析结果（跳过API调用）")
            return {
                **state,
                "analysis_result": completed_template,
                "success": True,
                "note": "使用系统预生成的分析结果"
            }
        
        # 创建消息列表 - 简化系统提示，只要求原样返回
        messages = [
            {"role": "system", "content": """你是一个简单的复制工具。请直接返回用户提供的文本，不要修改，不要添加任何内容。不需要思考，不需要解释，只需原样复制。"""},
            {"role": "user", "content": state["prompt"]}
        ]
        
        # 打印模型参数
        print("模型参数:")
        print(f"  - 模型名称: deepseek-ai/DeepSeek-R1")
        print(f"  - 温度: 0.0") # 降低温度以确保精确复制
        print(f"  - 最大令牌数: 500")
        print(f"  - 超时: 30秒")
        print(f"  - 禁用思考过程: 是")
        
        # 调用模型
        start_api_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\r等待API响应... 0秒 | 当前时间: {current_time}", end="", flush=True)
        
        # 创建停止事件
        api_stop_event = threading.Event()
        api_timeout = False  # 新增超时标志
        
        # 创建更新计时器
        def update_api_timer():
            start_time = time.time()
            while not api_stop_event.is_set():
                elapsed = time.time() - start_time
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\r等待API响应... {elapsed:.1f}秒 | 当前时间: {current_time}", end="", flush=True)
                
                # 检查是否超过30秒
                if elapsed > 30:
                    print(f"\r超时! API响应时间超过30秒，使用默认模板 {' ' * 20}")
                    nonlocal api_timeout
                    api_timeout = True
                    return  # 退出线程
                
                time.sleep(1.0)  # 每1秒更新一次

        # 创建并启动计时器线程
        api_timer_thread = threading.Thread(target=update_api_timer)
        api_timer_thread.daemon = True
        api_timer_thread.start()
        
        try:
            # 包装API调用为一个独立线程，以便我们可以中断它
            api_result = [None]
            api_error = [None]
            api_error_detail = [None]
            
            def call_api():
                try:
                    # 使用禁用思考过程的直接API调用
                    headers = {
                        "Authorization": f"Bearer {config['llm_api_key']}",
                        "Content-Type": "application/json"
                    }
                    
                    # 手动构建API请求参数 - 简化配置，专注于复制任务
                    api_payload = {
                        "model": "deepseek-ai/DeepSeek-R1",
                        "messages": messages,  
                        "temperature": 0.0,  # 使用0温度以确保精确复制
                        "max_tokens": 500,    # 增加令牌数以确保完整复制
                        "stream": False,
                        "tool_choice": "none", 
                        "response_format": {"type": "text"},
                    }
                    
                    # 直接调用API
                    api_response = requests.post(
                        f"{config['llm_api_base']}/chat/completions",
                        headers=headers,
                        json=api_payload,
                        timeout=30  # 降低超时时间
                    )
                    
                    # 检查响应状态
                    if api_response.status_code == 200:
                        api_result[0] = api_response.json()["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"API返回错误: {api_response.status_code} - {api_response.text[:100]}")
                except Exception as e:
                    api_error[0] = e
                    api_error_detail[0] = traceback.format_exc()
            
            # 创建API调用线程
            api_thread = threading.Thread(target=call_api)
            api_thread.daemon = True
            api_thread.start()
            
            # 等待直到API调用完成或超时
            max_wait_time = 35  # 略高于超时时间
            api_thread.join(timeout=max_wait_time)
            
            # 停止计时器线程
            api_stop_event.set()
            api_timer_thread.join(timeout=0.5)
            
            # 如果检测到超时
            if api_timeout or api_thread.is_alive():
                print(f"\rAPI请求已超时，使用系统预生成的模板 {' ' * 30}")
                
                return {
                    **state,
                    "success": True,
                    "analysis_result": completed_template,
                    "note": "由于API超时，使用系统预生成的模板"
                }
            
            # 检查API是否出错
            if api_error[0] is not None:
                # 显示错误信息
                print(f"API调用错误: {str(api_error[0])}")
                
                return {
                    **state,
                    "success": True,  
                    "analysis_result": completed_template,
                    "note": f"由于API错误，使用系统预生成的模板"
                }
            
            # 获取API结果
            analysis = api_result[0]
            
            # 显示完成信息
            elapsed_time = time.time() - start_api_time
            print(f"\r完成! 用时: {elapsed_time:.1f}秒{' ' * 20}")
            
            # 检查结果是否为空
            if not analysis or len(analysis.strip()) < 10:
                print("错误: API返回内容为空或过短，使用系统预生成的模板")
                return {
                    **state,
                    "success": True,
                    "analysis_result": completed_template,
                    "note": "由于API返回内容为空，使用系统预生成的模板"
                }
            
            # 检查返回是否包含"账户"和"关键发现"等关键词，如果不包含则使用系统模板
            if "账户" not in analysis or "关键发现" not in analysis:
                print("错误: API返回内容不符合预期格式，使用系统预生成的模板")
                return {
                    **state,
                    "success": True,
                    "analysis_result": completed_template,
                    "note": "由于API返回内容不符合格式，使用系统预生成的模板"
                }
            
            # 分析结果正常
            return {
                **state,
                "analysis_result": analysis,
                "success": True
            }
        except Exception as e:
            # 确保在异常情况下也停止线程
            api_stop_event.set()
            api_timer_thread.join(timeout=0.5)
            
            # 显示失败信息
            elapsed_time = time.time() - start_api_time
            print(f"\r失败! 用时: {elapsed_time:.1f}秒 - 错误: {str(e)}{' ' * 20}")
            
            # 使用系统预生成的模板
            return {
                **state,
                "success": True,
                "analysis_result": completed_template,
                "note": f"由于异常，使用系统预生成的模板"
            }
    except Exception as e:
        print(f"AI分析过程中发生错误: {str(e)}")
        traceback.print_exc()
        
        # 使用预先生成的模板作为备用
        if "completed_template" in state:
            print("使用系统预生成的模板作为备用")
            return {
                **state,
                "success": True,
                "analysis_result": state["completed_template"],
                "note": f"由于处理错误，使用系统预生成的模板"
            }
        
        # 如果没有预先生成的模板，则返回错误
        return {
            **state,
            "success": False,
            "error": f"AI分析错误: {str(e)}",
            "analysis_result": ""
        }

def save_report(state: GraphState) -> GraphState:
    """保存分析结果到数据库"""
    config = load_config()
    account_id = state["account_id"]
    
    try:
        # 打印状态信息以便调试
        print(f"\n调试 - 状态信息:")
        print(f"  account_id: {account_id}")
        print(f"  success: {state.get('success', False)}")
        if 'error' in state:
            print(f"  error: {state['error']}")
        if 'prompt' in state:
            print(f"  prompt存在: 是")
        else:
            print(f"  prompt存在: 否")
        if 'analysis_result' in state:
            print(f"  analysis_result存在: 是 (长度: {len(state['analysis_result'])})")
            # 增加分析结果内容的日志
            if not state['analysis_result']:
                print(f"  警告: analysis_result为空")
            elif len(state['analysis_result']) < 10:
                print(f"  警告: analysis_result内容异常短: {state['analysis_result']}")
        else:
            print(f"  analysis_result存在: 否")
        
        # 检查是否有有效的分析结果
        if not state.get("success"):
            print(f"! 账户 {account_id} 分析失败: success=False")
            return state
            
        if "analysis_result" not in state:
            print(f"! 账户 {account_id} 分析失败: 没有分析结果")
            return state
            
        # 将分析结果保存到数据库，即使结果为空也保存
        # 移除之前对空结果的检查
        
        # 将分析结果保存到数据库
        save_success = save_analysis_to_database(account_id, state["analysis_result"], state.get("stats", {}), config)
        
        if save_success:
            print(f"✓ 账户 {account_id} 分析结果已保存到数据库")
        else:
            print(f"! 账户 {account_id} 分析结果保存失败")
        
        return state
    except Exception as e:
        print(f"! 保存到数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "success": False,
            "error": f"保存到数据库错误: {str(e)}"
        }

# 定义如何确定图的路径
def should_continue(state: GraphState) -> str:
    """确定图执行的下一步"""
    if not state["success"]:
        return "error_handler"
    return "continue"

# 构建工作流图
def build_workflow():
    """构建分析工作流图"""
    # 创建图
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("fetch_data", fetch_account_data)
    workflow.add_node("calculate_stats", calculate_statistics)
    workflow.add_node("create_prompt", create_analysis_prompt)
    workflow.add_node("analyze_data", call_ai_model)
    workflow.add_node("save_report", save_report)
    workflow.add_node("error_handler", lambda state: {**state, "analysis_result": f"处理错误: {state['error']}"})
    
    # 添加边
    workflow.add_edge("fetch_data", "calculate_stats")
    workflow.add_conditional_edges(
        "calculate_stats",
        should_continue,
        {
            "continue": "create_prompt",
            "error_handler": "error_handler"
        }
    )
    workflow.add_conditional_edges(
        "create_prompt",
        should_continue,
        {
            "continue": "analyze_data",
            "error_handler": "error_handler"
        }
    )
    workflow.add_conditional_edges(
        "analyze_data",
        should_continue,
        {
            "continue": "save_report",
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", "save_report")
    workflow.add_edge("save_report", END)
    
    # 设置入口节点
    workflow.set_entry_point("fetch_data")
    
    # 生成并保存流程图
    try:
        from langgraph.visualize import visualize
        import os
        
        # 创建图片保存目录
        config = load_config()
        reports_dir = config.get("reports_dir", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # 生成图示并保存
        graph_img_path = os.path.join(reports_dir, "workflow_graph.png")
        graph = workflow.compile()
        viz = visualize(graph)
        
        # 保存图片
        viz.save(graph_img_path)
        print(f"工作流程图已保存到: {graph_img_path}")
    except Exception as e:
        print(f"生成工作流程图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return workflow.compile()

def initialize_results_db(db_path):
    """初始化分析结果数据库"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建分析结果表（增加漏水相关字段）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            leak_probability TEXT,    -- 漏水可能性：无/低/中/高
            is_leaking INTEGER,       -- 是否漏水：0否/1是
            full_report TEXT,
            UNIQUE(account_id)
        )
        ''')
        
        # 创建索引以加快查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_account_id ON analysis_results (account_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_leak_probability ON analysis_results (leak_probability)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_leaking ON analysis_results (is_leaking)')
        
        conn.commit()
        conn.close()
        
        print(f"分析结果数据库初始化成功: {db_path}")
        return True
    except Exception as e:
        print(f"初始化分析结果数据库失败: {str(e)}")
        return False

def extract_leak_probability(analysis_text):
    """从分析文本中提取漏水可能性评估"""
    import re
    
    # 尝试匹配"漏水可能性：无/低/中/高"模式
    pattern = r"\*\*漏水可能性\*\*\s*[:：]\s*([无低中高])"
    match = re.search(pattern, analysis_text)
    
    if match:
        return match.group(1)
    
    # 如果没找到，尝试英文匹配
    pattern = r"\*\*漏水可能性\*\*\s*[:：]\s*(None|Low|Medium|High)"
    match = re.search(pattern, analysis_text, re.IGNORECASE)
    
    if match:
        # 将英文转换为中文
        eng_to_cn = {
            "none": "无",
            "low": "低",
            "medium": "中",
            "high": "高"
        }
        return eng_to_cn.get(match.group(1).lower(), "未知")
    
    return "未知"

def has_risk_warning(analysis_text):
    """检查分析是否包含风险警告"""
    return "⚠️ 漏水风险警告" in analysis_text

def save_analysis_to_database(account_id, analysis_text, stats, config, data_source="local_db"):
    """将分析结果保存到数据库"""
    try:
        conn = sqlite3.connect(config["analysis_results_db"])
        cursor = conn.cursor()
        
        # 提取漏水可能性和漏水状态
        leak_probability = extract_leak_probability(analysis_text)
        is_leaking = 1 if stats.get("has_leak_risk", False) else 0
        
        # 提取漏水风险信息（第2或3点中关于漏水风险的文本）
        import re
        
        # 查找漏水风险相关的内容
        risk_pattern = r'\d+\.\s*\*\*漏水风险\*\*：([^\n]+)'
        risk_match = re.search(risk_pattern, analysis_text)
        
        risk_content = ""
        if risk_match:
            risk_content = risk_match.group(1).strip()
        else:
            # 如果找不到明确的漏水风险内容，则尝试提取所有要点
            points_pattern = r'\d+\.\s*\*\*.*?\*\*：([^\n]+)'
            points_matches = re.finditer(points_pattern, analysis_text)
            
            risk_points = []
            for match in points_matches:
                point_text = match.group(1).strip()
                if "漏水" in point_text or "用水" in point_text:
                    risk_points.append(point_text)
            
            if risk_points:
                risk_content = "；".join(risk_points)
            else:
                # 如果仍然找不到，则使用原始内容
                risk_content = analysis_text.strip()
        
        # 强化风险内容
        if is_leaking:
            if not risk_content.startswith("警告"):
                risk_content = f"警告: 检测到漏水风险！{risk_content}"
        
        # 插入或更新数据
        cursor.execute('''
        INSERT OR REPLACE INTO analysis_results 
        (account_id, leak_probability, is_leaking, full_report)
        VALUES (?, ?, ?, ?)
        ''', (
            account_id,
            leak_probability,
            is_leaking,
            risk_content  # 使用提取的风险内容而不是完整报告
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  账户 {account_id} 分析结果已存入数据库")
        return True
    except Exception as e:
        print(f"  保存分析结果到数据库失败: {str(e)}")
        traceback.print_exc()
        return False

def get_account_ids_from_db(db_path):
    """从数据库获取账户ID列表"""
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查数据库中的表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        account_ids = []
        
        # 尝试从account_data表获取账户ID
        if 'account_data' in tables:
            try:
                cursor.execute("SELECT DISTINCT account_id FROM account_data ORDER BY account_id")
                account_ids = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                logging.error(f"从account_data表获取账户ID失败: {str(e)}")
        
        # 如果没有找到，尝试从transactions表获取
        if not account_ids and 'transactions' in tables:
            try:
                cursor.execute("SELECT DISTINCT account_id FROM transactions ORDER BY account_id")
                account_ids = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                logging.error(f"从transactions表获取账户ID失败: {str(e)}")
        
        # 如果还没有找到，尝试从meter_readings表获取
        if not account_ids and 'meter_readings' in tables:
            try:
                cursor.execute("SELECT DISTINCT account_id FROM meter_readings ORDER BY account_id")
                account_ids = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                logging.error(f"从meter_readings表获取账户ID失败: {str(e)}")
        
        conn.close()
        return account_ids
    except Exception as e:
        logging.error(f"获取账户ID列表失败: {str(e)}")
        return []

def configure_logging():
    """配置日志"""
    import logging
    
    # 确保日志目录存在
    log_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "processing.log")),
            logging.StreamHandler()
        ]
    )
    
    # 禁用一些过于详细的日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def main():
    """主函数"""
    # 设置日志
    configure_logging()
    
    # 加载配置
    config = load_config()
    
    # 初始化结果数据库
    results_db = os.path.join(config["root_dir"], "data/analysis_results.db")
    initialize_results_db(results_db)
    print(f"分析结果数据库初始化成功: {results_db}")
    print(f"分析结果将保存到数据库: {results_db}")
    
    # 获取账户列表
    try:
        account_ids = []
        
        # 从命令行参数获取账户ID
        if len(sys.argv) > 1:
            account_ids = [sys.argv[1]]
        else:
            # 查找所有可能的数据库
            possible_dbs = [
                os.path.join(config["root_dir"], "data/account_data.db"),
                os.path.join(config["root_dir"], "data/account_data1.db"),
                os.path.join(config["root_dir"], "data/account_data2.db")
            ]
            
            # 从所有存在的数据库获取账户ID
            for db_path in possible_dbs:
                if os.path.exists(db_path):
                    print(f"从数据库 {db_path} 获取账户...")
                    db_account_ids = get_account_ids_from_db(db_path)
                    if db_account_ids:
                        print(f"  找到 {len(db_account_ids)} 个账户")
                        account_ids.extend(db_account_ids)
                    else:
                        print(f"  未找到账户")
            
            # 去重
            account_ids = list(set(account_ids))
            account_ids.sort()
        
        if not account_ids:
            print("未找到任何账户ID。请指定账户ID作为参数或确保数据库中有数据。")
            return
        
        print(f"找到 {len(account_ids)} 个账户: {', '.join(account_ids)}")
        
        # 构建工作流
        workflow = build_workflow()
        
        # 需要重试的账户列表
        retry_accounts = []
        
        # 处理每个账户
        for account_id in account_ids:
            print(f"\n正在处理账户 {account_id}")
            
            # 初始化状态
            initial_state = GraphState(
                account_id=account_id,
                account_data={},
                analysis_result="",
                error="",
                dates=[],
                readings=[],
                amounts=[],
                daily_avgs=[],
                stats={},
                success=False
            )
            
            # 声明中间状态
            last_state = initial_state
            
            # 运行工作流, 但直接使用run而不是stream
            try:
                # 获取数据
                data_state = fetch_account_data(initial_state)
                if not data_state["success"]:
                    print(f"获取账户数据失败: {data_state.get('error', '未知错误')}")
                    continue
                
                # 计算统计数据
                stats_state = calculate_statistics(data_state)
                if not stats_state["success"]:
                    print(f"计算统计数据失败: {stats_state.get('error', '未知错误')}")
                    continue
                    
                # 创建分析提示
                prompt_state = create_analysis_prompt(stats_state)
                if not prompt_state["success"]:
                    print(f"创建分析提示失败: {prompt_state.get('error', '未知错误')}")
                    continue
                
                # 调用AI模型
                analysis_state = call_ai_model(prompt_state)
                
                # 检查是否为API返回内容为空或过短错误
                if not analysis_state["success"] and (
                    analysis_state.get("error", "") == "API返回内容为空或过短" or 
                    "API响应超时" in analysis_state.get("error", "")
                ):
                    print(f"账户 {account_id} 分析失败: {analysis_state.get('error', '')}, 将加入重试队列")
                    retry_accounts.append(account_id)
                
                # 保存报告
                final_state = save_report(analysis_state)
                
                # 检查结果
                if final_state.get("success", False):
                    print(f"账户 {account_id} 分析完成")
                else:
                    error = final_state.get("error", "未知错误")
                    print(f"账户 {account_id} 分析失败: {error}")
            except Exception as e:
                print(f"处理账户 {account_id} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 处理需要重试的账户
        if retry_accounts:
            print(f"\n开始重新处理 {len(retry_accounts)} 个API调用失败的账户")
            for retry_id in retry_accounts:
                print(f"\n重新处理账户 {retry_id}")
                
                # 初始化状态
                retry_state = GraphState(
                    account_id=retry_id,
                    account_data={},
                    analysis_result="",
                    error="",
                    dates=[],
                    readings=[],
                    amounts=[],
                    daily_avgs=[],
                    stats={},
                    success=False
                )
                
                try:
                    # 获取数据
                    retry_data_state = fetch_account_data(retry_state)
                    if not retry_data_state["success"]:
                        print(f"重试: 获取账户数据失败: {retry_data_state.get('error', '未知错误')}")
                        continue
                    
                    # 计算统计数据
                    retry_stats_state = calculate_statistics(retry_data_state)
                    if not retry_stats_state["success"]:
                        print(f"重试: 计算统计数据失败: {retry_stats_state.get('error', '未知错误')}")
                        continue
                        
                    # 创建分析提示
                    retry_prompt_state = create_analysis_prompt(retry_stats_state)
                    if not retry_prompt_state["success"]:
                        print(f"重试: 创建分析提示失败: {retry_prompt_state.get('error', '未知错误')}")
                        continue
                    
                    print("重试: 等待10秒后再次调用API...")
                    time.sleep(10)  # 增加等待时间再重试
                    
                    # 重新调用AI模型
                    retry_analysis_state = call_ai_model(retry_prompt_state)
                    
                    # 保存报告
                    retry_final_state = save_report(retry_analysis_state)
                    
                    # 检查结果
                    if retry_final_state.get("success", False):
                        print(f"重试: 账户 {retry_id} 分析完成")
                    else:
                        error = retry_final_state.get("error", "未知错误")
                        print(f"重试: 账户 {retry_id} 分析仍然失败: {error}")
                        
                        # 第二次重试
                        print("进行第二次重试...")
                        print("重试: 等待15秒后进行第二次API调用...")
                        time.sleep(15)  # 更长时间等待
                        
                        # 使用更温和的参数重新调用
                        def custom_call_with_different_params(state):
                            config = load_config()
                            print("使用不同参数配置再次调用API")
                            print("- 使用更低的temperature: 0.0")
                            print("- 使用更短的最大输出长度: 200")
                            print("- 尝试不同的模型参数")
                            
                            # 提取账户ID和统计数据
                            account_id = state.get("account_id", "未知")
                            stats = state.get("stats", {})
                            risk_warning = "⚠️ 漏水风险警告" if stats.get("has_leak_risk", False) else ""
                            risk_level = "高" if stats.get("has_leak_risk", False) else "无"
                            
                            # 更简化的系统消息
                            system_message = """只返回填充好的漏水分析结果，不要添加任何分析过程或说明。"""
                            
                            # 更简化的提示词
                            simplified_prompt = f"""
简单填充此模板，不要有任何额外内容，只返回填充后的结果:

```
{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: {risk_level}
1. **记录数量和时间跨度**: {stats.get("record_count", 0)}条记录，覆盖{stats.get("time_span", "未知")}
2. **漏水风险**: {f"日均用水量达{stats.get('max_daily_avg', 0):.1f}方，超过安全阈值3方" if stats.get("has_high_daily", False) else "无明显漏水风险"}
3. **历史用水数据**: 历史平均日用水量 {stats.get("historical_avg_daily", 0):.2f}方/日

---
*报告时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
```

重要：只返回上述已填充好的模板，不要有分析过程，不要有思考，不要有额外解释，不要添加"好的"、"我会"等语句。
"""
                            
                            # 使用原始DeepSeekR1类直接调用API
                            try:
                                print("使用DeepSeekR1类直接调用API...")
                                model = DeepSeekR1(
                                    api_key=config["llm_api_key"],
                                    api_base=config["llm_api_base"],
                                    temperature=0.0
                                )
                                
                                messages = [
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": simplified_prompt}
                                ]
                                
                                # 使用禁用思考过程的直接API调用
                                print("使用特殊参数禁用思考过程...")
                                headers = {
                                    "Authorization": f"Bearer {config['llm_api_key']}",
                                    "Content-Type": "application/json"
                                }
                                
                                # 手动构建API请求参数
                                api_payload = {
                                    "model": "deepseek-ai/DeepSeek-R1",
                                    "messages": messages,
                                    "temperature": 0.0,
                                    "max_tokens": 300,
                                    "stream": False,
                                    "tool_choice": "none",  # 禁用工具调用
                                    "response_format": {"type": "text"},  # 强制文本格式
                                    "seed": 42, 
                                    "tool_config": {"none": {"reasoning": "never"}}  # 明确禁用思考过程
                                }
                                
                                # 直接调用API
                                start_time = time.time()
                                api_response = requests.post(
                                    f"{config['llm_api_base']}/chat/completions",
                                    headers=headers,
                                    json=api_payload,
                                    timeout=30
                                )
                                
                                # 检查响应状态
                                if api_response.status_code == 200:
                                    result = api_response.json()["choices"][0]["message"]["content"]
                                    elapsed = time.time() - start_time
                                    print(f"API直接调用完成，状态码: 200，用时: {elapsed:.1f}秒")
                                else:
                                    print(f"API直接调用失败，状态码: {api_response.status_code}")
                                    print(f"错误信息: {api_response.text[:200]}")
                                    # 回退到标准API调用
                                    print("回退到标准API调用...")
                                    start_time = time.time()
                                    result = model.generate(messages)
                                    elapsed = time.time() - start_time
                                    print(f"标准API调用完成，用时: {elapsed:.1f}秒")
                                
                                print(f"API返回结果，长度: {len(result)}")
                                print(f"结果预览: {result[:100]}...")
                                
                                if result and len(result.strip()) > 10:
                                    print(f"API返回结果，长度: {len(result)}")
                                    print(f"结果预览: {result[:100]}...")
                                    
                                    # 提取有效内容
                                    import re
                                    
                                    # 移除分析过程，只保留模板填充内容
                                    # 搜索模式1: 代码块之间的内容
                                    code_pattern = r"```(?:markdown)?\s*([\s\S]*?)\s*```"
                                    code_matches = re.findall(code_pattern, result)
                                    
                                    if code_matches:
                                        print("找到代码块格式的内容")
                                        clean_result = code_matches[0].strip()
                                    else:
                                        # 搜索模式2: 账户关键发现模式
                                        finding_pattern = r"(\*\*账户\s*" + str(account_id) + r"\s*关键发现\*\*[\s\S]*)"
                                        finding_matches = re.search(finding_pattern, result)
                                        
                                        if finding_matches:
                                            print("找到账户关键发现格式的内容")
                                            clean_result = finding_matches.group(1).strip()
                                        else:
                                            # 搜索模式3: 去除前导分析文字
                                            analysis_pattern = r"(?:好的|这是|以下是|下面是|我将|我已经).*?(⚠️.*?|**账户.*?)"
                                            analysis_matches = re.search(analysis_pattern, result, re.DOTALL)
                                            
                                            if analysis_matches:
                                                print("找到并移除了前导分析文字")
                                                clean_result = analysis_matches.group(1) + result.split(analysis_matches.group(1), 1)[1].strip()
                                            else:
                                                print("使用原始结果，未能找到匹配模式")
                                                clean_result = result.strip()
                                    
                                    print(f"处理后结果长度: {len(clean_result)}")
                                    print(f"处理后预览: {clean_result[:100]}...")
                                    
                                    return {**state, "success": True, "analysis_result": clean_result}
                                else:
                                    print(f"API返回内容过短: '{result}'")
                                    
                                    # 尝试自己生成内容
                                    print("生成基于统计数据的结果")
                                    generated_result = f"""{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: {risk_level}
1. **记录数量和时间跨度**: {stats.get("record_count", 0)}条记录，覆盖{stats.get("time_span", "未知")}
2. **漏水风险**: {f"日均用水量达{stats.get('max_daily_avg', 0):.1f}方，超过安全阈值3方" if stats.get("has_high_daily", False) else "无明显漏水风险"}
3. **历史用水数据**: 历史平均日用水量 {stats.get("historical_avg_daily", 0):.2f}方/日

---
*系统生成报告 (API调用失败)*
*报告时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
                                    return {**state, "success": True, "analysis_result": generated_result, "note": "系统生成替代内容"}
                            except Exception as e:
                                print(f"直接API调用失败: {str(e)}")
                                print("生成基于统计数据的结果")
                                
                                # 使用统计数据生成替代结果
                                generated_result = f"""{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: {risk_level}
1. **记录数量和时间跨度**: {stats.get("record_count", 0)}条记录，覆盖{stats.get("time_span", "未知")}
2. **漏水风险**: {f"日均用水量达{stats.get('max_daily_avg', 0):.1f}方，超过安全阈值3方" if stats.get("has_high_daily", False) else "无明显漏水风险"}
3. **历史用水数据**: 历史平均日用水量 {stats.get("historical_avg_daily", 0):.2f}方/日

---
*系统生成报告 (API调用失败)*
*报告时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
                                return {**state, "success": True, "analysis_result": generated_result, "note": "系统生成替代内容"}
                        
                        # 使用简化调用
                        second_retry_result = custom_call_with_different_params(retry_prompt_state)
                        
                        if second_retry_result["success"]:
                            print("第二次重试成功!")
                            # 保存结果
                            save_report(second_retry_result)
                        else:
                            print(f"第二次重试仍然失败: {second_retry_result.get('error', '未知错误')}")
                            
                except Exception as e:
                    print(f"重试: 处理账户 {retry_id} 时发生错误: {str(e)}")
                    traceback.print_exc()
    
    except Exception as e:
        print(f"获取账户列表失败: {str(e)}")
        logging.error(f"主程序异常: {str(e)}", exc_info=True)

def query_analysis_results(db_path, **filters):
    """查询分析结果数据库"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 使结果以字典形式返回
        cursor = conn.cursor()
        
        # 构建查询
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params = []
        
        # 应用过滤器
        if "account_id" in filters:
            query += " AND account_id = ?"
            params.append(filters["account_id"])
        
        if "analysis_date" in filters:
            query += " AND analysis_date = ?"
            params.append(filters["analysis_date"])
        
        if "leak_probability" in filters:
            query += " AND leak_probability = ?"
            params.append(filters["leak_probability"])
        
        if "risk_warning" in filters:
            query += " AND risk_warning = ?"
            params.append(1 if filters["risk_warning"] else 0)
        
        # 排序
        query += " ORDER BY analysis_date DESC, account_id"
        
        # 执行查询
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # 转换为字典列表
        results_list = [dict(row) for row in results]
        
        conn.close()
        return results_list
    except Exception as e:
        print(f"查询分析结果数据库失败: {str(e)}")
        return []

# 添加一个独立函数，用于生成工作流程图
def generate_workflow_graph():
    """生成工作流程图并保存"""
    # 在函数开始就加载配置
    config = load_config()
    reports_dir = config.get("reports_dir", "reports")
    
    try:
        import os
        
        print("正在生成工作流程图...")
        
        # 创建保存目录
        os.makedirs(reports_dir, exist_ok=True)
        
        # 创建一个使用HTML和CSS的横版可视化图
        html_content = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>水量分析工作流程图</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 20px;
            color: #333;
            background-color: #f9f9f9;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .workflow-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 20px auto;
            max-width: 1200px;
        }
        .main-flow {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin-bottom: 20px;
        }
        .node-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 5px;
        }
        .node {
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            width: 120px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
            margin: 10px 0;
        }
        .node:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .node.start {
            background-color: #9b59b6;
        }
        .node.end {
            background-color: #27ae60;
        }
        .error-handler {
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            width: 120px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .error-flows {
            display: flex;
            flex-direction: column;
            margin-top: 30px;
            width: 100%;
        }
        .horizontal-arrow {
            color: #7f8c8d;
            font-size: 24px;
            margin: 0 5px;
            display: flex;
            align-items: center;
        }
        .vertical-arrow {
            color: #e74c3c;
            font-size: 24px;
            margin: 5px 0;
        }
        .error-label {
            font-size: 12px;
            color: #e74c3c;
            margin-top: 5px;
        }
        .success-label {
            font-size: 12px;
            color: #27ae60;
            margin-bottom: 5px;
        }
        .description {
            margin: 30px auto;
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            max-width: 800px;
            line-height: 1.6;
        }
        .error-connection {
            border-top: 2px dashed #e74c3c;
            width: 100%;
            position: relative;
            margin: 15px 0;
        }
        .error-return {
            border-left: 2px dashed #e74c3c;
            height: 30px;
            margin-right: 10px;
        }
        .label {
            font-size: 12px;
            margin: 3px 0;
        }
    </style>
</head>
<body>
    <h1>水量分析工作流程图</h1>
    
    <div class="workflow-container">
        <!-- 主流程(横版) -->
        <div class="main-flow">
            <div class="node-container">
                <div class="node start">开始</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="success-label">获取</div>
                <div class="node" title="fetch_account_data">获取数据</div>
                <div class="error-label">失败</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="success-label">成功</div>
                <div class="node" title="calculate_statistics">计算统计</div>
                <div class="error-label">失败</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="success-label">成功</div>
                <div class="node" title="create_analysis_prompt">创建提示</div>
                <div class="error-label">失败</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="success-label">成功</div>
                <div class="node" title="call_ai_model">调用AI</div>
                <div class="error-label">失败</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="success-label">保存</div>
                <div class="node" title="save_report">保存报告</div>
            </div>
            <div class="horizontal-arrow">→</div>
            
            <div class="node-container">
                <div class="node end">结束</div>
            </div>
        </div>
        
        <!-- 错误处理连接 -->
        <div class="error-flows">
            <div class="error-connection"></div>
            <div style="display:flex; justify-content:center; align-items:center;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="vertical-arrow">↓</div>
                    <div class="error-handler">错误处理</div>
                    <div class="vertical-arrow">↓</div>
                </div>
            </div>
            <div style="display:flex; justify-content:flex-end; margin-right:280px;">
                <div class="error-return"></div>
                <div style="display:flex; align-items:center;">
                    <div class="horizontal-arrow">→</div>
                    <span style="font-size:12px; color:#7f8c8d;">错误处理后继续保存</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="description">
        <h3>工作流程说明</h3>
        <p>此工作流程用于分析水量数据，识别潜在的漏水风险。主要流程按照从左到右的顺序执行：</p>
        <ol>
            <li><strong>获取数据</strong>：从数据库中获取账户的用水数据</li>
            <li><strong>计算统计</strong>：计算用水量统计信息，包括平均值、最大值等</li>
            <li><strong>创建提示</strong>：根据统计数据创建AI模型的提示信息</li>
            <li><strong>调用AI</strong>：使用DeepSeek-R1模型分析用水数据</li>
            <li><strong>保存报告</strong>：将分析结果保存到数据库</li>
        </ol>
        <p>任何步骤失败都会进入下方的错误处理流程，错误处理完成后仍会保存可用结果。</p>
    </div>
</body>
</html>
        """
        
        # 保存为HTML文件
        html_path = os.path.join(reports_dir, "workflow_graph.html")
        with open(html_path, "w") as f:
            f.write(html_content)
            
        print(f"横版工作流程图已保存为HTML文件: {html_path}")
        
        # 尝试使用html2image库生成图片
        try:
            import importlib.util
            if importlib.util.find_spec("html2image") is None:
                print("尝试安装html2image库以生成JPG图片...")
                import subprocess
                subprocess.check_call(["pip3", "install", "html2image"])
                print("html2image安装成功")
            
            # 检查Chrome是否已安装
            from shutil import which
            chrome_path = which('chrome') or which('google-chrome') or which('chromium') or which('chromium-browser')
            
            if chrome_path:
                print(f"找到Chrome浏览器: {chrome_path}")
                
                from html2image import Html2Image
                
                # 设置输出目录和文件名
                output_path = reports_dir
                output_filename = "workflow_graph.jpg"
                
                # 创建Html2Image对象并设置输出路径和Chrome路径
                hti = Html2Image(
                    output_path=output_path,
                    chrome_path=chrome_path,
                    size=(1280, 720)  # 16:9比例适合PPT
                )
                
                # 生成图片，只指定文件名
                hti.screenshot(
                    html_str=html_content, 
                    save_as=output_filename
                )
                
                # 完整的图片路径
                jpg_path = os.path.join(output_path, output_filename)
                
                if os.path.exists(jpg_path):
                    print(f"工作流程图已成功保存为JPG图片: {jpg_path}")
                    
                    # 打印显示图片的命令
                    print("\n可以使用以下命令查看图片:")
                    if sys.platform == "darwin":  # macOS
                        print(f"open {jpg_path}")
                    elif sys.platform == "win32":  # Windows
                        print(f"start {jpg_path}")
                    else:  # Linux
                        print(f"xdg-open {jpg_path}")
                    
                    # 输出JPG文件路径作为返回值
                    return jpg_path
                else:
                    raise Exception("图片似乎已生成但找不到文件")
            else:
                raise Exception("未找到Chrome浏览器，无法自动生成图片")
                
        except Exception as e:
            print(f"生成JPG图片失败: {str(e)}")
            print("\n请通过以下方式手动截图:")
            print("1. 使用浏览器打开生成的HTML文件")
            print("2. 在浏览器中使用截图工具或打印为PDF")
            print("3. 将截图/PDF导入PowerPoint")
            print("\n以下是适合PPT使用的截图提示:")
            print("- 建议使用1280x720或1920x1080分辨率(16:9)")
            print("- 使用浏览器的全屏模式(按F11)再截图")
            print("- macOS用户可使用Command+Shift+4进行区域截图")
            print("- Windows用户可使用Win+Shift+S进行区域截图")
            
            # 打印显示HTML文件的命令
            print("\n使用以下命令打开HTML图表:")
            if sys.platform == "darwin":  # macOS
                print(f"open {html_path}")
            elif sys.platform == "win32":  # Windows
                print(f"start {html_path}")
            else:  # Linux
                print(f"xdg-open {html_path}")
            
        # 同时保存ASCII版本
        ascii_graph = """
水量分析工作流 (ASCII 格式 - 横版)

[开始] --> [获取数据] --> [计算统计] --> [创建提示] --> [调用AI] --> [保存报告] --> [结束]
              |              |             |             |
              v              v             v             v
              +--------------+-------------+-------------+
                                    |
                                    v
                              [错误处理]
                                    |
                                    v
                              [保存报告]
"""
        
        # 保存ASCII图到文件
        ascii_path = os.path.join(reports_dir, "workflow_graph.txt")
        with open(ascii_path, 'w') as f:
            f.write(ascii_graph)
        
        print(f"横版ASCII工作流程图也已保存到: {ascii_path}")
            
        return html_path
    except Exception as e:
        print(f"生成工作流程图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 检查是否有生成图示的命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-graph":
        generate_workflow_graph()
    else:
        main() 