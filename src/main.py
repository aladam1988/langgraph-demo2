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
    
    def generate(self, messages):
        """生成回复"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备请求体
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 800
        }
        
        logger.info(f"发送请求到DeepSeek API...")
        
        # 开始计时
        start_time = time.time()
        
        # 创建一个事件来控制计时器线程
        stop_event = threading.Event()
        
        def print_elapsed_time():
            # 每秒只更新一次，降低输出频率
            start_time = time.time()
            for i in range(120):  # 最多等待120秒
                if stop_event.is_set():
                    return
                elapsed = time.time() - start_time
                # 修改这里，添加当前的时间显示
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\r等待API响应... {elapsed:.1f}秒 | 当前时间: {current_time}", end="", flush=True)
                time.sleep(0.5)  # 每0.5秒更新一次
        
        # 创建并启动计时器线程
        timer_thread = threading.Thread(target=print_elapsed_time)
        timer_thread.daemon = True
        timer_thread.start()
        
        try:
            # 使用较长的超时时间
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=120  # 给API较多响应时间
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
            
            # 准备消息，使用更强制性的系统提示
            messages = [
                SystemMessage(content="""你是一个简洁高效的水量数据分析工具, 只输出分析结论:
1. 你只会输出格式化的分析结论, 绝对不会解释你的分析过程
2. 你永远不会写出"我分析了", "根据数据", "通过分析"等说明分析过程的语句
3. 你只关注用户提供的格式模板, 并在这个模板中填充你的结论
4. 你应当拒绝解释你是如何得出结论的
5. 你的回答必须极其简洁直接, 不包含任何多余内容"""),
                HumanMessage(content=prompt)
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
填充以下用水分析模板，不要添加任何额外内容：

```
{("⚠️ 漏水风险警告" if has_leak_risk else "")}
漏水可能性：（无/低/中/高）
**账户 {account_id} 关键发现**
1. **记录数量和时间跨度**：{len(dates)}条记录，覆盖{time_span}
2. **异常用水**：{f"存在{max_amount}方的高用水量（{max_date}）" if has_high_usage else "无明显异常用水"}
3. **漏水风险**：{f"日均用水量达{max_daily_avg:.1f}方，超过安全阈值3方" if has_high_daily else "无明显漏水风险"}{reading_info}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
```

提示：
- 账户ID: {account_id}
- 读表时间: {dates[-1] if dates else "未知"}
- 当前读数: {readings[-1] if readings else "未知"}
- 居民日均用水>1方或连续增长并超过3方表示漏水风险
- 商业用水突增1倍以上表示漏水风险
- 仅返回模板内容，不添加任何分析过程
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
        cursor1.execute(
            """
            SELECT 
                reading_time,
                current_reading,
                current_usage,
                daily_average
            FROM meter_readings 
            WHERE account_id = ? 
            ORDER BY reading_time
            """, 
            (account_id,)
        )
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
            daily_avgs.append(row[3])
        
        # 关闭连接
        conn1.close()
        
        # 连接数据库2 (account_data2.db)
        conn2 = sqlite3.connect(config["historical_data_db"])
        cursor2 = conn2.cursor()
        
        # 获取最近半年的历史抄表数据
        from datetime import datetime, timedelta
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y/%m/%d")
        
        cursor2.execute(
            """
            SELECT 
                reading_time,
                reading_value
            FROM historical_readings 
            WHERE account_id = ? AND reading_time >= ?
            ORDER BY reading_time
            """, 
            (account_id, six_months_ago)
        )
        
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
        reading_info = f"\n   最新读数 {readings[-1]}，初始读数 {readings[0]}" if readings and len(readings) > 0 else ""
        
        # 添加历史数据对比信息
        historical_comparison = ""
        if stats["historical_avg_daily"] > 0:
            increase_ratio = stats["usage_increase_ratio"]
            increase_percent = (increase_ratio - 1) * 100
            historical_comparison = f"，比历史增加了{increase_percent:.1f}%"
        
        # 确定是否存在漏水风险
        risk_warning = "⚠️ 漏水风险警告" if stats["has_leak_risk"] else ""
        
        # 构建提示模板
        prompt = f"""
请分析以下用水数据，并填充水用量分析报告模板：

## 用户信息
- 账户ID: {account_id}
- 读表时间: {dates[-1] if dates else "未知"}
- 当前读数: {readings[-1] if readings else "未知"}
- 记录数量: {stats["record_count"]}条
- 时间跨度: {stats["time_span"]}

## 用水数据
- 最大用水量: {stats["max_amount"]}方 (发生于{stats["max_date"]})
- 最小用水量: {stats["min_amount"]}方
- 平均用水量: {stats["avg_amount"]:.2f}方
- 最大日均用水量: {stats["max_daily_avg"]:.2f}方
- 历史半年日均用水量: {stats["historical_avg_daily"]:.2f}方
- 当前与历史用水量比率: {stats["usage_increase_ratio"]:.2f}

## 异常标准
- 近期出现两次漏水情况（日均用水>3方）
- 单次漏水超过5方
- 最近连续出现超过历史平均用水量一倍的情况

## 填充以下模板（无需解释分析过程）:

```
{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: [请基于上述数据，判断此账户漏水可能性为：无、低、中、高中的一个]
1. **记录数量和时间跨度**：{stats["record_count"]}条记录，覆盖{stats["time_span"]}
2. **异常用水**：{f"存在{stats['max_amount']}方的高用水量（{stats['max_date']}）" if stats["has_high_usage"] else "无明显异常用水"}
3. **漏水风险**：{f"日均用水量达{stats['max_daily_avg']:.1f}方，超过安全阈值3方" if stats["has_high_daily"] else "无明显漏水风险"}{reading_info}
4. **历史用水数据**：历史平均日用水量 {stats["historical_avg_daily"]:.2f}方/日{historical_comparison}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
```
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
            print("警告: 未找到API密钥，无法调用AI模型，返回模拟分析")
            # 返回模拟分析
            account_id = state["account_id"]
            stats = state.get("stats", {})
            
            risk_warning = "⚠️ 漏水风险警告" if stats.get("has_leak_risk", False) else ""
            reading_info = f"\n   最新读数 {state['readings'][-1]}，初始读数 {state['readings'][0]}" if state['readings'] and len(state['readings']) > 0 else ""
            
            mock_analysis = f"""
{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: {"高" if stats.get("has_leak_risk", False) else "无"}
1. **记录数量和时间跨度**：{stats.get("record_count", 0)}条记录，覆盖{stats.get("time_span", "未知")}
2. **异常用水**：{f"存在{stats.get('max_amount', 0)}方的高用水量（{stats.get('max_date', '未知')}）" if stats.get("has_high_usage", False) else "无明显异常用水"}
3. **漏水风险**：{f"日均用水量达{stats.get('max_daily_avg', 0):.1f}方，超过安全阈值3方" if stats.get("has_high_daily", False) else "无明显漏水风险"}{reading_info}
4. **历史用水数据**：历史平均日用水量 {stats.get("historical_avg_daily", 0):.2f}方/日

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
            
            print("使用模拟分析结果完成")
            return {
                **state,
                "analysis_result": mock_analysis,
                "success": True
            }
        
        # 创建LangChain模型
        model = ChatOpenAI(
            openai_api_key=config["llm_api_key"],
            openai_api_base=config["llm_api_base"],
            model_name="deepseek-ai/DeepSeek-R1",
            temperature=0.1,
            max_tokens=800,
            timeout=120
        )
        
        # 创建消息列表
        messages = [
            SystemMessage(content="""你是一个简洁高效的水量数据分析工具, 只输出分析结论:
1. 你只会输出格式化的分析结论, 绝对不会解释你的分析过程
2. 你永远不会写出"我分析了", "根据数据", "通过分析"等说明分析过程的语句
3. 你只关注用户提供的格式模板, 并在这个模板中填充你的结论
4. 你应当拒绝解释你是如何得出结论的
5. 你的回答必须极其简洁直接, 不包含任何多余内容"""),
            HumanMessage(content=state["prompt"])
        ]
        
        # 调用模型
        start_api_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\r等待API响应... 0秒 | 当前时间: {current_time}", end="", flush=True)
        
        # 创建停止事件
        api_stop_event = threading.Event()
        
        # 创建更新计时器
        def update_api_timer():
            start_time = time.time()
            while not api_stop_event.is_set():
                elapsed = time.time() - start_time
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\r等待API响应... {elapsed:.1f}秒 | 当前时间: {current_time}", end="", flush=True)
                time.sleep(0.5)  # 每0.5秒更新一次

        # 创建并启动计时器线程
        api_timer_thread = threading.Thread(target=update_api_timer)
        api_timer_thread.daemon = True
        api_timer_thread.start()
        
        try:
            # 注意使用正确的变量名称
            analysis = model.invoke(messages).content
            
            # 停止计时器线程
            api_stop_event.set()
            api_timer_thread.join(timeout=0.5)
            
            # 显示完成信息
            elapsed_time = time.time() - start_api_time
            print(f"\r完成! 用时: {elapsed_time:.1f}秒{' ' * 20}")
            
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
            
            # 返回模拟分析
            account_id = state["account_id"]
            mock_analysis = f"""# 账户 {account_id} 分析失败

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
            
            return {
                **state,
                "success": False,
                "error": f"AI分析错误: {str(e)}",
                "analysis_result": mock_analysis
            }
    except Exception as e:
        print(f"AI分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "success": False,
            "error": f"AI分析错误: {str(e)}",
            "analysis_result": f"""# 账户 {state["account_id"]} 分析失败

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
        else:
            print(f"  analysis_result存在: 否")
        
        # 只保存到数据库，不再生成报告文件
        if state["success"] and "analysis_result" in state and state["analysis_result"]:
            # 将分析结果保存到数据库
            save_analysis_to_database(account_id, state["analysis_result"], state.get("stats", {}), config)
            
            # 打印状态
            print(f"✓ 账户 {account_id} 分析结果已保存到数据库")
        else:
            print(f"! 账户 {account_id} 分析失败，未保存到数据库")
        
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
    
    return workflow.compile()

def initialize_results_db(db_path):
    """初始化分析结果数据库"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建分析结果表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            leak_probability TEXT,
            risk_warning INTEGER,
            max_usage REAL,
            max_usage_date TEXT,
            avg_usage REAL,
            max_daily_avg REAL,
            historical_avg REAL,
            increase_ratio REAL,
            abnormal_usage TEXT,
            leak_risk_text TEXT,
            full_report TEXT,
            data_source TEXT,
            UNIQUE(account_id, analysis_date)
        )
        ''')
        
        # 创建索引以加快查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_account_id ON analysis_results (account_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_results (analysis_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_leak_probability ON analysis_results (leak_probability)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_warning ON analysis_results (risk_warning)')
        
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
        
        # 提取关键信息
        leak_probability = extract_leak_probability(analysis_text)
        risk_warning = 1 if has_risk_warning(analysis_text) else 0
        
        # 从stats中获取详细数据
        max_usage = stats.get("max_amount", 0)
        max_usage_date = stats.get("max_date", "")
        avg_usage = stats.get("avg_amount", 0)
        max_daily_avg = stats.get("max_daily_avg", 0)
        historical_avg = stats.get("historical_avg_daily", 0)
        increase_ratio = stats.get("usage_increase_ratio", 1.0)
        
        # 提取异常用水和漏水风险文本
        import re
        abnormal_usage_pattern = r"2\.\s*\*\*异常用水\*\*\s*[:：]\s*([^\n]+)"
        abnormal_match = re.search(abnormal_usage_pattern, analysis_text)
        abnormal_usage = abnormal_match.group(1) if abnormal_match else "无明显异常用水"
        
        leak_risk_pattern = r"3\.\s*\*\*漏水风险\*\*\s*[:：]\s*([^\n]+)"
        leak_match = re.search(leak_risk_pattern, analysis_text)
        leak_risk_text = leak_match.group(1) if leak_match else "无明显漏水风险"
        
        # 获取当前日期作为分析日期
        analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        # 插入或更新数据
        cursor.execute('''
        INSERT OR REPLACE INTO analysis_results 
        (account_id, analysis_date, leak_probability, risk_warning, max_usage, 
        max_usage_date, avg_usage, max_daily_avg, historical_avg, increase_ratio, 
        abnormal_usage, leak_risk_text, full_report, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            account_id, 
            analysis_date, 
            leak_probability, 
            risk_warning, 
            max_usage, 
            max_usage_date, 
            avg_usage, 
            max_daily_avg, 
            historical_avg, 
            increase_ratio, 
            abnormal_usage, 
            leak_risk_text, 
            analysis_text, 
            data_source
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

if __name__ == "__main__":
    main() 