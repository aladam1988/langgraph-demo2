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
        "account_data_db": os.path.join(root_dir, "data/account_data.db"),
        "root_dir": root_dir,
        "use_mock_analysis": os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true",  # 添加模拟分析开关
        "reports_dir": os.path.join(root_dir, "reports")
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
        reading_info = ""
        if readings and len(readings) > 0:
            reading_info = f"\n4. **水表读数**: 最新读数 {readings[-1]}，初始读数 {readings[0]}"
        
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

def fetch_account_data(state: GraphState) -> GraphState:
    """从数据库获取账户数据"""
    config = load_config()
    account_id = state["account_id"]
    
    try:
        # 连接数据库
        conn = sqlite3.connect(config["account_data_db"])
        cursor = conn.cursor()
        
        # 获取账户数据
        cursor.execute(
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
        rows = cursor.fetchall()
        
        # 解析数据
        dates = []
        readings = []
        amounts = []
        daily_avgs = []
        
        for row in rows:
            dates.append(row[0])
            readings.append(row[1])
            amounts.append(row[2])
            daily_avgs.append(row[3])
        
        # 关闭连接
        conn.close()
        
        return {
            **state,
            "dates": dates,
            "readings": readings,
            "amounts": amounts,
            "daily_avgs": daily_avgs,
            "success": True
        }
    except Exception as e:
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
        amounts = state["amounts"]
        daily_avgs = state["daily_avgs"]
        
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
            "has_high_usage": any(float(amount) > 5 for amount in amounts),
        }
        
        # 计算日均用水
        if daily_avgs:
            daily_avgs_float = [float(avg) for avg in daily_avgs]
            stats["max_daily_avg"] = max(daily_avgs_float)
            stats["has_high_daily"] = stats["max_daily_avg"] > 3
        else:
            stats["max_daily_avg"] = 0
            stats["has_high_daily"] = False
            
        # 计算风险
        stats["has_leak_risk"] = stats["has_high_daily"] or stats["has_high_usage"]
        
        return {
            **state,
            "stats": stats,
            "success": True
        }
    except Exception as e:
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
        reading_info = ""
        if readings and len(readings) > 0:
            reading_info = f"\n4. **水表读数**: 最新读数 {readings[-1]}，初始读数 {readings[0]}"
        
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

## 异常标准
- 居民日均用水>1方为正常，>3方表示可能存在漏水
- 商业用水突增1倍以上表示可能存在漏水

## 填充以下模板（无需解释分析过程）:

```
{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: [请基于上述数据，判断此账户漏水可能性为：无、低、中、高中的一个]
1. **记录数量和时间跨度**：{stats["record_count"]}条记录，覆盖{stats["time_span"]}
2. **异常用水**：{f"存在{stats['max_amount']}方的高用水量（{stats['max_date']}）" if stats["has_high_usage"] else "无明显异常用水"}
3. **漏水风险**：{f"日均用水量达{stats['max_daily_avg']:.1f}方，超过安全阈值3方" if stats["has_high_daily"] else "无明显漏水风险"}{reading_info}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
```
        """

        return {
            **state,
            "prompt": prompt,
            "success": True
        }
    except Exception as e:
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
            print(f"\r失败! 用时: {elapsed_time:.1f}秒{' ' * 20}")
            
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
    except Exception as e:
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
    """保存分析报告"""
    config = load_config()
    account_id = state["account_id"]
    
    # 确保报告目录存在
    os.makedirs(config["reports_dir"], exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(config["reports_dir"], f"account_{account_id}_analysis_report.md")
    
    try:
        # 写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(state["analysis_result"])
        
        # 获取绝对路径
        report_path = os.path.abspath(output_file)
        file_url = f"file://{report_path}"
        
        # 打印状态
        if state["success"]:
            print(f"✓ 账户 {account_id} 分析报告已保存")
        else:
            print(f"! 账户 {account_id} 分析失败，超时信息已保存")
        print(f"  报告链接: {file_url}")
        
        return state
    except Exception as e:
        print(f"! 保存报告失败: {str(e)}")
        return {
            **state,
            "success": False,
            "error": f"保存报告错误: {str(e)}"
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

def main():
    """使用LangChain的主函数"""
    # 加载配置
    config = load_config()
    
    # 创建报告目录
    os.makedirs(config["reports_dir"], exist_ok=True)
    print(f"报告将保存到: {config['reports_dir']}")
    
    # 获取账户列表
    try:
        conn = sqlite3.connect(config["account_data_db"])
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT account_id FROM meter_readings ORDER BY account_id")
        account_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"找到 {len(account_ids)} 个账户")
    except Exception as e:
        print(f"获取账户列表失败: {str(e)}")
        return
    
    # 创建LangChain组件
    llm = ChatOpenAI(
        openai_api_key=config["llm_api_key"],
        openai_api_base=config["llm_api_base"],
        model_name="deepseek-ai/DeepSeek-R1",
        temperature=0.1,
        max_tokens=800,
        timeout=120
    )
    
    # 创建系统提示
    system_prompt = """你是一个简洁高效的水量数据分析工具, 只输出分析结论:
1. 你只会输出格式化的分析结论, 绝对不会解释你的分析过程
2. 你永远不会写出"我分析了", "根据数据", "通过分析"等说明分析过程的语句
3. 你只关注用户提供的格式模板, 并在这个模板中填充你的结论
4. 你应当拒绝解释你是如何得出结论的
5. 你的回答必须极其简洁直接, 不包含任何多余内容"""
    
    # 创建提示模板
    human_prompt = """
请分析以下用水数据，并填充水用量分析报告模板：

## 用户信息
- 账户ID: {account_id}
- 读表时间: {last_date}
- 当前读数: {last_reading}
- 记录数量: {record_count}条
- 时间跨度: {time_span}

## 用水数据
- 最大用水量: {max_amount}方 (发生于{max_date})
- 最小用水量: {min_amount}方
- 平均用水量: {avg_amount:.2f}方
- 最大日均用水量: {max_daily_avg:.2f}方

## 异常标准
- 居民日均用水>1方为正常，>3方表示可能存在漏水
- 商业用水突增1倍以上表示可能存在漏水

## 填充以下模板（无需解释分析过程）:

```
{risk_warning}
**账户 {account_id} 关键发现**
**漏水可能性**: [请基于上述数据，判断此账户漏水可能性为：无、低、中、高中的一个]
1. **记录数量和时间跨度**：{record_count}条记录，覆盖{time_span}
2. **异常用水**：{abnormal_usage}
3. **漏水风险**：{leak_risk_text}{reading_info}

---
*报告生成时间: {generation_time}*
```
    """
    
    # 创建提示模板对象
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # 创建LangChain链
    chain = prompt_template | llm | StrOutputParser()
    
    # 全局开始时间
    global_start_time = time.time()
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 为每个账户处理
    for i, account_id in enumerate(account_ids[:10], 1):
        print(f"\n[{i}/{min(10, len(account_ids))}] 分析账户 {account_id}...")
        
        # 记录开始时间
        account_start_time = time.time()
        
        try:
            # 数据获取开始时间
            fetch_start = time.time()
            
            # 获取账户数据
            conn = sqlite3.connect(config["account_data_db"])
            cursor = conn.cursor()
            cursor.execute(
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
            rows = cursor.fetchall()
            conn.close()
            
            # 数据获取耗时
            fetch_time = time.time() - fetch_start
            print(f"  数据获取: {fetch_time:.4f}秒")
            
            # 解析数据
            dates = []
            readings = []
            amounts = []
            daily_avgs = []
            
            for row in rows:
                dates.append(row[0])
                readings.append(row[1])
                amounts.append(row[2])
                daily_avgs.append(row[3])
            
            # 计算统计数据
            amounts_float = [float(amount) for amount in amounts]
            avg_amount = sum(amounts_float) / len(amounts_float) if amounts_float else 0
            max_amount = max(amounts_float) if amounts_float else 0
            max_date = dates[amounts_float.index(max(amounts_float))] if amounts_float else "未知"
            min_amount = min(amounts_float) if amounts_float else 0
            time_span = f"{dates[0]} 至 {dates[-1]}" if dates else "未知"
            record_count = len(dates)
            
            # 计算风险相关数据
            has_high_usage = any(float(amount) > 5 for amount in amounts)
            max_daily_avg = max([float(avg) for avg in daily_avgs]) if daily_avgs else 0
            has_high_daily = max_daily_avg > 3
            has_leak_risk = has_high_daily or has_high_usage
            
            # 准备提示变量
            reading_info = f"\n4. **水表读数**: 最新读数 {readings[-1]}，初始读数 {readings[0]}" if readings else ""
            risk_warning = "⚠️ 漏水风险警告" if has_leak_risk else ""
            abnormal_usage = f"存在{max_amount}方的高用水量（{max_date}）" if has_high_usage else "无明显异常用水"
            leak_risk_text = f"日均用水量达{max_daily_avg:.1f}方，超过安全阈值3方" if has_high_daily else "无明显漏水风险"
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 调用LangChain链
            print(f"\r等待API响应... ", end="", flush=True)
            
            # 注意使用正确的变量名称
            analysis = chain.invoke({
                "account_id": account_id,
                "last_date": dates[-1] if dates else "未知",
                "last_reading": readings[-1] if readings else "未知",
                "record_count": record_count,
                "time_span": time_span,
                "max_amount": max_amount,
                "max_date": max_date,
                "min_amount": min_amount,
                "avg_amount": avg_amount,
                "max_daily_avg": max_daily_avg,
                "risk_warning": risk_warning,
                "abnormal_usage": abnormal_usage,
                "leak_risk_text": leak_risk_text,
                "reading_info": reading_info,
                "generation_time": generation_time
            })
            
            # 处理响应
            elapsed_time = time.time() - account_start_time
            print(f"\r完成! 用时: {elapsed_time:.1f}秒{' ' * 20}")
            
            # 后处理：提取代码块
            import re
            code_pattern = r"```(?:markdown)?\s*([\s\S]*?)\s*```"
            code_matches = re.findall(code_pattern, analysis)
            if code_matches:
                analysis = code_matches[0].strip()
            
            # 保存分析结果
            output_file = os.path.join(config["reports_dir"], f"account_{account_id}_analysis_report.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(analysis)
            
            # 获取报告的绝对路径
            report_path = os.path.abspath(output_file)
            file_url = f"file://{report_path}"
            
            # 打印结果
            print(f"✓ 账户 {account_id} 分析报告已保存")
            print(f"  报告链接: {file_url}")
            
        except Exception as e:
            elapsed_time = time.time() - account_start_time
            print(f"\r失败! 用时: {elapsed_time:.1f}秒{' ' * 20}")
            print(f"! 处理账户 {account_id} 时出错: {type(e).__name__}: {str(e)}")
            
            # 生成错误报告
            error_report = f"""# 账户 {account_id} 分析失败

## ⚠️ 分析过程中出错

处理过程中发生错误: {type(e).__name__}

详细信息: {str(e)}

请考虑以下可能的解决方案:
1. 检查数据库连接和数据完整性
2. 检查API密钥和网络连接是否正常
3. 如果问题持续出现, 请联系系统管理员

**注意**: 由于分析未完成, 无法提供漏水可能性评估(无/低/中/高)

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
            
            # 保存错误报告
            output_file = os.path.join(config["reports_dir"], f"account_{account_id}_analysis_report.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(error_report)
                
            # 获取报告的绝对路径
            report_path = os.path.abspath(output_file)
            file_url = f"file://{report_path}"
            
            # 打印结果
            print(f"! 账户 {account_id} 分析失败，错误信息已保存")
            print(f"  报告链接: {file_url}")
    
    print("\n分析完成! 报告文件已保存在报告目录")

if __name__ == "__main__":
    main() 