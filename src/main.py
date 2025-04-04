import os
from typing import List, Dict, Any
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
        "max_workers": int(os.getenv("MAX_WORKERS", "10")),
        "batch_size": int(os.getenv("BATCH_SIZE", "1000")),
        "use_local_storage": os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true",
        "account_data_db": os.path.join(root_dir, "data/account_data.db"),
        "root_dir": root_dir
    }

class DeepSeekR1:
    """DeepSeek-R1 模型的包装器 (硅基流动平台)"""
    
    def __init__(self, api_key=None, api_base="https://api.siliconflow.cn/v1", temperature=0):
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
        
        try:
            # 准备请求体
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 800  # 减少最大token数，降低生成时间
            }
            
            logger.debug(f"API请求体: {payload}")
                
            logger.info(f"发送请求到DeepSeek API...")
            
            # 增加超时时间
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=180  # 增加超时时间到3分钟
            )
            
            # 记录响应状态和内容
            logger.debug(f"API响应状态码: {response.status_code}")
            logger.debug(f"API响应内容: {response.text[:500]}...")  # 只记录前500个字符
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"成功使用模型 {self.model_name}")
            
            # 检查是否有content字段
            content = result["choices"][0]["message"].get("content", "")
            
            # 如果content为空，检查是否有reasoning_content字段
            if (not content or content.strip() == "") and "reasoning_content" in result["choices"][0]["message"]:
                logger.info("使用reasoning_content字段作为回复内容")
                content = result["choices"][0]["message"]["reasoning_content"]
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"使用模型 {self.model_name} 请求失败: {str(e)}")
            
            # 如果是HTTP错误，尝试获取更详细的错误信息
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API错误详情: {error_detail}")
                except:
                    logger.error(f"API错误状态码: {e.response.status_code}")
                    logger.error(f"API错误响应: {e.response.text}")
            
            # 返回模拟回复
            return f"模拟回复：由于API调用失败({str(e)})，这是一个模拟的回复。"

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
    
    def get_water_usage_data(self):
        """获取用水量数据"""
        try:
            if not self.conn:
                self.connect_db()
            
            # 获取所有账户ID
            self.cursor.execute("SELECT DISTINCT account_id FROM water_usage ORDER BY account_id")
            account_ids = [row[0] for row in self.cursor.fetchall()]
            
            # 限制账户数量
            account_ids = account_ids[:5]  # 只取前5个账户
            
            result = []
            for account_id in account_ids:
                # 获取账户的用水量数据
                self.cursor.execute(
                    "SELECT date, amount FROM water_usage WHERE account_id = ? ORDER BY date DESC LIMIT 10", 
                    (account_id,)
                )
                rows = self.cursor.fetchall()
                
                dates = []
                amounts = []
                for row in rows:
                    dates.append(row[0])
                    amounts.append(row[1])
                
                account_data = {
                    "account_id": account_id,
                    "dates": dates,
                    "amount": amounts
                }
                
                result.append(account_data)
            
            logger.info(f"成功获取{len(result)}个账户的用水量数据")
            return result
            
        except sqlite3.Error as e:
            logger.error(f"获取用水量数据失败: {str(e)}")
            return []
    
    def prepare_data_for_llm(self):
        """准备用于大模型分析的数据"""
        # 获取用水量数据
        water_usage_data = self.get_water_usage_data()
        
        # 准备完整数据
        result = {
            "database_name": os.path.basename(self.db_path),
            "analysis_target": "用水量数据分析",
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
            temperature=0.3
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
            
            # 准备消息
            messages = [
                {"role": "system", "content": "你是一个专业的数据分析师，擅长分析数据库结构和内容，提供深入的见解。"},
                {"role": "user", "content": prompt}
            ]
            
            # 调用大模型API
            logger.info("调用大模型API...")
            
            analysis = self.model.generate(messages)
            
            # 检查是否是模拟回复
            if analysis.startswith("模拟回复："):
                logger.error("API调用失败，返回了模拟回复")
                return {
                    "success": False,
                    "error": "API调用失败",
                    "analysis": self._generate_mock_analysis(data)
                }
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 生成模拟分析结果
            return {
                "success": False,
                "error": str(e),
                "analysis": self._generate_mock_analysis(data)
            }
    
    def _create_prompt(self, data):
        """创建分析提示"""
        # 获取用水量数据
        db_name = data.get("database_name", "未知数据库")
        analysis_target = data.get("analysis_target", "未知分析目标")
        water_usage_data = data.get("water_usage_data", [])
        
        # 构建提示
        prompt = f"""
        请简要分析以下用水量数据，提供关键见解。请控制回复在300字以内。
        
        数据库名称: {db_name}
        分析目标: {analysis_target}
        
        用水量数据样本（每条记录代表一天的用水量）:
        """
        
        # 添加用水量数据（限制数据量）
        for account_data in water_usage_data[:2]:  # 只取前2个账户
            account_id = account_data.get("account_id", "未知")
            dates = account_data.get("dates", [])
            amounts = account_data.get("amount", [])
            
            prompt += f"\n账户ID: {account_id}\n"
            prompt += "日期和用水量数据:\n"
            
            # 只展示前5条数据点
            data_points = min(len(dates), 5)
            for i in range(data_points):
                prompt += f"- {dates[i]}: {amounts[i]}方\n"
        
        # 添加简化的分析要求
        prompt += """
        请提供简要分析:
        1. 用水量趋势是否正常
        2. 根据连续几天的用水量数据，判断是否存在漏水迹象
        
        判断标准：
        - 居民用水：正常情况下5天内用水量约1方以下；如果用量持续增加并超过3方可能是管道漏水；如果突然超过5方或连续每天都有0.5-1方用水可能是器具漏水
        - 商业用水：正常情况下10天用水量约10-50方；如果当前用量较之前增大一倍可能存在漏水
        
        请保持简洁，控制回复在300字以内。
        """
        
        return prompt
    
    def _generate_mock_analysis(self, data):
        """生成模拟分析结果（当API调用失败时使用）"""
        # 获取数据库结构信息
        db_schema = data.get("db_schema", {})
        sample_data = data.get("sample_data", {})
        account_details = data.get("account_details")
        
        # 构建模拟分析报告
        if account_details:
            # 账户分析报告
            account_id = account_details.get("account", {}).get("id", "未知")
            
            return f"""
            # 账户 {account_id} 分析报告（模拟结果）
            
            > 注意：这是一个模拟分析结果，因为大模型API调用失败。
            
            ## 1. 账户概述
            
            账户ID: {account_id}
            状态: {account_details.get("account", {}).get("status", "未知")}
            创建时间: {account_details.get("account", {}).get("created_at", "未知")}
            
            ## 2. 交易模式分析
            
            该账户有 {len(account_details.get("transactions", []))} 条交易记录。
            
            ## 3. 指标分析
            
            根据交易数据，该账户的活动主要集中在特定时间段。
            
            ## 4. 风险评估
            
            基于有限的数据，无法进行全面的风险评估。
            
            ## 5. 建议
            
            建议进一步收集和分析该账户的数据，以获得更准确的见解。
            """
        else:
            # 数据库分析报告
            tables = list(db_schema.keys())
            
            return f"""
            # 数据库分析报告（模拟结果）
            
            > 注意：这是一个模拟分析结果，因为大模型API调用失败。
            
            ## 1. 数据库结构分析
            
            数据库包含 {len(tables)} 个表: {', '.join(tables)}
            
            ## 2. 数据质量分析
            
            由于这是模拟结果，无法提供详细的数据质量分析。
            
            ## 3. 数据模式和趋势
            
            需要更多数据来识别模式和趋势。
            
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

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 设置日志级别
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("app.log", rotation="500 MB", level="DEBUG")
    
    logger.info("开始分析数据库...")
    
    # 数据库路径
    db_path = config.get("account_data_db")
    logger.info(f"数据库路径: {db_path}")
    
    try:
        # 初始化数据分析器
        analyzer = AccountDataAnalyzer(db_path)
        
        # 准备数据
        data = analyzer.prepare_data_for_llm()
        
        # 初始化大模型分析器
        llm_analyzer = LLMAnalyzer(config)
        
        # 分析整个数据库
        logger.info("开始分析整个数据库...")
        result = llm_analyzer.analyze_data(data)
        
        if result["success"]:
            logger.info("数据库分析成功")
            
            # 保存分析结果
            output_file = os.path.join(config["root_dir"], "database_analysis_report.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["analysis"])
            logger.info(f"分析报告已保存到: {output_file}")
        else:
            logger.error(f"数据库分析失败: {result.get('error', '未知错误')}")
            
            # 即使分析失败，也保存模拟分析结果
            if "analysis" in result:
                output_file = os.path.join(config["root_dir"], "database_analysis_report_mock.md")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result["analysis"])
                logger.info(f"模拟分析报告已保存到: {output_file}")
    
    except FileNotFoundError as e:
        logger.error(f"数据库文件不存在: {str(e)}")
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # 关闭连接
        if 'analyzer' in locals():
            analyzer.close_db()
    
    logger.info("分析完成")

if __name__ == "__main__":
    main() 