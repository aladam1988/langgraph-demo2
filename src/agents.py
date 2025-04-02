from typing import Dict, Any, List
from langgraph.graph import Graph, StateGraph
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from loguru import logger
import redis
import json
from datetime import datetime
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import requests
import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel

class DeepSeekR1(BaseChatModel):
    """DeepSeek-R1 模型的LangChain包装器 (硅基流动平台)"""
    
    api_key: str = Field(...)  # 必需字段
    api_base: str = Field(default="https://api.siliconflow.cn/v1")
    model_name: str = Field(default="deepseek-ai/DeepSeek-R1")
    temperature: float = Field(default=0)
    
    class Config:
        """配置类"""
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识符"""
        return "deepseek-r1"
        
    def _generate(self, messages, stop=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 转换LangChain消息格式为DeepSeek格式
        deepseek_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                deepseek_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                deepseek_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                deepseek_messages.append({"role": "assistant", "content": message.content})
        
        payload = {
            "model": self.model_name,
            "messages": deepseek_messages,
            "temperature": self.temperature,
        }
        
        if stop:
            payload["stop"] = stop
            
        response = requests.post(f"{self.api_base}/chat/completions", 
                                headers=headers, 
                                json=payload)
        response.raise_for_status()
        
        result = response.json()
        return AIMessage(content=result["choices"][0]["message"]["content"])
    
    async def _agenerate(self, messages, stop=None):
        # 异步实现可以在这里添加
        raise NotImplementedError("异步生成尚未实现")

    def test_model(self):
        """测试模型是否正常工作"""
        try:
            response = self.llm._generate([
                SystemMessage(content="你是一个有帮助的AI助手。"),
                HumanMessage(content="你好，请用一句话介绍自己。")
            ])
            return {
                "success": True,
                "response": response.content
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class AccountProcessor:
    def __init__(self, db_manager, config: Dict[str, Any]):
        self.db_manager = db_manager
        self.config = config
        
        # 首先获取模型类型
        model_type = config.get("model_type", "openai")
        
        # 然后检查API密钥
        if "deepseek_api_key" not in config and model_type in ["deepseek", "siliconflow"]:
            config["deepseek_api_key"] = "sk-qsybuxxdlcvuhmtmbollzxzxvkwzqzxbkmbockxpujpcjyfk"
        
        # 根据配置选择使用的LLM模型
        if model_type == "openai":
            self.llm = ChatOpenAI(
                model_name=config.get("llm_model", "gpt-4"),
                temperature=0
            )
        elif model_type == "deepseek" or model_type == "siliconflow":
            self.llm = DeepSeekR1(
                api_key=config.get("deepseek_api_key"),
                api_base=config.get("deepseek_api_base", "https://api.siliconflow.cn/v1"),
                model_name=config.get("llm_model", "deepseek-ai/DeepSeek-R1"),
                temperature=config.get("temperature", 0)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=0
        )
        
    def create_tools(self, account_id: str) -> List[Tool]:
        """为每个账号创建特定的工具"""
        return [
            Tool(
                name="fetch_db1_data",
                func=lambda query: self.db_manager.fetch_data(account_id, "db1", query),
                description="从第一个数据库获取数据"
            ),
            Tool(
                name="fetch_db2_data",
                func=lambda query: self.db_manager.fetch_data(account_id, "db2", query),
                description="从第二个数据库获取数据"
            ),
            Tool(
                name="check_historical_data",
                func=lambda: self.get_historical_data(account_id),
                description="获取历史处理数据"
            )
        ]
    
    def get_historical_data(self, account_id: str) -> Dict[str, Any]:
        """获取账号的历史处理数据"""
        key = f"account:{account_id}:history"
        data = self.redis_client.get(key)
        return json.loads(data) if data else {}
    
    def save_historical_data(self, account_id: str, data: Dict[str, Any]):
        """保存账号的处理历史"""
        key = f"account:{account_id}:history"
        self.redis_client.setex(
            key,
            86400 * 7,  # 保存7天
            json.dumps(data)
        )
    
    def is_account_changed(self, account_id: str, current_data: Dict[str, Any]) -> bool:
        """检查账号数据是否发生变化"""
        historical_data = self.get_historical_data(account_id)
        if not historical_data:
            return True
        return historical_data.get("data_hash") != current_data.get("data_hash")
    
    def process_account(self, account_id: str) -> Dict[str, Any]:
        """处理单个账号的数据"""
        try:
            # 获取当前数据
            current_data = {
                "db1": self.db_manager.fetch_data(account_id, "db1", "SELECT * FROM data"),
                "db2": self.db_manager.fetch_data(account_id, "db2", "SELECT * FROM data")
            }
            
            # 检查数据是否变化
            if not self.is_account_changed(account_id, current_data):
                logger.info(f"Account {account_id} data unchanged, skipping...")
                return {
                    "account_id": account_id,
                    "status": "skipped",
                    "reason": "data_unchanged"
                }
            
            # 创建工具
            tools = self.create_tools(account_id)
            
            # 创建智能体 - 使用新的初始化方法
            agent_executor = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent="structured_chat",
                verbose=True
            )
            
            # 执行处理逻辑
            result = agent_executor.run(
                f"分析账号 {account_id} 的数据，检查是否存在问题。重点关注：\n"
                f"1. 数据一致性\n"
                f"2. 异常值\n"
                f"3. 业务规则违反\n"
                f"4. 数据完整性"
            )
            
            # 保存处理结果
            self.save_historical_data(account_id, {
                "last_processed": datetime.now().isoformat(),
                "result": result,
                "data_hash": hash(str(current_data))
            })
            
            return {
                "account_id": account_id,
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing account {account_id}: {str(e)}")
            return {
                "account_id": account_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_model(self):
        """测试模型是否正常工作"""
        try:
            response = self.llm._generate([
                SystemMessage(content="你是一个有帮助的AI助手。"),
                HumanMessage(content="你好，请用一句话介绍自己。")
            ])
            return {
                "success": True,
                "response": response.content
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# 加载环境变量
load_dotenv()

# 创建一个简单的模拟数据库管理器
class MockDBManager:
    def fetch_data(self, account_id, db_name, query):
        return {"mock_data": "这是模拟数据"}

# 配置
config = {
    "model_type": "siliconflow",  # 使用硅基流动平台
    # 如果你已经设置了环境变量，这里可以不提供API密钥
}

# 初始化AccountProcessor
processor = AccountProcessor(MockDBManager(), config)

# 测试模型
result = processor.test_model()

# 输出结果
if result["success"]:
    print("✅ 模型测试成功！")
    print(f"模型响应: {result['response']}")
else:
    print("❌ 模型测试失败!")
    print(f"错误信息: {result['error']}") 