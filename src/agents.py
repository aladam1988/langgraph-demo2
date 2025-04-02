from typing import Dict, Any, List
from langgraph.graph import Graph, StateGraph
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from loguru import logger
import redis
import json
from datetime import datetime

class AccountProcessor:
    def __init__(self, db_manager, config: Dict[str, Any]):
        self.db_manager = db_manager
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.get("llm_model", "gpt-4"),
            temperature=0
        )
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
            
            # 创建智能体
            agent = create_structured_chat_agent(self.llm, tools)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
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