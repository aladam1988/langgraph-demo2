import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from data_access import DatabaseManager
from agents import AccountProcessor
import redis
import json
from datetime import datetime
import hashlib

def load_config():
    """加载配置"""
    load_dotenv()
    return {
        "db1_connection_string": os.getenv("DB1_CONNECTION_STRING"),
        "db2_connection_string": os.getenv("DB2_CONNECTION_STRING"),
        "db1_mongo_uri": os.getenv("DB1_MONGO_URI"),
        "db2_mongo_uri": os.getenv("DB2_MONGO_URI"),
        "db1_database": os.getenv("DB1_DATABASE"),
        "db2_database": os.getenv("DB2_DATABASE"),
        "db1_collection": os.getenv("DB1_COLLECTION"),
        "db2_collection": os.getenv("DB2_COLLECTION"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4"),
        "max_workers": int(os.getenv("MAX_WORKERS", "10")),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "batch_size": int(os.getenv("BATCH_SIZE", "1000"))
    }

class AccountBatchProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.redis_client = redis.Redis(
            host=config["redis_host"],
            port=config["redis_port"],
            db=0
        )
        self.db_manager = DatabaseManager(config)
        self.processor = AccountProcessor(self.db_manager, config)
        
    def get_processed_accounts(self) -> set:
        """获取已处理的账号列表"""
        key = f"processed:accounts:{datetime.now().strftime('%Y%m%d')}"
        data = self.redis_client.get(key)
        return set(json.loads(data)) if data else set()
    
    def save_processed_accounts(self, account_ids: set):
        """保存已处理的账号列表"""
        key = f"processed:accounts:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.setex(
            key,
            86400 * 7,  # 保存7天
            json.dumps(list(account_ids))
        )
    
    def get_problematic_accounts(self) -> List[str]:
        """获取有问题的账号列表"""
        key = f"problematic:accounts:{datetime.now().strftime('%Y%m%d')}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else []
    
    def save_problematic_accounts(self, account_ids: List[str]):
        """保存有问题的账号列表"""
        key = f"problematic:accounts:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.setex(
            key,
            86400 * 7,  # 保存7天
            json.dumps(account_ids)
        )
    
    def process_batch(self, account_ids: List[str]) -> List[Dict[str, Any]]:
        """处理一批账号"""
        results = []
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = [
                executor.submit(self.processor.process_account, account_id)
                for account_id in account_ids
            ]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Processed account: {result['account_id']}")
                except Exception as e:
                    logger.error(f"Error in future execution: {str(e)}")
        
        return results

def main():
    """主函数"""
    # 配置日志
    logger.add("processing.log", rotation="500 MB")
    
    # 加载配置
    config = load_config()
    
    # 创建批处理器
    batch_processor = AccountBatchProcessor(config)
    
    # 获取已处理的账号
    processed_accounts = batch_processor.get_processed_accounts()
    
    # 读取所有账号
    with open("account_ids.txt", "r") as f:
        all_accounts = [line.strip() for line in f]
    
    # 过滤出未处理的账号
    accounts_to_process = [acc for acc in all_accounts if acc not in processed_accounts]
    
    # 分批处理账号
    batch_size = config["batch_size"]
    problematic_accounts = []
    
    for i in range(0, len(accounts_to_process), batch_size):
        batch = accounts_to_process[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {(len(accounts_to_process) + batch_size - 1)//batch_size}")
        
        results = batch_processor.process_batch(batch)
        
        # 收集有问题的账号
        problematic_accounts.extend([
            result["account_id"] 
            for result in results 
            if result["status"] == "success" and "problem" in result["result"].lower()
        ])
        
        # 更新已处理账号列表
        processed_accounts.update(batch)
        batch_processor.save_processed_accounts(processed_accounts)
    
    # 保存有问题的账号
    batch_processor.save_problematic_accounts(problematic_accounts)
    
    # 输出统计信息
    logger.info(f"Processing completed. Total accounts: {len(all_accounts)}")
    logger.info(f"Processed accounts: {len(processed_accounts)}")
    logger.info(f"Problematic accounts: {len(problematic_accounts)}")
    
    # 生成报告
    report = {
        "date": datetime.now().isoformat(),
        "total_accounts": len(all_accounts),
        "processed_accounts": len(processed_accounts),
        "problematic_accounts": len(problematic_accounts),
        "problematic_account_ids": problematic_accounts
    }
    
    with open(f"report_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main() 