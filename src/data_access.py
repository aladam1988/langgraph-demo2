from typing import Dict, Any
from sqlalchemy import create_engine
from pymongo import MongoClient
from loguru import logger

class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        
    def get_sql_connection(self, account_id: str, db_type: str):
        """获取SQL数据库连接"""
        if f"{account_id}_{db_type}" not in self.connections:
            connection_string = self.config[f"{db_type}_connection_string"]
            self.connections[f"{account_id}_{db_type}"] = create_engine(connection_string)
        return self.connections[f"{account_id}_{db_type}"]
    
    def get_mongo_connection(self, account_id: str, db_type: str):
        """获取MongoDB连接"""
        if f"{account_id}_{db_type}" not in self.connections:
            connection_string = self.config[f"{db_type}_mongo_uri"]
            self.connections[f"{account_id}_{db_type}"] = MongoClient(connection_string)
        return self.connections[f"{account_id}_{db_type}"]
    
    def fetch_data(self, account_id: str, db_type: str, query: str):
        """从指定数据库获取数据"""
        try:
            if "mongo" in db_type:
                client = self.get_mongo_connection(account_id, db_type)
                db = client[self.config[f"{db_type}_database"]]
                return list(db[self.config[f"{db_type}_collection"]].find())
            else:
                engine = self.get_sql_connection(account_id, db_type)
                return engine.execute(query).fetchall()
        except Exception as e:
            logger.error(f"Error fetching data for account {account_id} from {db_type}: {str(e)}")
            raise 