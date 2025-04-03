import os
import sys
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

def main():
    """创建账号为1的数据库记录"""
    # 加载环境变量
    load_dotenv()
    
    print("开始创建账号为1的数据库记录...")
    
    try:
        # 修改导入方式，避免执行模块级代码
        # 直接导入DBManager类，而不是整个模块
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.agents import DBManager
        
        # 初始化数据库管理器
        db_manager = DBManager("data/account_data.db")
        
        # 账户ID
        account_id = "1"
        
        # 1. 创建账户
        print(f"创建账户 (ID: {account_id})...")
        result = db_manager.write_data("accounts", {
            "id": account_id,
            "name": "时间序列测试账户",
            "status": "active",
            "metadata": {
                "description": "包含时间序列数据的测试账户",
                "created_at": time.time()
            }
        })
        print(f"账户创建结果: {result}")
        
        # 2. 创建交易记录
        # 数据格式: 日期, 指标1, 指标2, 指标3, 指标4, 指标5, 指标6
        data = [
            ["2025-01-01", 1, 0, 1.6, 0, 0, 0],
            ["2025-01-11", 1, 0, 1.6, 0, 0, 0],
            ["2025-01-21", 1, 0, 1.6, 0, 0, 0],
            ["2025-02-01", 1, 0, 1.6, 0, 0, 0],
            ["2025-02-11", 1, 0, 2, 0, 0, 0],
            ["2025-02-21", 1, 0, 2.4, 0, 0, 0],
            ["2025-03-01", 1, 0, 3.4, 0, 0, 0],
            ["2025-03-11", 6, 5, 3.4, 5, 5, 5],
            ["2025-03-21", 12, 6, 3.4, 6, 6, 6],
            ["2025-04-01", 24, 12, 3.4, 12, 12, 12]
        ]
        
        print("\n创建交易记录...")
        for entry in data:
            date_str, metric1, metric2, metric3, metric4, metric5, metric6 = entry
            
            # 将日期字符串转换为时间戳
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            timestamp = date_obj.timestamp()
            
            # 创建交易记录
            transaction_id = str(uuid.uuid4())
            result = db_manager.write_data("transactions", {
                "id": transaction_id,
                "account_id": account_id,
                "amount": metric1,  # 使用指标1作为交易金额
                "type": "data_entry",
                "description": f"数据记录 - {date_str}",
                "timestamp": timestamp,
                "metadata": {
                    "date": date_str,
                    "metric1": metric1,
                    "metric2": metric2,
                    "metric3": metric3,
                    "metric4": metric4,
                    "metric5": metric5,
                    "metric6": metric6
                }
            })
            print(f"日期 {date_str} 的数据记录创建结果: {result}")
        
        # 3. 创建一个初始分析结果
        print("\n创建初始分析结果...")
        result = db_manager.write_data("analysis_results", {
            "id": str(uuid.uuid4()),
            "account_id": account_id,
            "result_type": "initial_assessment",
            "result_data": "这是账户1的初始数据分析。数据显示在2025年3月开始有显著变化，多个指标从0上升到较高值。",
            "metadata": {
                "created_at": time.time(),
                "data_points": len(data),
                "date_range": f"{data[0][0]} 至 {data[-1][0]}"
            }
        })
        print(f"初始分析结果创建结果: {result}")
        
        # 4. 读取账户数据
        print("\n读取账户数据...")
        account_data = db_manager.fetch_data(
            account_id, 
            "accounts", 
            "SELECT * FROM accounts WHERE id = ?"
        )
        print(f"账户数据: {account_data}")
        
        # 5. 读取交易数据
        print("\n读取交易数据...")
        transaction_data = db_manager.fetch_data(
            account_id, 
            "transactions", 
            "SELECT * FROM transactions WHERE account_id = ? ORDER BY timestamp ASC"
        )
        print(f"交易数据: {transaction_data}")
        
        # 6. 读取分析结果
        print("\n读取分析结果...")
        analysis_data = db_manager.fetch_data(
            account_id, 
            "analysis_results", 
            "SELECT * FROM analysis_results WHERE account_id = ? ORDER BY timestamp DESC"
        )
        print(f"分析结果数据: {analysis_data}")
        
        print("\n✅ 账号为1的数据库记录创建完成！")
        
    except Exception as e:
        print(f"❌ 创建过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 