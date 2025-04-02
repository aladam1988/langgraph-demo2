import os
import sys
import time
import uuid
from dotenv import load_dotenv

def main():
    """测试数据库操作"""
    # 加载环境变量
    load_dotenv()
    
    print("开始测试数据库操作...")
    
    try:
        # 导入所需模块
        from src.agents import DBManager
        
        # 初始化数据库管理器
        db_manager = DBManager("data/test.db")
        
        # 生成测试账户ID
        account_id = str(uuid.uuid4())
        
        # 1. 写入测试账户
        print(f"创建测试账户 (ID: {account_id})...")
        result = db_manager.write_data("accounts", {
            "id": account_id,
            "name": "测试账户",
            "status": "active",
            "metadata": {"test": True, "created_at": time.time()}
        })
        print(f"账户创建结果: {result}")
        
        # 2. 写入测试交易
        print("创建测试交易...")
        for i in range(3):
            transaction_id = str(uuid.uuid4())
            result = db_manager.write_data("transactions", {
                "id": transaction_id,
                "account_id": account_id,
                "amount": 100 * (i + 1),
                "type": "deposit" if i % 2 == 0 else "withdrawal",
                "description": f"测试交易 #{i+1}",
                "metadata": {"test": True, "index": i}
            })
            print(f"交易 #{i+1} 创建结果: {result}")
        
        # 3. 读取账户数据
        print("\n读取账户数据...")
        account_data = db_manager.fetch_data(
            account_id, 
            "accounts", 
            "SELECT * FROM accounts WHERE id = ?"
        )
        print(f"账户数据: {account_data}")
        
        # 4. 读取交易数据
        print("\n读取交易数据...")
        transaction_data = db_manager.fetch_data(
            account_id, 
            "transactions", 
            "SELECT * FROM transactions WHERE account_id = ? ORDER BY timestamp DESC"
        )
        print(f"交易数据: {transaction_data}")
        
        # 5. 写入分析结果
        print("\n写入分析结果...")
        result = db_manager.write_data("analysis_results", {
            "account_id": account_id,
            "result_type": "risk_assessment",
            "result_data": "该账户风险评级为低风险",
            "metadata": {"confidence": 0.95, "factors": ["交易频率低", "金额适中"]}
        })
        print(f"分析结果写入: {result}")
        
        # 6. 读取分析结果
        print("\n读取分析结果...")
        analysis_data = db_manager.fetch_data(
            account_id, 
            "analysis_results", 
            "SELECT * FROM analysis_results WHERE account_id = ? ORDER BY timestamp DESC"
        )
        print(f"分析结果数据: {analysis_data}")
        
        print("\n✅ 数据库测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 