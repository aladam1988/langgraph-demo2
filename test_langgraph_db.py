import os
import sys
import uuid
import time
from dotenv import load_dotenv

def main():
    """测试LangGraph与数据库集成"""
    # 加载环境变量
    load_dotenv()
    
    print("开始测试LangGraph与数据库集成...")
    
    try:
        # 导入所需模块
        from src.agents import DBManager
        from src.workflow import run_analysis_workflow
        
        # 初始化数据库管理器
        db_manager = DBManager("data/test_workflow.db")
        
        # 生成测试账户ID
        account_id = str(uuid.uuid4())
        
        # 1. 写入测试账户
        print(f"创建测试账户 (ID: {account_id})...")
        result = db_manager.write_data("accounts", {
            "id": account_id,
            "name": "测试工作流账户",
            "status": "active",
            "metadata": {"test": True, "created_at": time.time()}
        })
        print(f"账户创建结果: {result}")
        
        # 2. 写入测试交易
        print("创建测试交易...")
        # 正常交易
        for i in range(3):
            transaction_id = str(uuid.uuid4())
            result = db_manager.write_data("transactions", {
                "id": transaction_id,
                "account_id": account_id,
                "amount": 100 * (i + 1),
                "type": "deposit" if i % 2 == 0 else "withdrawal",
                "description": f"正常测试交易 #{i+1}",
                "metadata": {"test": True, "index": i}
            })
            print(f"交易 #{i+1} 创建结果: {result}")
        
        # 添加一个可疑交易
        transaction_id = str(uuid.uuid4())
        result = db_manager.write_data("transactions", {
            "id": transaction_id,
            "account_id": account_id,
            "amount": 50000,  # 大额交易
            "type": "withdrawal",
            "description": "大额提现",
            "metadata": {"test": True, "suspicious": True}
        })
        print(f"可疑交易创建结果: {result}")
        
        # 3. 运行工作流
        print("\n运行LangGraph工作流...")
        final_state = run_analysis_workflow(account_id)
        
        # 4. 显示结果
        print("\n工作流执行结果:")
        print(f"分析类型: {final_state['analysis_result']['result_type']}")
        print(f"分析结果: {final_state['analysis_result']['result_data']}")
        
        # 5. 从数据库读取分析结果
        print("\n从数据库读取分析结果...")
        analysis_data = db_manager.fetch_data(
            account_id, 
            "analysis_results", 
            "SELECT * FROM analysis_results WHERE account_id = ? ORDER BY timestamp DESC"
        )
        print(f"数据库中的分析结果: {analysis_data}")
        
        print("\n✅ LangGraph与数据库集成测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 