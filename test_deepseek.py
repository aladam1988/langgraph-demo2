import os
import sys
import time
import subprocess
from dotenv import load_dotenv

# 创建一个新的Python文件来执行实际的测试
def create_test_runner():
    with open("_test_runner.py", "w") as f:
        f.write("""
import os
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("开始测试DeepSeek-R1模型...")

try:
    # 创建一个简单的模拟数据库管理器
    class MockDBManager:
        def fetch_data(self, account_id, db_name, query):
            return {"mock_data": "这是模拟数据"}
    
    # 配置
    config = {
        "model_type": "siliconflow",  # 使用硅基流动平台
    }
    
    # 导入AccountProcessor
    from src.agents import AccountProcessor
    
    # 初始化AccountProcessor
    processor = AccountProcessor(MockDBManager(), config)
    
    # 测试模型并计时
    start_time = time.time()
    result = processor.test_model()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 输出结果
    if result["success"]:
        print("✅ 模型测试成功！")
        print(f"模型响应: {result['response']}")
        print(f"响应时间: {elapsed_time:.2f} 秒")
    else:
        print("❌ 模型测试失败!")
        print(f"错误信息: {result['error']}")
    
    # 关闭可能的连接
    if hasattr(processor, 'redis_client'):
        processor.redis_client.close()
    
    print("测试完成。")
    
except Exception as e:
    print(f"❌ 测试过程中发生错误: {str(e)}")
    import traceback
    traceback.print_exc()
""")

# 主函数：创建并运行测试脚本
def main():
    # 创建测试运行器脚本
    create_test_runner()
    
    # 在单独的进程中运行测试
    try:
        # 使用subprocess运行测试脚本
        result = subprocess.run([sys.executable, "_test_runner.py"], 
                               check=True, 
                               text=True)
        
        # 删除临时测试脚本
        if os.path.exists("_test_runner.py"):
            os.remove("_test_runner.py")
            
    except subprocess.CalledProcessError as e:
        print(f"测试执行失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    
    # 确保程序退出
    print("测试流程结束。")

# 执行主函数
if __name__ == "__main__":
    main() 