import os
from dotenv import load_dotenv
from src.agents import AccountProcessor, DeepSeekR1, MockDBManager

# 加载环境变量
load_dotenv()

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