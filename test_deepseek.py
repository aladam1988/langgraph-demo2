import os
import requests
import json
from dotenv import load_dotenv
from loguru import logger

def test_deepseek_analysis():
    """测试DeepSeek模型分析用水量数据"""
    # 从环境变量加载API密钥
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1")
    model_name = "deepseek-ai/DeepSeek-R1"
    
    if not api_key:
        logger.error("未找到DEEPSEEK_API_KEY环境变量")
        return False
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 创建用水量数据分析提示
    prompt = """
    请分析以下用水量数据，提供详细见解。
    
    账户ID: 1
    日期和用水量数据:
    - 2023-01-01: 120
    - 2023-01-02: 115
    - 2023-01-03: 125
    - 2023-01-04: 110
    - 2023-01-05: 105
    
    账户ID: 2
    日期和用水量数据:
    - 2023-01-01: 200
    - 2023-01-02: 210
    - 2023-01-03: 195
    - 2023-01-04: 220
    - 2023-01-05: 205
    
    请提供以下分析:
    1. 用水模式分析：识别用水量的模式和趋势
    2. 账户比较：比较两个账户的用水模式差异
    3. 异常检测：识别异常的用水量数据点
    4. 建议：基于分析提供节水或优化用水的建议
    
    请以结构化的方式提供你的分析，使用标题和小节组织内容。
    """
    
    try:
        logger.info(f"使用模型 {model_name} 分析用水量数据...")
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        logger.info("发送请求到DeepSeek API...")
        
        response = requests.post(
            f"{api_base}/chat/completions", 
            headers=headers, 
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # 保存响应内容到文件
        output_file = os.path.join(os.getcwd(), "deepseek_analysis_result.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"模型响应已保存到文件: {output_file}")
        
        # 显示响应内容的前200个字符
        preview = content[:200] + "..." if len(content) > 200 else content
        logger.info(f"响应预览: {preview}")
        
        logger.success("测试成功: 能够分析用水量数据")
        return True
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        
        # 如果是HTTP错误，尝试获取更详细的错误信息
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"API错误详情: {error_detail}")
            except:
                logger.error(f"API错误状态码: {e.response.status_code}")
                logger.error(f"API错误响应: {e.response.text}")
        
        return False

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 运行测试
    test_deepseek_analysis()