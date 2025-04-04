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
    
    # 创建简化的用水量数据分析提示
    prompt = """
    分析以下简单的用水量数据:
    
    账户1: [120, 115, 125, 110, 105]
    账户2: [200, 210, 195, 220, 205]
    
    请简要分析用水模式和差异。
    """
    
    try:
        logger.info(f"使用模型 {model_name} 分析用水量数据...")
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500  # 减少生成的token数量
        }
        
        logger.info("发送请求到DeepSeek API...")
        
        # 增加超时时间到120秒
        response = requests.post(
            f"{api_base}/chat/completions", 
            headers=headers, 
            json=payload,
            timeout=120  # 增加到120秒
        )
        
        response.raise_for_status()
        
        # 记录完整的API响应
        logger.info(f"API响应状态码: {response.status_code}")
        logger.info(f"API响应头: {dict(response.headers)}")
        
        # 保存原始响应到文件
        raw_response_file = os.path.join(os.getcwd(), "deepseek_raw_response.json")
        with open(raw_response_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info(f"原始API响应已保存到: {raw_response_file}")
        
        # 解析JSON响应
        result = response.json()
        logger.info(f"API响应JSON结构: {json.dumps(result, indent=2)[:500]}...")
        
        # 检查响应结构
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("API响应中没有choices字段或为空")
            return False
            
        if "message" not in result["choices"][0]:
            logger.error("API响应中没有message字段")
            return False
        
        message = result["choices"][0]["message"]
        
        # 尝试获取content或reasoning_content
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        
        # 如果content为空但reasoning_content不为空，使用reasoning_content
        if (not content or content.strip() == "") and reasoning_content:
            logger.info("content为空，使用reasoning_content")
            content = reasoning_content
        
        # 检查内容是否为空
        if not content or content.strip() == "":
            logger.error("API返回的内容为空")
            return False
        
        # 保存响应内容到文件
        output_file = os.path.join(os.getcwd(), "deepseek_analysis_result.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"模型响应已保存到文件: {output_file}")
        
        # 显示完整响应内容
        logger.info("模型响应:")
        print("\n" + "-"*80)
        print(content)
        print("-"*80 + "\n")
        
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
        
        # 尝试备用模型
        logger.info("尝试使用备用模型...")
        try:
            backup_model = "deepseek-chat"
            logger.info(f"使用备用模型: {backup_model}")
            
            payload["model"] = backup_model
            
            response = requests.post(
                f"{api_base}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 保存响应内容到文件
            output_file = os.path.join(os.getcwd(), "deepseek_backup_result.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"备用模型响应已保存到文件: {output_file}")
            logger.success("备用模型测试成功")
            return True
        except Exception as backup_e:
            logger.error(f"备用模型测试失败: {str(backup_e)}")
            return False
        
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