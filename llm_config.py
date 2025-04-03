import os
import importlib.util
import requests

class DeepSeekR1:
    """DeepSeek-R1 模型的包装器 (硅基流动平台)"""
    
    def __init__(self, api_key=None, api_base="https://api.siliconflow.cn/v1", 
                 model_name="deepseek-ai/DeepSeek-R1", temperature=0):
        """初始化DeepSeek-R1模型"""
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请提供DeepSeek API密钥或设置DEEPSEEK_API_KEY环境变量")
            
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
    
    def generate(self, messages):
        """生成回复"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
            
        response = requests.post(f"{self.api_base}/chat/completions", 
                                headers=headers, 
                                json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

def get_llm(provider="deepseek", api_key=None):
    """获取LLM模型"""
    if provider == "deepseek":
        return DeepSeekR1(api_key=api_key)
    else:
        raise ValueError(f"不支持的提供商: {provider}")