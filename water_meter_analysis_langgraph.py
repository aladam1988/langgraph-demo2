import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from llm_config import get_llm, DeepSeekR1
from dotenv import load_dotenv
import concurrent.futures
import requests

# 尝试导入LangGraph相关依赖
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    langgraph_available = True
except ImportError:
    print("警告: LangGraph未安装。将使用简化版分析。")
    print("要使用完整功能，请运行: pip install langgraph")
    langgraph_available = False

# 导入现有的WaterMeterDataReader类
from read_water_meter_db import WaterMeterDataReader

# 确保加载.env文件
load_dotenv(verbose=True)  # 添加verbose=True参数以显示加载信息

# 初始化DeepSeek-R1模型
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    api_key = input("请输入DeepSeek API密钥: ")
    os.environ["DEEPSEEK_API_KEY"] = api_key

llm = get_llm("deepseek")

# 定义状态类型
class GraphState:
    """图状态类"""
    def __init__(self):
        self.account_id = None
        self.data = None
        self.analysis_results = {}
        self.questions = []
        self.answers = []
        self.current_question = None
        self.charts = []
        self.recommendations = []
        self.final_report = None

# 工具函数
def get_account_data(state: GraphState, account_id: str) -> GraphState:
    """获取指定账户的数据"""
    reader = WaterMeterDataReader()
    try:
        df = reader.get_readings_as_dataframe(account_id)
        if df.empty:
            state.analysis_results["error"] = f"账户 {account_id} 没有数据"
        else:
            state.account_id = account_id
            state.data = df.to_dict(orient='records')
            state.analysis_results["basic_info"] = {
                "total_records": len(df),
                "date_range": f"{df['reading_time'].min().strftime('%Y-%m-%d')} 至 {df['reading_time'].max().strftime('%Y-%m-%d')}",
                "total_usage": df['current_usage'].sum(),
                "average_usage": df['current_usage'].mean()
            }
    finally:
        reader.close()
    return state

def answer_question(state: GraphState) -> GraphState:
    """使用大模型回答问题"""
    if not state.current_question or not state.analysis_results:
        return state
    
    # 准备上下文信息
    context = {
        "account_id": state.account_id,
        "basic_info": state.analysis_results.get("basic_info", {}),
        "usage_patterns": state.analysis_results.get("usage_patterns", {})
    }
    
    # 根据问题类型生成回答
    question = state.current_question
    answer = ""
    
    if "用水模式" in question:
        patterns = context["usage_patterns"]
        answer = f"账户{state.account_id}的用水模式分析如下：\n\n"
        answer += f"- 总用水量: {patterns['total_usage']:.2f}\n"
        answer += f"- 平均用水量: {patterns['average_usage']:.2f}\n"
        answer += f"- 最大用水量: {patterns['max_usage']:.2f} (日期: {patterns['max_usage_date']})\n"
        answer += f"- 零用水天数: {patterns['zero_usage_days']}\n"
        answer += f"- 用水趋势: {patterns['trend']}\n"
        
        # 根据数据特征添加更多描述
        if patterns['zero_usage_days'] > len(state.data) * 0.3:
            answer += "\n该账户有较多零用水天数，可能是间歇性使用或者长期不在家。"
        
        if patterns['trend'] == "上升":
            answer += "\n该账户的用水量呈上升趋势，可能是家庭成员增加或用水习惯改变。"
        elif patterns['trend'] == "下降":
            answer += "\n该账户的用水量呈下降趋势，可能是采取了节水措施或家庭成员减少。"
        else:
            answer += "\n该账户的用水量相对稳定，没有明显的变化趋势。"
    
    elif "异常用水" in question:
        patterns = context["usage_patterns"]
        anomalies = patterns.get("anomalies", [])
        
        if anomalies:
            answer = f"账户{state.account_id}存在以下异常用水情况：\n\n"
            for anomaly in anomalies:
                answer += f"- 日期: {anomaly['date']}, 用水量: {anomaly['usage']:.2f}, "
                answer += f"超过平均值({anomaly['average']:.2f})的 {anomaly['usage']/anomaly['average']:.1f}倍\n"
            
            answer += "\n可能的原因包括：\n"
            answer += "1. 家庭聚会或特殊活动\n"
            answer += "2. 水管破裂或漏水\n"
            answer += "3. 季节性用水增加（如夏季）\n"
            answer += "4. 家庭成员临时增加\n"
        else:
            answer = f"账户{state.account_id}没有检测到明显的异常用水情况。用水量基本在平均水平附近波动。"
    
    elif "漏水" in question:
        patterns = context["usage_patterns"]
        leakage_periods = patterns.get("leakage_periods", [])
        
        if leakage_periods:
            answer = f"账户{state.account_id}可能存在漏水情况：\n\n"
            for period in leakage_periods:
                answer += f"- 时间段: {period['start_date']} 至 {period['end_date']}\n"
                answer += f"- 这段时间的用水量: {period['usage_values']}\n"
            
            answer += "\n漏水的可能迹象：\n"
            answer += "1. 连续多天有小量但持续的用水记录\n"
            answer += "2. 这些用水量低于平均值但始终不为零\n"
            answer += "3. 用水模式不符合正常生活习惯\n\n"
            answer += "建议检查家中水管、水龙头和马桶等设备是否有漏水现象。"
        else:
            answer = f"账户{state.account_id}没有检测到明显的漏水迹象。"
    
    elif "用水趋势" in question:
        patterns = context["usage_patterns"]
        trend = patterns.get("trend", "未知")
        
        answer = f"账户{state.account_id}的用水趋势分析：\n\n"
        answer += f"整体趋势: {trend}\n\n"
        
        if trend == "上升":
            answer += "用水量呈上升趋势可能的原因：\n"
            answer += "1. 家庭成员增加\n"
            answer += "2. 用水习惯改变\n"
            answer += "3. 季节性因素（如夏季用水增加）\n"
            answer += "4. 新增用水设备（如洗碗机、浇花系统等）\n"
            answer += "5. 可能存在未发现的漏水问题\n"
        elif trend == "下降":
            answer += "用水量呈下降趋势可能的原因：\n"
            answer += "1. 家庭成员减少\n"
            answer += "2. 采取了节水措施\n"
            answer += "3. 更换了节水型设备\n"
            answer += "4. 季节性因素（如冬季用水减少）\n"
            answer += "5. 长期不在家\n"
        elif trend == "稳定":
            answer += "用水量保持稳定表明：\n"
            answer += "1. 用水习惯没有明显变化\n"
            answer += "2. 家庭成员数量稳定\n"
            answer += "3. 没有新增或减少主要用水设备\n"
        else:
            answer += "数据不足，无法进行可靠的趋势分析。需要更长时间的数据来确定趋势。"
    
    elif "优化" in question or "效率" in question:
        answer = f"针对账户{state.account_id}的用水效率优化建议：\n\n"
        
        # 基于分析结果提供个性化建议
        patterns = context["usage_patterns"]
        
        if patterns.get("anomalies", []):
            answer += "1. 注意异常用水情况，查找原因并采取措施\n"
        
        if patterns.get("leakage_periods", []):
            answer += "2. 检查并修复可能的漏水问题\n"
        
        if patterns.get("trend") == "上升":
            answer += "3. 考虑安装节水设备，如节水龙头、低流量马桶等\n"
        
        # 通用建议
        answer += "4. 安装智能水表，实时监控用水情况\n"
        answer += "5. 收集雨水用于园艺和清洁\n"
        answer += "6. 洗衣机和洗碗机满载再使用\n"
        answer += "7. 缩短淋浴时间，使用节水型淋浴喷头\n"
        answer += "8. 修复滴水的水龙头和漏水的马桶\n"
        answer += "9. 使用节水型家电\n"
        answer += "10. 考虑安装灰水回收系统\n"
    
    else:
        answer = "抱歉，我无法理解这个问题。请尝试询问关于用水模式、异常用水、漏水检测、用水趋势或优化建议的问题。"
    
    # 保存回答
    state.answers.append({"question": state.current_question, "answer": answer})
    
    # 移动到下一个问题
    if state.questions and state.current_question in state.questions:
        current_index = state.questions.index(state.current_question)
        if current_index + 1 < len(state.questions):
            state.current_question = state.questions[current_index + 1]
        else:
            state.current_question = None
    
    return state

def generate_recommendations(state: GraphState) -> GraphState:
    """生成用水建议"""
    # 直接返回原始状态，不生成任何建议
    return state

def generate_final_report(state: GraphState) -> GraphState:
    """生成最终分析报告"""
    if not state.analysis_results or not state.answers:
        return state
    
    # 准备报告内容
    report = {
        "account_id": state.account_id,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "basic_info": state.analysis_results.get("basic_info", {}),
        "usage_patterns": state.analysis_results.get("usage_patterns", {}),
        "qa_section": state.answers,
        "recommendations": state.recommendations,
        "charts": state.charts
    }
    
    state.final_report = report
    return state

# 定义路由逻辑
def router(state: GraphState) -> str:
    """决定下一步执行哪个节点"""
    if not state.account_id or not state.data:
        return "error"
    
    if not state.analysis_results.get("usage_patterns"):
        return "analyze_usage_patterns"
    
    if not state.charts:
        return "generate_charts"
    
    if not state.questions:
        return "generate_questions"
    
    if state.current_question:
        return "answer_question"
    
    if not state.recommendations:
        return "generate_recommendations"
    
    if not state.final_report:
        return "generate_final_report"
    
    return END

# 创建工作流
def create_water_analysis_workflow():
    """创建水表分析工作流"""
    # 创建图
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("get_account_data", get_account_data)
    workflow.add_node("analyze_usage_patterns", analyze_usage_patterns)
    workflow.add_node("generate_charts", generate_charts)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("generate_final_report", generate_final_report)
    workflow.add_node("error", lambda state: state)
    
    # 设置入口点
    workflow.set_entry_point("get_account_data")
    
    # 添加边
    workflow.add_edge("get_account_data", router)
    workflow.add_edge("analyze_usage_patterns", router)
    workflow.add_edge("generate_charts", router)
    workflow.add_edge("generate_questions", router)
    workflow.add_edge("answer_question", router)
    workflow.add_edge("generate_recommendations", router)
    workflow.add_edge("generate_final_report", END)
    workflow.add_edge("error", END)
    
    # 编译工作流
    return workflow.compile()

def analyze_account(account_id: str) -> Dict:
    """分析指定账户的水表数据"""
    # 创建工作流
    workflow = create_water_analysis_workflow()
    
    # 创建初始状态
    initial_state = GraphState()
    
    # 运行工作流
    final_state = workflow.invoke({
        "account_id": account_id
    })
    
    # 返回最终报告
    return final_state.final_report

# 添加单个账户分析函数，用于并行处理
def analyze_single_account(account_id: str, api_key: str) -> Dict[str, Any]:
    """分析单个账户，用于并行处理"""
    # 创建新的LLM实例，因为每个线程需要独立的实例
    llm = DeepSeekR1(api_key=api_key)
    
    reader = WaterMeterDataReader()
    try:
        # 获取数据
        df = reader.get_readings_as_dataframe(account_id)
        if df.empty:
            return {"account_id": account_id, "error": f"账户 {account_id} 没有数据"}
        
        # 基本统计
        total_usage = df['current_usage'].sum()
        avg_usage = df['current_usage'].mean()
        max_usage = df['current_usage'].max()
        zero_usage_days = (df['current_usage'] == 0).sum()
        
        # 准备数据摘要
        data_summary = {
            "account_id": account_id,
            "total_records": len(df),
            "date_range": f"{df['reading_time'].min().strftime('%Y-%m-%d')} 到 {df['reading_time'].max().strftime('%Y-%m-%d')}",
            "total_usage": float(total_usage),
            "average_usage": float(avg_usage),
            "max_usage": float(max_usage),
            "zero_usage_days": int(zero_usage_days),
            "daily_usage": [
                {"date": row['reading_time'].strftime('%Y-%m-%d'), "usage": float(row['current_usage'])}
                for _, row in df.iterrows()
            ]
        }
        
        print(f"正在使用DeepSeek-R1模型分析账户 {account_id} 的用水数据...")
        
        # 使用DeepSeek-R1模型分析数据
        prompt = f"""
        你是一个水表数据分析专家。我将给你一些关于账户 {account_id} 的水表数据。
        请分析这些数据，重点关注是否存在漏水的可能性。
        
        漏水的典型特征包括：
        1. 连续多天（通常3天以上）的小量用水（低于平均用水量的50%但大于0）
        2. 夜间用水量异常（如果有时间数据）
        3. 用水量突然增加但没有明显原因
        
        以下是账户的用水数据摘要：
        {json.dumps(data_summary, ensure_ascii=False, indent=2)}
        
        请分析这些数据，并给出以下内容：
        1. 是否存在漏水的可能性（是/否）
        2. 漏水的可能性有多大（低/中/高）
        3. 支持你结论的证据
        4. 如果可能存在漏水，请指出可能的漏水时间段
        
        请以JSON格式返回你的分析结果，格式如下：
        {{
            "leakage_detected": true/false,
            "leakage_probability": "低"/"中"/"高",
            "evidence": "你的分析依据...",
            "leakage_periods": [
                {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "description": "描述..."}}
            ]
        }}
        
        只返回JSON格式的结果，不要有其他文字。
        """
        
        # 调用DeepSeek-R1模型
        try:
            response_text = llm.generate([
                {"role": "system", "content": "你是一个水表数据分析专家，专注于漏水检测。请只返回JSON格式的结果。"},
                {"role": "user", "content": prompt}
            ])
            
            print("DeepSeek-R1模型分析完成")
            
            # 解析JSON响应
            try:
                # 尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group(0))
                else:
                    # 如果无法提取JSON，则创建一个默认结果
                    analysis_result = {
                        "leakage_detected": False,
                        "leakage_probability": "低",
                        "evidence": "模型未返回有效的JSON格式结果",
                        "leakage_periods": []
                    }
            except json.JSONDecodeError:
                # 如果JSON解析失败，则创建一个默认结果
                analysis_result = {
                    "leakage_detected": False,
                    "leakage_probability": "低",
                    "evidence": "模型返回的结果无法解析为JSON",
                    "leakage_periods": []
                }
            
            # 创建最终报告
            return {
                "account_id": account_id,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "leakage_detected": analysis_result.get("leakage_detected", False),
                "leakage_probability": analysis_result.get("leakage_probability", "低"),
                "evidence": analysis_result.get("evidence", ""),
                "leakage_periods": analysis_result.get("leakage_periods", []),
                "model_used": "DeepSeek-R1"
            }
        except Exception as e:
            print(f"账户 {account_id} 分析出错: {str(e)}")
            return {
                "account_id": account_id,
                "error": f"分析出错: {str(e)}"
            }
    finally:
        reader.close()

# 修改批量分析函数，使用并行处理
def batch_analyze_accounts(account_ids: List[str], llm: DeepSeekR1) -> Dict[str, Dict[str, Any]]:
    """批量并行分析多个账户的漏水情况"""
    results = {}
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    print(f"开始并行分析 {len(account_ids)} 个账户的漏水情况...")
    
    # 使用ThreadPoolExecutor进行并行处理
    # 设置max_workers为5，表示最多同时分析5个账户
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任务
        future_to_account = {
            executor.submit(analyze_single_account, account_id, api_key): account_id
            for account_id in account_ids
        }
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_account):
            account_id = future_to_account[future]
            try:
                result = future.result()
                results[account_id] = result
            except Exception as e:
                print(f"账户 {account_id} 处理失败: {str(e)}")
                results[account_id] = {
                    "account_id": account_id,
                    "error": f"处理失败: {str(e)}"
                }
    
    print(f"批量分析完成，共分析了 {len(results)} 个账户")
    return results

# 修改display_batch_results函数，显示更详细的分析结果
def display_batch_results(results: Dict[str, Dict[str, Any]]):
    """显示批量分析结果"""
    print("\n" + "="*50)
    print("批量漏水检测结果")
    print("="*50)
    
    # 统计漏水账户数量
    leakage_accounts = [account_id for account_id, result in results.items() 
                        if result.get("leakage_detected", False)]
    
    print(f"分析账户总数: {len(results)}")
    print(f"检测到漏水的账户数: {len(leakage_accounts)}")
    
    if leakage_accounts:
        print("\n⚠️ 以下账户可能存在漏水:")
        for account_id in leakage_accounts:
            result = results[account_id]
            print(f"\n账户 {account_id}:")
            print(f"  分析模型: {result.get('model_used', 'N/A')}")
            print(f"  漏水可能性: {result.get('leakage_probability', 'N/A')}")
            print(f"  分析依据: {result.get('evidence', 'N/A')}")
            
            if result.get("leakage_periods"):
                print("  可能的漏水时间段:")
                for period in result.get("leakage_periods"):
                    print(f"  - 从 {period.get('start_date')} 到 {period.get('end_date')}")
                    if 'description' in period:
                        print(f"    描述: {period.get('description')}")
    else:
        print("\n✅ 未检测到任何账户存在漏水")
    
    print("\n" + "="*50)

# 修改批量分析函数，减少一次处理的账户数量并增加超时设置
def batch_analyze_accounts_at_once(account_ids: List[str], llm: DeepSeekR1) -> Dict[str, Dict[str, Any]]:
    """一次性分析多个账户的漏水情况，但每次只处理少量账户"""
    results = {}
    
    # 将账户分成更小的批次，每批最多3个账户
    batch_size = 10
    account_batches = [account_ids[i:i+batch_size] for i in range(0, len(account_ids), batch_size)]
    
    print(f"将 {len(account_ids)} 个账户分成 {len(account_batches)} 个批次进行分析...")
    
    for batch_index, batch_account_ids in enumerate(account_batches):
        print(f"\n处理批次 {batch_index+1}/{len(account_batches)}，包含 {len(batch_account_ids)} 个账户...")
        
        # 收集当前批次账户的数据
        accounts_data = {}
        reader = WaterMeterDataReader()
        try:
            for account_id in batch_account_ids:
                print(f"收集账户 {account_id} 的数据...")
                
                # 获取数据
                df = reader.get_readings_as_dataframe(account_id)
                if df.empty:
                    results[account_id] = {"error": f"账户 {account_id} 没有数据"}
                    continue
                
                # 基本统计
                total_usage = df['current_usage'].sum()
                avg_usage = df['current_usage'].mean()
                max_usage = df['current_usage'].max()
                zero_usage_days = (df['current_usage'] == 0).sum()
                
                # 准备数据摘要
                accounts_data[account_id] = {
                    "account_id": account_id,
                    "total_records": len(df),
                    "date_range": f"{df['reading_time'].min().strftime('%Y-%m-%d')} 到 {df['reading_time'].max().strftime('%Y-%m-%d')}",
                    "total_usage": float(total_usage),
                    "average_usage": float(avg_usage),
                    "max_usage": float(max_usage),
                    "zero_usage_days": int(zero_usage_days),
                    "daily_usage": [
                        {"date": row['reading_time'].strftime('%Y-%m-%d'), "usage": float(row['current_usage'])}
                        for _, row in df.iterrows()
                    ]
                }
        finally:
            reader.close()
        
        if not accounts_data:
            print("当前批次没有有效的账户数据可供分析")
            continue
        
        print(f"正在使用DeepSeek-R1模型分析当前批次的 {len(accounts_data)} 个账户...")
        
        # 使用DeepSeek-R1模型分析当前批次的账户数据
        prompt = f"""
        角色：数据观测工程师
        
        请根据以下数据分析多个账户是否存在漏水问题。
        
        说明：数据包含抄表时间、当前用量。
        漏水分为管道漏水、室内器具漏水。
        住户分为民居民用水和商业用水。
        
        漏水判断标准（请严格按照以下标准判断）：
        1. 居民用水漏水类型：
           - 管道漏水：如果连续多天（3天以上）的用水量在0.1到0.5之间，且随着时间推移逐渐增加到3以上，则判定为管道漏水。
           - 室内器具漏水：如果某天用水量突然超过5，或者连续多天每天用水量都在0.5到1之间，则判定为室内器具漏水。
           - 正常情况：5天内总用水量低于1。
        
        2. 商业用水漏水类型：
           - 如果连续多天的用水量比之前同期增加一倍以上，则判定为漏水。
           - 正常情况：10天内总用水量在10到50之间。
        
        请注意：这是连续的每天用量数据，不要直接当成是一天的用量分析。
        请仔细检查每个账户的用水模式，特别关注异常的用水量变化。
        
        重要提示：根据历史经验，至少有30%的账户存在漏水问题，请不要轻易判断为无漏水。
        
        以下是所有账户的用水数据摘要：
        {json.dumps(accounts_data, ensure_ascii=False, indent=2)}
        
        请分析每个账户的数据，并给出漏水可能性评级：
        - 无：完全没有漏水迹象
        - 低：有轻微异常，但可能不是漏水
        - 中：有一些漏水迹象，需要关注
        - 高：明显的漏水迹象，需要立即处理
        
        请以JSON格式返回你的分析结果，格式如下：
        {{
            "accounts": [
                {{
                    "account_id": "账户ID",
                    "leakage_probability": "无"/"低"/"中"/"高"
                }},
                // 其他账户...
            ]
        }}
        
        只返回JSON格式的结果，不要有其他文字。
        """
        
        # 调用DeepSeek-R1模型，增加超时设置
        try:
            # 增加超时时间到120秒
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # 创建一个带有重试机制的会话
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('https://', adapter)
            
            # 设置DeepSeek API的请求参数
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-ai/DeepSeek-R1",
                "messages": [
                    {"role": "system", "content": "你是一个水表数据分析专家，专注于漏水检测。请只返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # 降低温度以获得更确定性的结果
                "max_tokens": 4000   # 增加最大令牌数
            }
            
            # 发送请求，设置较长的超时时间
            response = session.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120  # 设置120秒超时
            )
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"]
                print("DeepSeek-R1模型分析完成")
                
                # 解析JSON响应
                try:
                    # 尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group(0))
                        
                        # 处理分析结果
                        if "accounts" in analysis_result:
                            for account_analysis in analysis_result["accounts"]:
                                account_id = account_analysis.get("account_id")
                                if account_id and account_id in accounts_data:
                                    results[account_id] = {
                                        "account_id": account_id,
                                        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "leakage_probability": account_analysis.get("leakage_probability", "无"),
                                        "model_used": "DeepSeek-R1 (批量分析)"
                                    }
                        else:
                            print("模型返回的JSON中没有accounts字段")
                            # 创建默认结果
                            for account_id in accounts_data:
                                results[account_id] = {
                                    "account_id": account_id,
                                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "leakage_probability": "无"
                                }
                    else:
                        print("无法从模型响应中提取JSON")
                        # 创建默认结果
                        for account_id in accounts_data:
                            results[account_id] = {
                                "account_id": account_id,
                                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "leakage_probability": "无"
                            }
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}")
                    # 创建默认结果
                    for account_id in accounts_data:
                        results[account_id] = {
                            "account_id": account_id,
                            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "leakage_probability": "无"
                        }
            else:
                print(f"API请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                # 创建默认结果
                for account_id in accounts_data:
                    results[account_id] = {
                        "account_id": account_id,
                        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "leakage_probability": "无"
                    }
        except Exception as e:
            print(f"调用模型时出错: {str(e)}")
            # 创建默认结果
            for account_id in accounts_data:
                results[account_id] = {
                    "account_id": account_id,
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "leakage_probability": "无"
                }
        
        # 在批次之间添加延迟，避免API限制
        if batch_index < len(account_batches) - 1:
            delay = 5  # 5秒延迟
            print(f"等待 {delay} 秒后处理下一批次...")
            import time
            time.sleep(delay)
    
    print(f"批量分析完成，共分析了 {len(results)} 个账户")
    return results

# 修改main函数，只保留批量分析选项
def main():
    """主函数"""
    # 加载环境变量（再次确认加载）
    load_dotenv(verbose=True)
    
    print("水表数据批量漏水检测工具")
    print("="*50)
    
    # 直接设置环境变量，确保API密钥可用
    os.environ["DEEPSEEK_API_KEY"] = "sk-qsybuxxdlcvuhmtmbollzxzxvkwzqzxbkmbockxpujpcjyfk"
    
    # 初始化DeepSeek-R1模型
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    print(f"使用API密钥: {api_key[:5]}...{api_key[-5:]}")
    
    llm = DeepSeekR1(api_key=api_key)
    
    # 测试模型连接
    print("测试DeepSeek-R1模型连接...")
    try:
        # 简单测试模型是否能正常工作
        test_response = llm.generate([
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "你好，请用一句话介绍自己。"}
        ])
        print("✅ 模型连接成功!")
        print(f"模型响应: {test_response[:50]}...")
    except Exception as e:
        print(f"❌ 模型连接失败: {str(e)}")
        return
    
    # 获取所有账户
    reader = WaterMeterDataReader()
    try:
        accounts = reader.get_all_accounts()
    finally:
        reader.close()
    
    while True:
        print("\n请选择操作:")
        print("1. 并行分析10个账户")
        print("2. 一次性分析10个账户")
        print("0. 退出")
        
        choice = input("\n请选择 (0-2): ")
        
        if choice == '0':
            break
        elif choice == '1':
            # 并行分析10个账户
            all_account_ids = [account['id'] for account in accounts]
            batch_size = min(10, len(all_account_ids))
            batch_account_ids = all_account_ids[:batch_size]
            
            print(f"并行分析前 {batch_size} 个账户...")
            results = batch_analyze_accounts(batch_account_ids, llm)
            display_batch_results(results)
            
            # 保存报告到文件
            filename = "batch_accounts_leakage_report.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存到文件: {filename}")
        elif choice == '2':
            # 一次性分析10个账户
            all_account_ids = [account['id'] for account in accounts]
            batch_size = min(10, len(all_account_ids))
            batch_account_ids = all_account_ids[:batch_size]
            
            print(f"一次性分析前 {batch_size} 个账户...")
            results = batch_analyze_accounts_at_once(batch_account_ids, llm)
            display_batch_results(results)
            
            # 保存报告到文件
            filename = "batch_accounts_at_once_leakage_report.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存到文件: {filename}")
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    main() 