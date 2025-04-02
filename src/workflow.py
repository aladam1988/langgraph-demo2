from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
from src.agents import DBManager

# 定义状态类型
class GraphState(TypedDict):
    account_id: str
    account_data: Dict[str, Any]
    transaction_data: Dict[str, Any]
    analysis_result: Dict[str, Any]
    messages: List[Any]

# 初始化数据库管理器
db_manager = DBManager("data/accounts.db")

# 1. 读取账户数据
def fetch_account_data(state: GraphState) -> GraphState:
    """从数据库读取账户数据"""
    account_id = state["account_id"]
    
    # 读取账户基本信息
    account_data = db_manager.fetch_data(
        account_id, 
        "accounts", 
        "SELECT * FROM accounts WHERE id = ?"
    )
    
    # 更新状态
    return {
        **state,
        "account_data": account_data
    }

# 2. 读取交易数据
def fetch_transaction_data(state: GraphState) -> GraphState:
    """从数据库读取交易数据"""
    account_id = state["account_id"]
    
    # 读取交易记录
    transaction_data = db_manager.fetch_data(
        account_id, 
        "transactions", 
        "SELECT * FROM transactions WHERE account_id = ? ORDER BY timestamp DESC"
    )
    
    # 更新状态
    return {
        **state,
        "transaction_data": transaction_data
    }

# 3. 分析数据
def analyze_data(state: GraphState) -> GraphState:
    """使用LLM分析数据"""
    # 获取账户和交易数据
    account_data = state["account_data"]
    transaction_data = state["transaction_data"]
    
    # 准备消息
    messages = state.get("messages", [])
    messages.append(
        SystemMessage(content="你是一个金融分析专家，负责分析账户和交易数据，识别潜在风险。")
    )
    
    # 准备数据摘要
    data_summary = f"""
    账户信息: {json.dumps(account_data, ensure_ascii=False)}
    
    最近交易: {json.dumps(transaction_data, ensure_ascii=False)}
    
    请分析这些数据，识别任何潜在的风险或异常模式。提供一个简洁的风险评估报告。
    """
    
    messages.append(HumanMessage(content=data_summary))
    
    # 使用LLM进行分析
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)
    
    messages.append(response)
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "analysis_result": {
            "result_type": "risk_assessment",
            "result_data": response.content
        }
    }

# 4. 写入分析结果
def save_analysis_result(state: GraphState) -> GraphState:
    """将分析结果写入数据库"""
    account_id = state["account_id"]
    analysis_result = state["analysis_result"]
    
    # 写入分析结果
    result = db_manager.write_data("analysis_results", {
        "account_id": account_id,
        "result_type": analysis_result["result_type"],
        "result_data": analysis_result["result_data"],
        "metadata": {"created_by": "langgraph_workflow"}
    })
    
    # 更新状态
    return {
        **state,
        "save_result": result
    }

# 5. 决策节点：是否需要进一步分析
def should_analyze_further(state: GraphState) -> str:
    """决定是否需要进一步分析"""
    analysis_result = state["analysis_result"]
    
    # 检查分析结果中是否包含高风险关键词
    high_risk_keywords = ["高风险", "异常", "欺诈", "可疑", "警告"]
    
    for keyword in high_risk_keywords:
        if keyword in analysis_result["result_data"]:
            return "需要进一步分析"
    
    return "分析完成"

# 6. 进一步分析
def detailed_analysis(state: GraphState) -> GraphState:
    """进行更详细的分析"""
    # 获取之前的消息
    messages = state.get("messages", [])
    
    # 添加请求详细分析的消息
    messages.append(
        HumanMessage(content="检测到潜在高风险。请提供更详细的分析，包括具体的风险因素和建议的行动。")
    )
    
    # 使用LLM进行详细分析
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)
    
    messages.append(response)
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "analysis_result": {
            "result_type": "detailed_risk_assessment",
            "result_data": response.content
        }
    }

# 创建工作流图
def create_workflow():
    """创建工作流图"""
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("fetch_account", fetch_account_data)
    workflow.add_node("fetch_transactions", fetch_transaction_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("save_result", save_analysis_result)
    workflow.add_node("detailed_analysis", detailed_analysis)
    
    # 添加边
    workflow.add_edge("fetch_account", "fetch_transactions")
    workflow.add_edge("fetch_transactions", "analyze")
    workflow.add_edge("analyze", "should_analyze_further")
    workflow.add_conditional_edges(
        "should_analyze_further",
        {
            "需要进一步分析": "detailed_analysis",
            "分析完成": "save_result"
        }
    )
    workflow.add_edge("detailed_analysis", "save_result")
    
    # 设置入口节点
    workflow.set_entry_point("fetch_account")
    
    return workflow.compile()

# 运行工作流
def run_analysis_workflow(account_id: str):
    """运行分析工作流"""
    # 创建工作流
    graph = create_workflow()
    
    # 初始状态
    initial_state = {
        "account_id": account_id,
        "account_data": {},
        "transaction_data": {},
        "analysis_result": {},
        "messages": []
    }
    
    # 运行工作流
    for event in graph.stream(initial_state):
        if event["type"] == "node":
            print(f"执行节点: {event['node']}")
    
    # 获取最终状态
    final_state = event["state"]
    
    return final_state 