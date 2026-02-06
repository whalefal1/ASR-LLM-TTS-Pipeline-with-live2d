from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, before_model
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any
import os

# 加载环境变量
load_dotenv()

# 模型配置
model_configs = {
    "base": {
        "model_name": "glm-4-flashx-250414",
        "temperature": 0.7,
        "max_tokens": 1024
    },
    "advanced": {
        "model_name": "glm-4",
        "temperature": 0.7,
        "max_tokens": 2048
    }
}

# 关键词列表
advanced_keywords = ["专业", "高级", "详细", "深入", "全面", "完整", "详尽", "专业级", "高级别", "详细说明"]


def contains_advanced_keywords(prompt):
    """
    检查prompt中是否包含高级关键词
    :param prompt: 用户输入的prompt
    :return: bool，True表示包含高级关键词
    """
    for keyword in advanced_keywords:
        if keyword in prompt:
            return True
    return False


# 加载API密钥和基础URL
api_key = os.getenv("ZHIPU_API_KEY")
base_url = os.getenv("ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4")

# 初始化模型实例
base_model = ChatOpenAI(
    api_key=api_key,
    model_name=model_configs["base"]["model_name"],
    base_url=base_url,
    temperature=model_configs["base"]["temperature"],
    max_tokens=model_configs["base"]["max_tokens"]
)

advanced_model = ChatOpenAI(
    api_key=api_key,
    model_name=model_configs["advanced"]["model_name"],
    base_url=base_url,
    temperature=model_configs["advanced"]["temperature"],
    max_tokens=model_configs["advanced"]["max_tokens"]
)

# 尝试获取Tavily API密钥
TAVILY_API_KEY = os.getenv("SEARCH_TOOL_INDEX_NAME")

# 初始化搜索工具列表
search_tools = []

# 只有当API密钥存在时，才初始化Tavily搜索工具
if TAVILY_API_KEY:
    try:
        search_tool = TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True
        )
        search_tools.append(search_tool)
        print("Tavily搜索工具初始化成功")
    except Exception as e:
        print(f"Tavily搜索工具初始化失败: {str(e)}")
else:
    print("未配置TAVILY_API_KEY环境变量，搜索功能将不可用")


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    消息截断中间件，保留最近的消息以适应上下文窗口
    """
    messages = state["messages"]
    
    # 只保留最近的10条消息（5轮对话）
    if len(messages) <= 10:
        return None  # 无需截断
    
    # 保留最近的10条消息
    recent_messages = messages[-10:]
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages
        ]
    }


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    动态模型选择中间件
    仅根据prompt关键词选择合适的模型
    """
    # 获取当前查询内容
    current_query = ""
    
    if request.state and "messages" in request.state:
        messages = request.state["messages"]
        
        # 获取最后一条用户消息
        for msg in reversed(messages):
            # 检查消息类型，而不是使用get方法
            if hasattr(msg, "content"):
                # 检查是否为用户消息（通过消息类型或属性）
                if hasattr(msg, "role") and msg.role == "user":
                    current_query = msg.content
                    break
                # 或者通过类名判断
                elif msg.__class__.__name__ == "HumanMessage":
                    current_query = msg.content
                    break
    
    # 检查是否需要使用高级模型（仅根据关键词）
    use_advanced = contains_advanced_keywords(current_query)
    
    # 选择模型
    if use_advanced:
        selected_model = advanced_model
        model_type = "advanced"
        reason = "包含高级关键词"
    else:
        selected_model = base_model
        model_type = "base"
        reason = "标准查询"
    
    print(f"使用远程模型类型: {model_type} ({model_configs[model_type]['model_name']})")
    print(f"选择原因: {reason}")
    
    # 更新请求中的模型（使用推荐的override方法）
    updated_request = request.override(model=selected_model)
    
    # 调用原始处理器
    return handler(updated_request)


# 创建内存检查点保存器，用于短期记忆
memory_saver = InMemorySaver()

# 创建智能体（根据API密钥配置决定是否添加搜索工具）
agent = create_agent(
    model=base_model,  # 默认模型
    tools=search_tools,  # 添加搜索工具（如果API密钥已配置）
    middleware=[trim_messages, dynamic_model_selection],  # 添加消息截断和动态模型选择中间件
    checkpointer=memory_saver,  # 添加短期记忆功能
    state_schema=AgentState  # 使用默认的AgentState
)


class RemoteLLMModel:
    """
    远程模型调用类，封装智能体调用
    仅用于远程模型API调用，本地模型调用请使用ollama_llm.py
    """
    def __init__(self):
        self.agent = agent
    
    def invoke(self, query, conversation_history=None, thread_id=None):
        """
        统一调用接口，根据query动态选择远程模型
        :param query: 用户的查询文本
        :param conversation_history: 对话历史，可选
        :param thread_id: 对话线程ID（已取消线程隔离，所有对话共享同一记忆）
        :return: 模型的回复文本
        """
        try:
            # 构建消息列表
            messages = []
            
            # 添加对话历史
            if conversation_history:
                messages.extend(conversation_history)
            
            # 添加当前查询
            messages.append({
                "role": "user",
                "content": query
            })
            
            # 构建RunnableConfig，使用固定的thread_id取消线程隔离
            config = {
                "configurable": {
                    "thread_id": "shared_memory"  # 固定使用同一个thread_id，取消线程隔离
                }
            }
            
            # 调用智能体，传入config以支持短期记忆
            result = self.agent.invoke(
                {"messages": messages},
                config=config
            )
            
            # 返回最新的助手回复
            if result and "messages" in result:
                for msg in reversed(result["messages"]):
                    # 检查消息类型，处理不同的消息对象
                    if hasattr(msg, "content"):
                        if hasattr(msg, "role") and msg.role == "assistant":
                            return msg.content
                        elif msg.__class__.__name__ == "AIMessage":
                            return msg.content
            
            return str(result)
        except Exception as e:
            print(f"调用远程大模型出错：{str(e)}")
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"


# 创建全局实例
remote_llm = RemoteLLMModel()


def call_remote_llm(query, conversation_history=None, thread_id=None):
    """
    外部调用接口，直接使用全局实例
    仅用于远程模型API调用，本地模型调用请使用call_ollama_llm
    :param query: 用户的查询文本
    :param conversation_history: 对话历史，可选
    :param thread_id: 对话线程ID（已取消线程隔离，所有对话共享同一记忆，该参数不再起作用）
    :return: 模型的回复文本
    """
    return remote_llm.invoke(query, conversation_history, thread_id)


def get_remote_llm_instance():
    """
    获取远程LLM实例
    :return: RemoteLLMModel实例
    """
    return remote_llm



