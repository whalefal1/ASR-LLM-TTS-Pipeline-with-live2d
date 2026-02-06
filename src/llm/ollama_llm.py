#!/usr/bin/env python3
# coding=utf-8
"""
基于 Ollama 的本地 LLM 抽象层
支持完整的 LangChain Agent 和 Tools 功能
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langchain_core.tools import Tool
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('OllamaLLM')


class OllamaLLM:
    """
    基于 Ollama 的本地 LLM 抽象类
    完全支持 LangChain Agent 和 Tools 功能
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5vl:7b",
                 base_url: str = "http://localhost:11434",
                 tools: Optional[List[Tool]] = None,
                 temperature: float = 0.7,
                 verbose: bool = False):
        """
        初始化 Ollama LLM 抽象
        
        Args:
            model_name (str): Ollama 模型名称
            base_url (str): Ollama 服务地址
            tools (Optional[List[Tool]]): LangChain 工具列表
            temperature (float): 温度参数
            verbose (bool): 是否启用详细日志
        """
        self.model_name = model_name
        self.base_url = base_url
        self.tools = tools or []
        self.temperature = temperature
        self.verbose = verbose
        
        # 初始化组件
        self._init_llm()
        self._init_agent()
    
    def _init_llm(self):
        """
        初始化 Ollama LLM
        """
        try:
            logger.info(f"开始初始化 Ollama LLM: {self.model_name}")
            logger.info(f"Ollama 服务地址: {self.base_url}")
            
            # 创建 Ollama LLM 实例
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            logger.info("Ollama LLM 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化 Ollama LLM 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _init_agent(self):
        """
        初始化 LangChain Agent
        """
        try:
            if self.tools:
                logger.info(f"初始化 Agent，工具数量: {len(self.tools)}")
                self.agent = create_agent(
                    model=self.llm,
                    tools=self.tools
                )
                logger.info("Agent 初始化完成（支持工具调用）")
            else:
                logger.info("未配置工具，跳过 Agent 初始化")
                self.agent = None
        except Exception as e:
            logger.error(f"初始化 Agent 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.agent = None
    
    def invoke(self, query: str, conversation_history: Optional[List[Dict]] = None):
        """
        调用本地 LLM
        
        Args:
            query (str): 用户查询
            conversation_history (Optional[List[Dict]]): 对话历史
            
        Returns:
            str: 模型响应
        """
        try:
            logger.info(f"调用 Ollama LLM，查询: {query}")
            
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
            
            # 如果有 Agent 且工具可用，使用 Agent 调用
            if self.agent and self.tools:
                logger.info("使用 Agent 调用 LLM（支持工具调用）")
                try:
                    result = self.agent.invoke({"messages": messages})
                    
                    # 提取助手回复
                    if result and "messages" in result:
                        for msg in reversed(result["messages"]):
                            if hasattr(msg, "content"):
                                if hasattr(msg, "role") and msg.role == "assistant":
                                    return msg.content
                                elif msg.__class__.__name__ == "AIMessage":
                                    return msg.content
                    return str(result)
                except Exception as e:
                    logger.warning(f"Agent 调用失败，回退到直接 LLM: {str(e)}")
                    # 回退到直接使用 LLM
            
            # 直接使用 LLM 调用
            logger.info("直接调用 LLM")
            response = self.llm.invoke(messages)
            
            # 提取响应文本
            if hasattr(response, "content"):
                return response.content
            return str(response)
                
        except Exception as e:
            logger.error(f"调用 Ollama LLM 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"
    
    def add_tool(self, tool: Tool):
        """
        添加工具
        
        Args:
            tool (Tool): LangChain Tool 对象
        """
        self.tools.append(tool)
        # 重新初始化 Agent
        self._init_agent()


class OllamaLLMFactory:
    """
    Ollama LLM 工厂类
    用于创建和管理不同的 Ollama LLM 实例
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, name: str, model_name: str = "qwen2.5vl:7b", **kwargs):
        """
        获取或创建 Ollama LLM 实例
        
        Args:
            name (str): 实例名称
            model_name (str): Ollama 模型名称
            **kwargs: 其他参数
            
        Returns:
            OllamaLLM: Ollama LLM 实例
        """
        if name not in cls._instances:
            cls._instances[name] = OllamaLLM(model_name, **kwargs)
        return cls._instances[name]
    
    @classmethod
    def list_instances(cls):
        """
        列出所有实例
        
        Returns:
            List[str]: 实例名称列表
        """
        return list(cls._instances.keys())
    
    @classmethod
    def remove_instance(cls, name: str):
        """
        移除实例
        
        Args:
            name (str): 实例名称
        """
        if name in cls._instances:
            del cls._instances[name]


# 示例工具定义
def create_ollama_tools():
    """
    创建示例工具
    
    Returns:
        List[Tool]: 工具列表
    """
    from langchain_core.tools import Tool
    
    # 示例工具 1: 计算工具
    def calculate(expression: str) -> str:
        """
        计算数学表达式
        
        Args:
            expression (str): 数学表达式
            
        Returns:
            str: 计算结果
        """
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算失败: {str(e)}"
    
    # 示例工具 2: 时间工具
    def get_current_time() -> str:
        """
        获取当前时间
        
        Returns:
            str: 当前时间
        """
        import datetime
        return f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # 示例工具 3: 搜索工具
    def search_web(query: str) -> str:
        """
        网络搜索（示例）
        
        Args:
            query (str): 搜索查询
            
        Returns:
            str: 搜索结果
        """
        # 这里可以集成真实的搜索 API
        return f"搜索结果: 关于 '{query}' 的信息"
    
    tools = [
        Tool(
            name="Calculate",
            func=calculate,
            description="用于计算数学表达式，输入为数学表达式字符串"
        ),
        Tool(
            name="GetCurrentTime",
            func=get_current_time,
            description="用于获取当前时间"
        ),
        Tool(
            name="SearchWeb",
            func=search_web,
            description="用于网络搜索，输入为搜索查询"
        )
    ]
    
    return tools


# 创建全局实例
def get_default_ollama_llm(model_name: str = "qwen2.5vl:7b", **kwargs):
    """
    获取默认的 Ollama LLM 实例
    
    Args:
        model_name (str): Ollama 模型名称
        **kwargs: 其他参数
        
    Returns:
        OllamaLLM: Ollama LLM 实例
    """
    return OllamaLLMFactory.get_instance("default", model_name, **kwargs)


def call_ollama_llm(query: str, conversation_history: Optional[List[Dict]] = None, **kwargs):
    """
    外部调用接口
    
    Args:
        query (str): 用户查询
        conversation_history (Optional[List[Dict]]): 对话历史
        **kwargs: 其他参数
        
    Returns:
        str: 模型响应
    """
    llm = get_default_ollama_llm(**kwargs)
    return llm.invoke(query, conversation_history)
