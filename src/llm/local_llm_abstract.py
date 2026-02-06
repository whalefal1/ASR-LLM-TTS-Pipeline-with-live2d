#!/usr/bin/env python3
# coding=utf-8
"""
本地 LLM 可复用抽象层
支持 LangChain 的 tools 和 memory 功能
"""

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.agents import create_agent, AgentState
from langchain_core.tools import Tool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('LocalLLMAbstract')


class LocalLLMAbstract:
    """
    本地 LLM 可复用抽象类
    支持 LangChain 的 tools 和 memory 功能
    """
    
    def __init__(self, 
                 model_path: str, 
                 device: Optional[str] = None, 
                 tools: Optional[List[Tool]] = None,
                 memory_key: str = "chat_history",
                 verbose: bool = False):
        """
        初始化本地 LLM 抽象
        
        Args:
            model_path (str): 本地模型路径
            device (Optional[str]): 运行设备，默认为自动检测
            tools (Optional[List[Tool]]): LangChain 工具列表
            memory_key (str): 记忆模块的键名
            verbose (bool): 是否启用详细日志
        """
        self.model_path = model_path
        self.tools = tools or []
        self.memory_key = memory_key
        self.verbose = verbose
        
        # 自动检测设备
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self._init_model()
        self._init_memory()
        self._init_agent()
    
    def _init_model(self):
        """
        初始化本地模型
        """
        try:
            logger.info(f"开始初始化本地模型: {self.model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("分词器加载成功")
            
            # 尝试使用 Qwen2VLForConditionalGeneration 加载模型（针对 Qwen2-VL 模型）
            try:
                from transformers import Qwen2VLForConditionalGeneration
                logger.info("尝试使用 Qwen2VLForConditionalGeneration 加载模型")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    dtype=torch.bfloat16 if self.device == "cuda:0" else torch.float32,
                    device_map=self.device
                )
                logger.info("Qwen2-VL 模型加载成功")
            except Exception as e:
                logger.warning(f"加载 Qwen2-VL 模型失败，尝试使用 AutoModelForCausalLM: {str(e)}")
                # 尝试使用 AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    dtype=torch.bfloat16 if self.device == "cuda:0" else torch.float32,
                    device_map=self.device
                )
                logger.info("AutoModelForCausalLM 模型加载成功")
            
            # 创建 transformers pipeline
            self.transformers_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            # 创建 HuggingFacePipeline 用于 LangChain
            self.llm = HuggingFacePipeline(
                pipeline=self.transformers_pipeline
            )
            logger.info("本地模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化本地模型失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _init_memory(self):
        """
        初始化记忆模块
        """
        try:
            # 暂时使用简单的内存实现
            self.memory = []
            logger.info("记忆模块初始化完成")
        except Exception as e:
            logger.error(f"初始化记忆模块失败: {str(e)}")
            self.memory = []
    
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
                logger.info("Agent 初始化完成")
            else:
                logger.info("未配置工具，跳过 Agent 初始化")
                self.agent = None
        except Exception as e:
            logger.error(f"初始化 Agent 失败: {str(e)}")
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
            logger.info(f"调用本地 LLM，查询: {query}")
            
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
                logger.info("使用 Agent 调用 LLM")
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
            prompt = f"你是一个智能助手，用简洁的中文回答用户问题。\n\n用户问题: {query}\n\n回答:"
            response = self.llm.invoke(prompt)
            
            # 提取响应文本
            if isinstance(response, str):
                if "回答:" in response:
                    return response.split("回答:")[-1].strip()
                return response
            return str(response)
                
        except Exception as e:
            logger.error(f"调用本地 LLM 失败: {str(e)}")
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
    
    def clear_memory(self):
        """
        清除记忆
        """
        if self.memory:
            self.memory.clear()
            logger.info("记忆已清除")


class LocalLLMFactory:
    """
    本地 LLM 工厂类
    用于创建和管理不同的本地 LLM 实例
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, name: str, model_path: str, **kwargs):
        """
        获取或创建本地 LLM 实例
        
        Args:
            name (str): 实例名称
            model_path (str): 模型路径
            **kwargs: 其他参数
            
        Returns:
            LocalLLMAbstract: 本地 LLM 实例
        """
        if name not in cls._instances:
            cls._instances[name] = LocalLLMAbstract(model_path, **kwargs)
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
def create_example_tools():
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
        )
    ]
    
    return tools


# 创建全局实例
def get_default_llm(model_path: str = "./models/qwen_vl/qwen/Qwen2-VL-2B", **kwargs):
    """
    获取默认的本地 LLM 实例
    
    Args:
        model_path (str): 模型路径
        **kwargs: 其他参数
        
    Returns:
        LocalLLMAbstract: 本地 LLM 实例
    """
    return LocalLLMFactory.get_instance("default", model_path, **kwargs)


def call_local_llm(query: str, conversation_history: Optional[List[Dict]] = None, **kwargs):
    """
    外部调用接口
    
    Args:
        query (str): 用户查询
        conversation_history (Optional[List[Dict]]): 对话历史
        **kwargs: 其他参数
        
    Returns:
        str: 模型响应
    """
    llm = get_default_llm(**kwargs)
    return llm.invoke(query, conversation_history)
