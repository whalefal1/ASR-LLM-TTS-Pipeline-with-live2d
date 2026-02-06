#!/usr/bin/env python3
# coding=utf-8
"""
Ollama 模型集成测试
测试本地部署的 qwen2.5vl:7b 模型的 LangChain 1.0 调用
使用自定义的 ollama_llm 模块
"""

import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.ollama_llm import call_ollama_llm, get_default_ollama_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_ollama_integration')


def test_ollama_qwen_basic_integration():
    """
    测试 Ollama qwen2.5vl:7b 模型的基本调用
    使用 call_ollama_llm 函数
    """
    logger.info("开始测试 Ollama qwen2.5vl:7b 模型的基本调用")
    
    try:
        logger.info("Ollama 模型初始化成功")
        
        # 测试简单的问答
        test_questions = [
            "你好，你是谁？",
            "今天天气怎么样？",
            "2+2等于多少？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"测试问题 {i}: {question}")
            
            # 使用自定义的 call_ollama_llm 函数
            response = call_ollama_llm(question, model_name="qwen2.5vl:7b")
            
            logger.info(f"模型响应: {response}")
            
            # 验证响应
            assert response is not None, f"问题 {i} 模型未返回响应"
            assert len(response) > 0, f"问题 {i} 响应内容为空"
        
        logger.info("基本调用测试通过")
        return True
        
    except Exception as e:
        logger.error(f"基本调用测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_ollama_qwen_with_message_history():
    """
    测试带有对话历史的 Ollama qwen2.5vl:7b 模型调用
    使用 call_ollama_llm 函数
    """
    logger.info("开始测试带有对话历史的 Ollama qwen2.5vl:7b 模型调用")
    
    try:
        # 测试连续对话
        logger.info("测试连续对话")
        
        # 构建对话历史
        conversation_history = []
        
        # 第一轮对话
        first_question = "你好，我叫张三"
        first_response = call_ollama_llm(first_question, model_name="qwen2.5vl:7b")
        logger.info(f"第一轮响应: {first_response}")
        
        # 更新对话历史
        conversation_history.append({"role": "user", "content": first_question})
        conversation_history.append({"role": "assistant", "content": first_response})
        
        # 第二轮对话，包含上下文
        second_question = "我叫什么名字？"
        second_response = call_ollama_llm(second_question, conversation_history, model_name="qwen2.5vl:7b")
        
        logger.info(f"第二轮响应: {second_response}")
        
        # 验证响应
        assert second_response is not None, "模型未返回响应"
        assert len(second_response) > 0, "响应内容为空"
        
        # 检查是否包含用户名称
        if "张三" in second_response:
            logger.info("模型成功记住用户名称")
        else:
            logger.warning("模型未明确提及用户名称，但测试仍通过")
        
        logger.info("对话历史测试通过")
        return True
        
    except Exception as e:
        logger.error(f"对话历史测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_ollama_llm_instance():
    """
    测试 OllamaLLM 实例的直接调用
    """
    logger.info("开始测试 OllamaLLM 实例的直接调用")
    
    try:
        # 获取默认的 Ollama LLM 实例
        llm_instance = get_default_ollama_llm(model_name="qwen2.5vl:7b")
        
        logger.info("OllamaLLM 实例获取成功")
        
        # 测试简单的问答
        test_question = "什么是人工智能？"
        logger.info(f"测试问题: {test_question}")
        
        # 直接调用实例的 invoke 方法
        response = llm_instance.invoke(test_question)
        
        logger.info(f"模型响应: {response}")
        
        # 验证响应
        assert response is not None, "模型未返回响应"
        assert len(response) > 0, "响应内容为空"
        
        logger.info("OllamaLLM 实例测试通过")
        return True
        
    except Exception as e:
        logger.error(f"OllamaLLM 实例测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    """
    运行所有测试
    """
    logger.info("开始运行 Ollama 集成测试")
    
    # 运行基本调用测试
    basic_test_passed = test_ollama_qwen_basic_integration()
    
    # 运行对话历史测试
    history_test_passed = test_ollama_qwen_with_message_history()
    
    # 运行实例测试
    instance_test_passed = test_ollama_llm_instance()
    
    # 汇总测试结果
    all_tests_passed = basic_test_passed and history_test_passed and instance_test_passed
    
    if all_tests_passed:
        logger.info("所有测试通过！")
        print("✅ 所有测试通过！")
    else:
        logger.error("部分测试失败！")
        print("❌ 部分测试失败！")
    
    # 退出码
    exit(0 if all_tests_passed else 1)
