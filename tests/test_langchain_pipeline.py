#!/usr/bin/env python3
# coding=utf-8
"""
测试使用LangChain调用本地大模型的ASR+LLM+TTS pipeline
"""

import time
import logging
from src.pipeline.asr_llm_tts_pipeline import ASRLLMTTSPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_langchain_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_langchain_pipeline():
    """
    测试使用LangChain调用本地大模型的pipeline
    """
    logger.info("开始测试LangChain pipeline...")
    
    try:
        # 初始化pipeline，使用LangChain
        pipeline = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            llm_type="langchain",  # 使用LangChain
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        logger.info("LangChain pipeline初始化成功")
        
        # 测试完整流程
        logger.info("请开始说话，录音将持续5秒...")
        time.sleep(1)
        
        result = pipeline.run(duration=5)
        
        logger.info(f"测试结果: {result}")
        
        if result.get("success"):
            logger.info("LangChain pipeline测试成功!")
            logger.info(f"ASR识别结果: {result.get('asr_text')}")
            logger.info(f"LLM响应结果: {result.get('llm_response')}")
        else:
            logger.error(f"LangChain pipeline测试失败: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def test_direct_vs_langchain():
    """
    对比测试：直接调用vs LangChain调用
    """
    logger.info("开始对比测试: 直接调用 vs LangChain调用")
    
    # 测试直接调用
    logger.info("\n=== 测试1: 直接调用大模型 ===")
    try:
        pipeline_direct = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            llm_type="direct",  # 直接调用
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        logger.info("请说'你好，今天天气怎么样？'，录音将持续5秒...")
        time.sleep(1)
        result_direct = pipeline_direct.run(duration=5)
        logger.info(f"直接调用结果: {result_direct.get('llm_response')}")
    except Exception as e:
        logger.error(f"直接调用测试失败: {str(e)}")
    
    # 测试LangChain调用
    logger.info("\n=== 测试2: LangChain调用大模型 ===")
    try:
        pipeline_langchain = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            llm_type="langchain",  # LangChain调用
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        logger.info("请再次说'你好，今天天气怎么样？'，录音将持续5秒...")
        time.sleep(1)
        result_langchain = pipeline_langchain.run(duration=5)
        logger.info(f"LangChain调用结果: {result_langchain.get('llm_response')}")
    except Exception as e:
        logger.error(f"LangChain调用测试失败: {str(e)}")

if __name__ == "__main__":
    # 运行基本测试
    test_langchain_pipeline()
    
    # 运行对比测试
    # test_direct_vs_langchain()
