#!/usr/bin/env python3
# coding=utf-8
"""
测试使用Edge TTS的ASR+LLM+TTS pipeline
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
        logging.FileHandler('test_edge_tts_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_edge_tts_pipeline():
    """
    测试使用Edge TTS的pipeline
    """
    logger.info("开始测试Edge TTS pipeline...")
    
    try:
        # 初始化pipeline，使用Edge TTS
        pipeline = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            edge_tts_voice="zh-CN-XiaoyiNeural"  # 设置声音
        )
        
        logger.info("pipeline初始化成功")
        
        # 测试完整流程
        logger.info("请开始说话，录音将持续5秒...")
        time.sleep(1)
        
        result = pipeline.run(duration=5)
        
        logger.info(f"测试结果: {result}")
        
        if result.get("success"):
            logger.info("Edge TTS pipeline测试成功!")
            logger.info(f"ASR识别结果: {result.get('asr_text')}")
            logger.info(f"LLM响应结果: {result.get('llm_response')}")
        else:
            logger.error(f"Edge TTS pipeline测试失败: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_edge_tts_pipeline()
