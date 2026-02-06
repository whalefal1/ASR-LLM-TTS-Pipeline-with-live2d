#!/usr/bin/env python3
# coding=utf-8
"""
固定时间多次对话测试
测试在固定时间内进行多次对话，验证ASR-LLM-TTS管道的稳定性和可靠性
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
        logging.FileHandler('test_multi_dialogue.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_multi_dialogue(duration=30, max_retries=3):
    """
    固定时间多次对话测试
    
    Args:
        duration (int): 测试总时长（秒）
        max_retries (int): 单次对话失败后的最大重试次数
    """
    logger.info(f"开始固定时间多次对话测试，测试时长: {duration}秒")
    
    try:
        # 初始化pipeline
        pipeline = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            llm_type="langchain",
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        logger.info("Pipeline初始化成功")
        
        # 测试统计
        total_dialogues = 0
        successful_dialogues = 0
        failed_dialogues = 0
        error_messages = []
        
        # 测试开始时间
        start_time = time.time()
        end_time = start_time + duration
        
        logger.info(f"测试开始，将在 {duration} 秒后结束")
        logger.info("请在每次提示后开始说话")
        
        # 主测试循环
        while time.time() < end_time:
            remaining_time = end_time - time.time()
            if remaining_time < 5:  # 确保有足够时间完成一次对话
                logger.info(f"剩余时间不足5秒，测试即将结束")
                break
            
            total_dialogues += 1
            dialogue_start = time.time()
            
            logger.info(f"\n=== 对话轮次 {total_dialogues} ===")
            logger.info(f"剩余测试时间: {remaining_time:.1f}秒")
            logger.info("请开始说话...")
            
            # 尝试进行对话
            dialogue_success = False
            
            for retry in range(max_retries):
                try:
                    # 录音5秒
                    result = pipeline.run(duration=5)
                    
                    if result.get("success"):
                        asr_text = result.get("asr_text", "")
                        llm_response = result.get("llm_response", "")
                        
                        logger.info(f"ASR识别结果: {asr_text}")
                        logger.info(f"LLM响应结果: {llm_response}")
                        logger.info("对话成功!")
                        
                        successful_dialogues += 1
                        dialogue_success = True
                        break
                    else:
                        error = result.get("error", "未知错误")
                        logger.warning(f"对话失败: {error}")
                        if retry < max_retries - 1:
                            logger.info(f"{retry+1}秒后重试...")
                            time.sleep(1)
                except Exception as e:
                    logger.error(f"对话过程中发生异常: {str(e)}")
                    if retry < max_retries - 1:
                        logger.info(f"{retry+1}秒后重试...")
                        time.sleep(1)
            
            if not dialogue_success:
                failed_dialogues += 1
                error_messages.append(f"对话轮次 {total_dialogues} 失败")
            
            # 对话间隔，避免连续录音
            dialogue_interval = 2
            logger.info(f"{dialogue_interval}秒后开始下一轮对话...")
            time.sleep(dialogue_interval)
        
        # 测试结束
        test_end = time.time()
        actual_duration = test_end - start_time
        
        # 计算成功率
        success_rate = (successful_dialogues / total_dialogues * 100) if total_dialogues > 0 else 0
        
        # 打印测试结果
        logger.info("\n=== 测试结果汇总 ===")
        logger.info(f"测试时长: {actual_duration:.1f}秒 (计划: {duration}秒)")
        logger.info(f"总对话轮次: {total_dialogues}")
        logger.info(f"成功对话: {successful_dialogues}")
        logger.info(f"失败对话: {failed_dialogues}")
        logger.info(f"成功率: {success_rate:.2f}%")
        
        if error_messages:
            logger.info("\n=== 错误信息 ===")
            for msg in error_messages:
                logger.info(msg)
        
        logger.info("\n=== 测试完成 ===")
        
        return {
            "total_dialogues": total_dialogues,
            "successful_dialogues": successful_dialogues,
            "failed_dialogues": failed_dialogues,
            "success_rate": success_rate,
            "actual_duration": actual_duration,
            "error_messages": error_messages
        }
        
    except Exception as e:
        logger.error(f"测试过程中发生严重错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "total_dialogues": 0,
            "successful_dialogues": 0,
            "failed_dialogues": 0,
            "success_rate": 0,
            "actual_duration": 0,
            "error_messages": [f"测试初始化失败: {str(e)}"]
        }

if __name__ == "__main__":
    # 运行测试，测试时长60秒
    test_result = test_multi_dialogue(duration=60)
    
    # 打印最终结果
    print("\n===== 固定时间多次对话测试结果 =====")
    print(f"测试时长: {test_result['actual_duration']:.1f}秒")
    print(f"总对话轮次: {test_result['total_dialogues']}")
    print(f"成功对话: {test_result['successful_dialogues']}")
    print(f"失败对话: {test_result['failed_dialogues']}")
    print(f"成功率: {test_result['success_rate']:.2f}%")
    if test_result['error_messages']:
        print("\n错误信息:")
        for msg in test_result['error_messages']:
            print(f"- {msg}")
    print("====================================")
