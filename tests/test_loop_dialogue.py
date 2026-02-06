#!/usr/bin/env python3
# coding=utf-8
"""
循环多轮对话测试
验证系统的多轮对话能力和记忆功能
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
        logging.FileHandler('test_loop_dialogue.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_loop_dialogue(rounds=5, duration_per_round=5):
    """
    循环多轮对话测试
    
    Args:
        rounds (int): 对话轮数
        duration_per_round (int): 每轮录音时长（秒）
    """
    logger.info(f"开始循环多轮对话测试，共 {rounds} 轮对话")
    
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
        total_rounds = 0
        successful_rounds = 0
        failed_rounds = 0
        conversation_history = []
        
        # 对话轮次
        for round_num in range(1, rounds + 1):
            total_rounds += 1
            round_start = time.time()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"第 {round_num}/{rounds} 轮对话")
            logger.info(f"{'='*50}")
            logger.info("请开始说话...")
            
            # 尝试进行对话
            try:
                # 录音并处理
                result = pipeline.run(duration=duration_per_round)
                
                if result.get("success"):
                    asr_text = result.get("asr_text", "")
                    llm_response = result.get("llm_response", "")
                    
                    logger.info(f"ASR识别结果: {asr_text}")
                    logger.info(f"LLM响应结果: {llm_response}")
                    logger.info(f"对话历史长度: {len(conversation_history)}")
                    
                    # 更新对话历史
                    conversation_history.append({"role": "user", "content": asr_text})
                    conversation_history.append({"role": "assistant", "content": llm_response})
                    
                    # 限制对话历史长度（保留最近10轮）
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]
                        logger.info("对话历史已截断，保留最近10轮")
                    
                    successful_rounds += 1
                    logger.info(f"第 {round_num} 轮对话成功")
                else:
                    error = result.get("error", "未知错误")
                    logger.error(f"第 {round_num} 轮对话失败: {error}")
                    failed_rounds += 1
                
                # 对话间隔
                if round_num < rounds:
                    interval = 3
                    logger.info(f"{interval}秒后开始下一轮对话...")
                    time.sleep(interval)
                
            except Exception as e:
                logger.error(f"第 {round_num} 轮对话过程中发生异常: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                failed_rounds += 1
            
            round_end = time.time()
            round_duration = round_end - round_start
            logger.info(f"第 {round_num} 轮对话耗时: {round_duration:.1f}秒")
        
        # 测试结束
        test_end = time.time()
        total_duration = test_end - time.time() if 'test_end' in locals() else 0
        
        # 计算成功率
        success_rate = (successful_rounds / total_rounds * 100) if total_rounds > 0 else 0
        
        # 打印测试结果
        logger.info(f"\n{'='*50}")
        logger.info("循环多轮对话测试结果汇总")
        logger.info(f"{'='*50}")
        logger.info(f"总对话轮数: {total_rounds}")
        logger.info(f"成功对话: {successful_rounds}")
        logger.info(f"失败对话: {failed_rounds}")
        logger.info(f"成功率: {success_rate:.2f}%")
        logger.info(f"最终对话历史长度: {len(conversation_history)}")
        
        # 打印对话历史摘要
        logger.info(f"\n{'='*50}")
        logger.info("对话历史摘要")
        logger.info(f"{'='*50}")
        for i, msg in enumerate(conversation_history):
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            logger.info(f"{i+1}. [{role}] {content}")
        
        logger.info(f"\n{'='*50}")
        logger.info("测试完成")
        logger.info(f"{'='*50}")
        
        return {
            "total_rounds": total_rounds,
            "successful_rounds": successful_rounds,
            "failed_rounds": failed_rounds,
            "success_rate": success_rate,
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        logger.error(f"测试过程中发生严重错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "total_rounds": 0,
            "successful_rounds": 0,
            "failed_rounds": 0,
            "success_rate": 0,
            "conversation_history": []
        }

def test_memory_continuity():
    """
    测试记忆连续性
    验证模型是否能够记住之前的对话内容
    """
    logger.info("\n开始记忆连续性测试")
    logger.info(f"{'='*50}")
    
    try:
        # 初始化pipeline
        pipeline = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            llm_type="langchain",
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        logger.info("Pipeline初始化成功")
        
        # 测试问题序列
        test_questions = [
            "我叫什么名字？",
            "我的爱好是什么？",
            "记得我们刚才聊了什么吗？"
        ]
        
        logger.info("请按照以下顺序进行对话，测试记忆连续性：")
        for i, question in enumerate(test_questions, 1):
            logger.info(f"{i}. {question}")
        
        logger.info("\n开始测试...")
        
        # 进行多轮对话
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"测试问题 {i}: {question}")
            logger.info(f"{'='*50}")
            logger.info("请说出这个问题...")
            
            try:
                # 录音并处理
                result = pipeline.run(duration=5)
                
                if result.get("success"):
                    asr_text = result.get("asr_text", "")
                    llm_response = result.get("llm_response", "")
                    
                    logger.info(f"ASR识别结果: {asr_text}")
                    logger.info(f"LLM响应结果: {llm_response}")
                    
                    # 检查是否记得之前的对话
                    if i > 1:
                        if "不记得" in llm_response or "不知道" in llm_response:
                            logger.warning("⚠️  模型可能没有记住之前的对话内容")
                        else:
                            logger.info("✓ 模型似乎记得之前的对话内容")
                else:
                    error = result.get("error", "未知错误")
                    logger.error(f"对话失败: {error}")
                
                # 对话间隔
                if i < len(test_questions):
                    logger.info("3秒后进行下一个问题...")
                    time.sleep(3)
                
            except Exception as e:
                logger.error(f"对话过程中发生异常: {str(e)}")
        
        logger.info(f"\n{'='*50}")
        logger.info("记忆连续性测试完成")
        logger.info(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"测试过程中发生严重错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行循环多轮对话测试
    print("\n===== 循环多轮对话测试 =====")
    test_result = test_loop_dialogue(rounds=5, duration_per_round=5)
    
    # 打印最终结果
    print("\n===== 测试结果 =====")
    print(f"总对话轮数: {test_result['total_rounds']}")
    print(f"成功对话: {test_result['successful_rounds']}")
    print(f"失败对话: {test_result['failed_rounds']}")
    print(f"成功率: {test_result['success_rate']:.2f}%")
    print(f"对话历史长度: {len(test_result['conversation_history'])}")
    print("====================")
    
    # 运行记忆连续性测试
    print("\n===== 记忆连续性测试 =====")
    test_memory_continuity()
    print("\n====================")
