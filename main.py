#!/usr/bin/env python3
# coding=utf-8
"""
ASR+LLM+TTS+Live2D集成主程序
实现完整的语音交互流程：语音识别 → LLM处理 → 语音合成 → Live2D对话框展示
"""

import logging
import sys
import os
import time
import pygame
import asyncio
import edge_tts

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.asr.asr_model import ASRModule
from src.llm.ollama_llm import call_ollama_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

class ASRLLMTTSLive2DPipeline:
    """
    ASR+LLM+TTS+Live2D级联系统
    实现完整的语音交互流程：语音识别 → LLM处理 → 语音合成 → Live2D对话框展示
    """
    
    # 类级别的ASR模型实例，确保全局只初始化一次
    _asr_instance = None
    _asr_initialized = False
    
    def __init__(self, 
                 asr_model_dir="./models/SenseVoice",
                 llm_model_name="qwen2.5vl:7b",
                 edge_tts_voice="zh-CN-XiaoyiNeural"):
        """
        初始化级联系统
        
        Args:
            asr_model_dir (str): SenseVoice模型目录
            llm_model_name (str): LLM模型名称
            edge_tts_voice (str): Edge TTS使用的声音
        """
        self.llm_model_name = llm_model_name
        self.edge_tts_voice = edge_tts_voice
        
        # 初始化音频播放
        logger.info("开始初始化音频播放...")
        try:
            # 强制初始化pygame
            pygame.init()
            logger.info("pygame初始化成功")
            
            # 初始化音频 mixer
            pygame.mixer.init()
            logger.info("pygame.mixer初始化成功")
            
            # 检查音频设备
            if pygame.mixer.get_init():
                logger.info(f"音频设备初始化成功，声道数: {pygame.mixer.get_num_channels()}")
                self.audio_playback_available = True
            else:
                logger.error("音频设备初始化失败")
                self.audio_playback_available = False
        except Exception as e:
            logger.error(f"初始化音频播放失败: {str(e)}")
            # 打印详细的错误信息
            import traceback
            logger.error(traceback.format_exc())
            self.audio_playback_available = False
        
        logger.info(f"音频播放初始化完成，可用状态: {self.audio_playback_available}")
        
        # 初始化对话历史
        self.conversation_history = []
        
        # 初始化组件
        self._init_asr(asr_model_dir)
        
        logger.info("ASR-LLM-TTS-Live2D级联系统初始化完成")
    
    def _init_asr(self, model_dir):
        """
        初始化ASR组件（全局只初始化一次）
        """
        try:
            if not ASRLLMTTSLive2DPipeline._asr_initialized:
                logger.info("正在初始化ASR组件...")
                logger.info(f"ASR模型目录: {model_dir}")
                ASRLLMTTSLive2DPipeline._asr_instance = ASRModule(use_sensevoice=True, model_dir=model_dir)
                ASRLLMTTSLive2DPipeline._asr_initialized = True
                logger.info("ASR组件初始化完成（全局首次初始化）")
            else:
                logger.info("ASR组件已初始化，使用现有实例")
            
            # 赋值给实例变量
            self.asr = ASRLLMTTSLive2DPipeline._asr_instance
            logger.info("ASR组件引用成功")
        except Exception as e:
            logger.error(f"初始化ASR组件失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.asr = None
    
    def speech_to_text(self, duration=5):
        """
        语音转文本（ASR）
        
        Args:
            duration (int): 录音时长（秒）
            
        Returns:
            str: 识别的文本
        """
        if not self.asr:
            logger.error("ASR组件未初始化，无法进行语音识别")
            return ""
        
        try:
            logger.info("开始语音识别...")
            start_time = time.time()
            
            # 录制并识别语音（使用固定长度录音）
            text = self.asr.record_and_recognize(duration=duration, use_vad=False)
            
            end_time = time.time()
            logger.info(f"语音识别完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"识别结果: {text}")
            
            return text
        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            return ""
    
    def text_to_response(self, text, conversation_history=None):
        """
        文本转响应（使用LLM模型）
        
        Args:
            text (str): 输入文本
            conversation_history (list): 对话历史
            
        Returns:
            str: 生成的响应文本
        """
        if not text:
            logger.error("输入文本为空，无法进行LLM模型处理")
            return ""
        
        try:
            logger.info("开始LLM模型处理...")
            start_time = time.time()
            
            # 使用LLM模型生成响应
            response = call_ollama_llm(text, conversation_history, model_name=self.llm_model_name)
            
            end_time = time.time()
            logger.info(f"LLM模型处理完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"LLM响应: {response}")
            logger.info(f"响应长度: {len(response)}字")
            
            return response
        except Exception as e:
            logger.error(f"LLM模型处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"
    
    def response_to_speech(self, response, output_file="output_response.wav"):
        """
        响应转语音（TTS）
        
        Args:
            response (str): LLM模型的响应
            output_file (str): 输出音频文件路径
            
        Returns:
            bool: 是否成功
        """
        if not response:
            logger.error("响应文本为空，无法进行语音合成")
            return False
        
        try:
            logger.info("开始语音合成...")
            start_time = time.time()
            
            # 使用Edge TTS
            logger.info(f"使用Edge TTS进行语音合成，声音: {self.edge_tts_voice}")
            
            # Edge TTS生成MP3，需要使用MP3格式的文件名
            mp3_output = output_file.replace('.wav', '.mp3')
            
            async def generate_speech():
                communicate = edge_tts.Communicate(response, self.edge_tts_voice)
                with open(mp3_output, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk['type'] == 'audio':
                            f.write(chunk['data'])
            
            # 运行异步函数
            asyncio.run(generate_speech())
            logger.info(f"Edge TTS语音合成完成，输出文件: {mp3_output}")
            
            # 播放音频
            if self.audio_playback_available:
                try:
                    logger.info(f"正在播放合成的语音文件: {mp3_output}")
                    
                    if os.path.exists(mp3_output):
                        # 直接使用MP3格式播放
                        pygame.mixer.music.load(mp3_output)
                        pygame.mixer.music.play()
                        
                        # 同步等待播放完成
                        max_wait_time = 15
                        wait_start_time = time.time()
                        while pygame.mixer.music.get_busy() and (time.time() - wait_start_time) < max_wait_time:
                            time.sleep(0.1)
                        
                        # 确保停止播放并释放资源
                        pygame.mixer.music.stop()
                        
                        # 尝试使用 unload() 方法释放音乐资源（pygame 2.0+）
                        try:
                            if hasattr(pygame.mixer.music, 'unload'):
                                pygame.mixer.music.unload()
                                logger.debug("音乐资源已显式卸载")
                            else:
                                # 对于旧版本的 pygame，尝试加载一个空的声音
                                pygame.mixer.music.load("empty")
                                logger.debug("尝试通过加载空声音释放资源")
                        except Exception as e:
                            logger.debug(f"释放资源时出现异常: {e}")
                            # 即使失败也继续
                            pass
                        
                        # 额外等待一小段时间确保资源释放
                        time.sleep(1)
                        logger.info("语音播放完成")
                        
                        # 播放完成后删除音频文件
                        try:
                            if os.path.exists(mp3_output):
                                # 使用重试机制确保文件删除成功
                                max_retries = 5
                                retry_delay = 0.5
                                
                                for retry in range(max_retries):
                                    try:
                                        # 尝试删除文件
                                        os.remove(mp3_output)
                                        logger.info(f"音频文件已删除: {mp3_output}")
                                        break
                                    except Exception as e:
                                        if retry < max_retries - 1:
                                            # 重试前等待
                                            logger.debug(f"删除失败，{retry_delay}秒后重试...")
                                            time.sleep(retry_delay)
                                            retry_delay *= 1.5
                                        else:
                                            # 达到最大重试次数
                                            logger.warning(f"删除音频文件失败（可能被其他进程占用）: {str(e)}")
                        except Exception as e:
                            logger.warning(f"删除音频文件失败（可能被其他进程占用）: {str(e)}")
                            # 即使删除失败也不影响整个流程
                except Exception as e:
                    logger.error(f"播放语音失败: {str(e)}")
                    # 确保停止播放
                    pygame.mixer.music.stop()
                    time.sleep(1)
            else:
                logger.warning("音频播放不可用，跳过播放步骤")
            
            end_time = time.time()
            logger.info(f"语音合成完成，耗时: {end_time - start_time:.2f}秒")
            return True
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def show_live2d_response(self, response):
        """
        在Live2D对话框中展示LLM回复（通过WebSocket）
        
        Args:
            response (str): LLM模型的响应
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info("开始更新Live2D对话框（通过WebSocket）...")
            
            # 导入WebSocket客户端
            from src.websocket.live2d_ws_server import send_llm_response
            
            # 使用异步方式发送消息
            try:
                # 尝试获取现有事件循环
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(send_llm_response(response, duration=8000))
            
            if success:
                logger.info("Live2D对话框已更新（通过WebSocket）")
            else:
                logger.warning("WebSocket发送失败，可能没有连接的客户端")
            
            return success
        except Exception as e:
            logger.error(f"更新Live2D对话框失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run(self, duration=5):
        """
        运行完整的ASR→LLM→Live2D流程
        
        Args:
            duration (int): 录音时长（秒）
            
        Returns:
            dict: 包含整个流程结果的字典
        """
        logger.info("开始运行ASR→LLM→Live2D流程...")
        start_time = time.time()
        
        result = {
            "success": False,
            "asr_text": "",
            "llm_response": "",
            "tts_success": False,
            "live2d_success": False,
            "error": ""
        }
        
        try:
            # 1. 语音转文本
            asr_text = self.speech_to_text(duration=duration)
            if not asr_text:
                result["error"] = "语音识别失败或未检测到有效语音"
                logger.error(result["error"])
                return result
            result["asr_text"] = asr_text
            
            # 2. 文本转响应（使用实例的对话历史）
            llm_response = self.text_to_response(asr_text, self.conversation_history)
            if not llm_response:
                result["error"] = "LLM模型处理失败"
                logger.error(result["error"])
                return result
            result["llm_response"] = llm_response
            
            # 3. 在Live2D对话框中展示响应（在Edge TTS播放之前）
            live2d_success = self.show_live2d_response(llm_response)
            result["live2d_success"] = live2d_success
            
            # 等待Live2D对话框更新完成
            logger.info("等待Live2D对话框更新完成...")
            time.sleep(1)
            
            # 4. 响应转语音（TTS）
            tts_success = self.response_to_speech(llm_response, output_file="test_live2d_tts.wav")
            result["tts_success"] = tts_success
            
            # 5. 更新对话历史
            self.conversation_history.append({"role": "user", "content": asr_text})
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # 限制对话历史长度（保留最近10轮）
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # 流程成功完成
            result["success"] = True
            
        except Exception as e:
            error_msg = f"流程执行失败: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
        
        end_time = time.time()
        logger.info(f"流程执行完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 打印结果摘要
        if result["success"]:
            logger.info("\n=== 流程执行结果 ===")
            logger.info(f"语音识别结果: {result['asr_text']}")
            logger.info(f"LLM响应: {result['llm_response']}")
            logger.info(f"Edge TTS合成成功: {result['tts_success']}")
            logger.info(f"Live2D对话框更新成功: {result['live2d_success']}")
            logger.info("===================")
        else:
            logger.error(f"流程执行失败: {result['error']}")
        
        return result

def main():
    """
    主函数：运行多轮固定时间对话测试
    """
    print("=== ASR-LLM-TTS-Live2D多轮对话系统 ===")
    print("测试流程：")
    print("1. 启动WebSocket服务器")
    print("2. 打开Live2D页面")
    print("3. 语音识别（5秒录音）")
    print("4. 本地LLM处理")
    print("5. Edge TTS合成并播放")
    print("6. Live2D对话框展示回复（通过WebSocket）")
    print()
    print("🔊 提示：请在录音时说出您的问题，或说'退出'结束对话")
    print()
    
    # 启动WebSocket服务器
    print("正在启动WebSocket服务器...")
    from src.websocket.live2d_ws_server import get_ws_server
    import threading
    
    # 在后台线程中启动WebSocket服务器
    def start_ws_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = loop.run_until_complete(get_ws_server())
        loop.run_forever()
    
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()
    print("✅ WebSocket服务器已启动（ws://localhost:8765）")
    time.sleep(1)
    
    # 打开Live2D页面（只打开一次）
    print("正在打开Live2D页面...")
    import webbrowser
    webbrowser.open("http://localhost:8000/demo/demo.html")
    print("✅ Live2D页面已打开")
    print()
    print("等待WebSocket连接建立...")
    time.sleep(3)
    
    # 创建测试实例
    pipeline = ASRLLMTTSLive2DPipeline(
        asr_model_dir="./models/SenseVoice",
        llm_model_name="qwen2.5vl:7b",
        edge_tts_voice="zh-CN-XiaoyiNeural"
    )
    
    # 多轮对话循环
    round_count = 1
    while True:
        print(f"\n=== 第 {round_count} 轮对话 ===")
        print("请准备好，3秒后开始录音...")
        # 3秒倒计时
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("开始录音！请说出您的问题...")
        
        # 运行一轮测试
        result = pipeline.run(duration=5)
        
        # 打印本轮结果
        if result["success"]:
            print("\n✅ 本轮对话完成！")
            print(f"语音识别结果: {result['asr_text']}")
            print(f"LLM回复: {result['llm_response']}")
            print(f"Edge TTS合成: {'成功' if result['tts_success'] else '失败'}")
            print(f"Live2D对话框更新: {'成功' if result['live2d_success'] else '失败'}")
        else:
            print(f"\n❌ 本轮对话失败: {result.get('error', '未知错误')}")
        
        # 检查是否需要退出
        asr_text = result["asr_text"].strip()
        if asr_text.lower() == "退出" or "退出" in asr_text:
            print("\n👋 检测到'退出'指令，正在结束对话...")
            exit_response = "对话已结束，再见！"
            pipeline.response_to_speech(exit_response, output_file="exit_prompt.wav")
            pipeline.show_live2d_response(exit_response)
            time.sleep(2)
            break
        
        # 等待一段时间再进行下一轮
        print("\n等待2秒后开始下一轮对话...")
        time.sleep(2)
        round_count += 1

if __name__ == "__main__":
    main()
