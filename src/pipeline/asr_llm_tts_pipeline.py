# coding=utf-8
# ASR+LLM+TTS级联系统

import torch
import soundfile as sf
import time
import logging
import pygame
import os
import pyaudio
import webrtcvad
import numpy as np
import asyncio
import edge_tts
from src.asr.asr_model import ASRModule
from src.llm.local_llm_abstract import get_default_llm, create_example_tools
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ASR-LLM-TTS Pipeline')


class ASRLLMTTSPipeline:
    """
    ASR+LLM+TTS级联系统
    实现完整的语音交互流程：语音识别 → 大模型处理 → 语音合成
    """
    
    def __init__(self, 
                 asr_model_dir="./SenseVoice",
                 llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
                 llm_type="direct",  # "direct" 或 "langchain"
                 edge_tts_voice="zh-CN-XiaoyiNeural"):
        """
        初始化级联系统
        
        Args:
            asr_model_dir (str): SenseVoice模型目录
            llm_model_path (str): 本地LLM模型路径
            llm_type (str): LLM调用类型，"direct" 或 "langchain"
            edge_tts_voice (str): Edge TTS使用的声音
        """
        self.llm_type = llm_type
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
        self._init_llm(llm_model_path)
        
        # 初始化LangChain（如果需要）
        if llm_type == "langchain":
            self._init_langchain()
        
        # 初始化Edge TTS组件
        self._init_edge_tts()
        
        logger.info("ASR-LLM-TTS级联系统初始化完成")
    
    def _init_asr(self, model_dir):
        """
        初始化ASR组件
        """
        try:
            logger.info("正在初始化ASR组件...")
            self.asr = ASRModule(use_sensevoice=True, model_dir=model_dir)
            logger.info("ASR组件初始化完成")
        except Exception as e:
            logger.error(f"初始化ASR组件失败: {str(e)}")
            self.asr = None
    
    def _init_llm(self, model_path):
        """
        初始化本地LLM组件
        """
        try:
            logger.info("正在初始化本地LLM组件...")
            logger.info(f"LLM模型路径: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"LLM模型路径不存在: {model_path}")
                self.llm_model = None
                self.llm_tokenizer = None
                self.llm_device = None
                return
            
            # 获取可用设备
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU 可用，内存: {gpu_mem:.2f}GB")
                    self.llm_device = "cuda:0"
                except Exception as e:
                    logger.warning(f"GPU 检查失败: {e}，使用 CPU")
                    self.llm_device = "cpu"
            else:
                logger.info("GPU 不可用，使用 CPU")
                self.llm_device = "cpu"
            
            # 加载模型
            start_time = time.time()
            
            if self.llm_type == "langchain":
                # 使用新的本地LLM抽象层
                logger.info("使用本地LLM抽象层初始化LangChain...")
                
                # 创建示例工具
                tools = create_example_tools()
                logger.info(f"创建了 {len(tools)} 个示例工具")
                
                # 初始化本地LLM抽象层
                self.llm_model = get_default_llm(
                    model_path=model_path,
                    device=self.llm_device,
                    tools=tools,
                    verbose=True
                )
                
                logger.info("LangChain初始化完成，支持tools和memory功能")
            else:
                # 直接使用原始模型
                logger.info("使用CPU模式加载模型...")
                base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=self.llm_device,
                    trust_remote_code=True
                )
                logger.info("LLM模型加载成功 (CPU模式)")
                
                # 加载分词器
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                logger.info("LLM分词器加载成功")
                
                self.llm_model = base_model
            
            end_time = time.time()
            logger.info(f"LLM组件初始化完成，耗时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"初始化LLM组件失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.llm_model = None
            self.llm_tokenizer = None
            self.llm_device = None
    

    
    def _init_edge_tts(self):
        """
        初始化Edge TTS组件
        """
        try:
            logger.info("正在初始化Edge TTS组件...")
            logger.info(f"Edge TTS声音设置: {self.edge_tts_voice}")
            logger.info("Edge TTS组件初始化完成")
        except Exception as e:
            logger.error(f"初始化Edge TTS组件失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _init_langchain(self):
        """
        初始化LangChain组件
        """
        try:
            logger.info("正在初始化LangChain组件...")
            
            # 简化LangChain集成，直接使用HuggingFacePipeline
            logger.info("LangChain组件初始化完成")
        except Exception as e:
            logger.error(f"初始化LangChain组件失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        文本转响应（使用本地LLM）
        
        Args:
            text (str): 输入文本
            conversation_history (list): 对话历史
            
        Returns:
            str: 生成的响应文本
        """
        if not text:
            logger.error("输入文本为空，无法进行大模型处理")
            return ""
        
        try:
            logger.info("开始大模型处理...")
            start_time = time.time()
            
            if self.llm_type == "langchain":
                # 使用LangChain调用大模型（通过本地LLM抽象层）
                logger.info("使用LangChain调用大模型...")
                
                # 使用新的invoke方法，支持conversation_history
                response = self.llm_model.invoke(text, conversation_history)
                
                logger.info("LangChain调用完成")
            else:
                # 直接调用大模型
                if not self.llm_model or not self.llm_tokenizer:
                    logger.error("LLM组件未初始化，无法进行大模型处理")
                    return "抱歉，大模型未初始化，请稍后再试。"
                
                # 添加响应长度限制
                limited_text = f"{text}\n\n请用不超过100字的简短回答来回应这个问题。"
                
                # 转换对话历史格式
                history = []
                if conversation_history:
                    for msg in conversation_history:
                        if msg["role"] == "user" and len(history) > 0 and history[-1][0] is None:
                            history[-1] = (msg["content"], history[-1][1])
                        elif msg["role"] == "assistant":
                            if len(history) > 0 and history[-1][1] is None:
                                history[-1] = (history[-1][0], msg["content"])
                            else:
                                history.append((None, msg["content"]))
                        elif msg["role"] == "user":
                            history.append((msg["content"], None))
                
                # 调用本地大模型
                if hasattr(self.llm_model, "chat"):
                    response = self.llm_model.chat(
                        self.llm_tokenizer,
                        limited_text,
                        history=history,
                        max_new_tokens=100
                    )
                else:
                    logger.info("使用generate方法生成响应...")
                    inputs = self.llm_tokenizer(limited_text, return_tensors="pt").to(self.llm_device)
                    with torch.no_grad():
                        outputs = self.llm_model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.llm_tokenizer.eos_token_id
                        )
                    response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 二次检查响应长度
            if len(response) > 100:
                response = response[:97] + "..."
                logger.info("大模型响应过长，已进行截断")
            
            end_time = time.time()
            logger.info(f"大模型处理完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"大模型响应: {response}")
            logger.info(f"响应长度: {len(response)}字")
            
            return response
        except Exception as e:
            logger.error(f"大模型处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"
    
    def response_to_speech(self, response, output_file="output_response.wav"):
        """
        响应转语音（TTS）
        
        Args:
            response (str): 大模型的响应
            output_file (str): 输出音频文件路径
            
        Returns:
            tuple: (None, None) 因为使用Edge TTS
        """
        if not response:
            logger.error("响应文本为空，无法进行语音合成")
            return None, None
        
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
                        time.sleep(0.5)
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
                                        break  # 删除成功，退出循环
                                    except Exception as e:
                                        if retry < max_retries - 1:
                                            # 重试前等待
                                            logger.debug(f"删除失败，{retry_delay}秒后重试...")
                                            time.sleep(retry_delay)
                                            retry_delay *= 1.5  # 指数退避
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
                    time.sleep(0.5)
            
            end_time = time.time()
            logger.info(f"语音合成完成，耗时: {end_time - start_time:.2f}秒")
            return None, None
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def run(self, duration=5, output_file="output_response.wav"):
        """
        运行完整的ASR→LLM→TTS流程（支持多轮对话）
        
        Args:
            duration (int): 录音时长（秒）
            output_file (str): 输出音频文件路径
            
        Returns:
            dict: 包含整个流程结果的字典
        """
        logger.info("开始运行ASR→LLM→TTS流程...")
        start_time = time.time()
        
        result = {
            "success": False,
            "asr_text": "",
            "llm_response": "",
            "tts_output": None,
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
                result["error"] = "大模型处理失败"
                logger.error(result["error"])
                return result
            result["llm_response"] = llm_response
            
            # 3. 响应转语音
            wavs, sr = self.response_to_speech(llm_response, output_file)
            
            # 对于Edge TTS，wavs和sr为None但实际上是成功的
            if wavs is None or sr is None:
                # Edge TTS返回None, None是正常的
                logger.info("Edge TTS语音合成成功")
            
            # 对于Edge TTS，使用MP3格式的文件名
            result["tts_output"] = output_file.replace('.wav', '.mp3')
            
            # 4. 更新对话历史
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
            logger.info(f"大模型响应: {result['llm_response']}")
            logger.info(f"语音合成输出: {result['tts_output']}")
            logger.info("====================")
        else:
            logger.error(f"流程执行失败: {result['error']}")
        
        return result
    
    def process_text(self, text, output_file="output_response.wav", conversation_history=None):
        """
        直接处理文本输入，跳过ASR步骤
        
        Args:
            text (str): 输入文本
            output_file (str): 输出音频文件路径
            conversation_history (list): 对话历史
            
        Returns:
            dict: 包含处理结果的字典
        """
        logger.info(f"开始处理文本输入: {text}")
        start_time = time.time()
        
        result = {
            "success": False,
            "input_text": text,
            "llm_response": "",
            "tts_output": None,
            "error": ""
        }
        
        try:
            # 1. 文本转响应
            llm_response = self.text_to_response(text, conversation_history)
            if not llm_response:
                result["error"] = "大模型处理失败"
                logger.error(result["error"])
                return result
            result["llm_response"] = llm_response
            
            # 2. 响应转语音
            wavs, sr = self.response_to_speech(llm_response, output_file)
            if wavs is None or sr is None:
                result["error"] = "语音合成失败"
                logger.error(result["error"])
                return result
            result["tts_output"] = output_file
            
            # 处理成功完成
            result["success"] = True
            
        except Exception as e:
            error_msg = f"处理执行失败: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
        
        end_time = time.time()
        logger.info(f"处理执行完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 打印结果摘要
        if result["success"]:
            logger.info("\n=== 处理执行结果 ===")
            logger.info(f"输入文本: {result['input_text']}")
            logger.info(f"大模型响应: {result['llm_response']}")
            logger.info(f"语音合成输出: {result['tts_output']}")
            logger.info("====================")
        else:
            logger.error(f"处理执行失败: {result['error']}")
        
        return result


def main():
    """
    主函数，用于测试级联系统（支持多轮对话）
    """
    print("\n=== ASR+LLM+TTS级联系统测试（使用本地大模型）===")
    print("支持多轮对话，按Ctrl+C退出\n")
    
    try:
        # 初始化级联系统
        pipeline = ASRLLMTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            llm_model_path="./models/qwen_vl/qwen/Qwen2-VL-2B",
            edge_tts_voice="zh-CN-XiaoyiNeural"
        )
        
        round_count = 1
        while True:
            print(f"\n=== 对话轮次 {round_count} ===")
            print("请说话，系统将识别您的语音并生成响应...")
            print("（提示：录音将持续5秒，请在这段时间内完成说话）")
            
            # 运行完整流程
            result = pipeline.run(
                duration=5,
                output_file=f"asr_llm_tts_output_round{round_count}.wav"
            )
            
            if result["success"]:
                print("\n✅ 流程执行成功！")
                print(f"语音识别结果: {result['asr_text']}")
                print(f"大模型响应: {result['llm_response']}")
            else:
                print(f"\n❌ 流程执行失败: {result['error']}")
            
            round_count += 1
            print("\n--- 准备下一轮对话 ---\n")
            # 短暂暂停，让用户准备
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🔄 用户终止对话")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
    finally:
        print("\n测试完成！")


if __name__ == "__main__":
    main()
