# coding=utf-8
# ASR+LLM+edge-TTS级联系统测试

from src.asr.asr_model import ASRModule
from src.llm.llmModel import call_llm
import edge_tts
import asyncio
import os
import time
import pygame

class ASRLLMEdgeTTSPipeline:
    """
    ASR+LLM+edge-TTS级联系统
    实现完整的语音交互流程：语音识别 → 大模型处理 → 语音合成
    """
    
    def __init__(self, 
                 asr_model_dir="./models/SenseVoice",
                 edge_tts_voice="zh-CN-XiaoyiNeural",  # 使用女声
                 edge_tts_rate="+0%",
                 edge_tts_volume="+0%"):
        """
        初始化级联系统
        
        Args:
            asr_model_dir (str): SenseVoice模型目录
            edge_tts_voice (str): edge-tts使用的声音
            edge_tts_rate (str): edge-tts语速调整
            edge_tts_volume (str): edge-tts音量调整
        """
        self.edge_tts_voice = edge_tts_voice
        self.edge_tts_rate = edge_tts_rate
        self.edge_tts_volume = edge_tts_volume
        self.conversation_history = []
        
        # 初始化音频播放
        try:
            pygame.mixer.init()
            self.audio_playback_available = True
        except Exception as e:
            print(f"初始化音频播放失败: {str(e)}")
            self.audio_playback_available = False
        
        # 初始化ASR组件
        self._init_asr(asr_model_dir)
        
        print("ASR+LLM+edge-TTS级联系统初始化完成")
    
    def _init_asr(self, model_dir):
        """
        初始化ASR组件
        """
        try:
            print("正在加载SenseVoice模型...")
            self.asr = ASRModule(model_dir=model_dir)
            print("SenseVoice模型加载完成")
        except Exception as e:
            print(f"初始化ASR组件失败: {str(e)}")
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
            print("ASR组件未初始化，无法进行语音识别")
            return ""
        
        try:
            print("开始语音识别...")
            start_time = time.time()
            
            # 录制并识别语音（使用固定长度录音）
            text = self.asr.record_and_recognize(duration=duration, use_vad=False)
            
            end_time = time.time()
            print(f"语音识别完成，耗时: {end_time - start_time:.2f}秒")
            print(f"识别结果: {text}")
            
            return text
        except Exception as e:
            print(f"语音识别失败: {str(e)}")
            return ""
    
    def text_to_response(self, text, conversation_history=None):
        """
        文本转响应（LLM）
        
        Args:
            text (str): 输入文本
            conversation_history (list): 对话历史
            
        Returns:
            str: 生成的响应文本
        """
        if not text:
            print("输入文本为空，无法进行大模型处理")
            return ""
        
        try:
            print("开始大模型处理...")
            start_time = time.time()
            
            # 添加响应长度限制
            limited_text = f"{text}\n\n请用不超过100字的简短回答来回应这个问题。"
            
            # 调用大模型
            response = call_llm(limited_text, conversation_history)
            
            # 二次检查响应长度
            if len(response) > 100:
                response = response[:97] + "..."
                print("大模型响应过长，已进行截断")
            
            end_time = time.time()
            print(f"大模型处理完成，耗时: {end_time - start_time:.2f}秒")
            print(f"大模型响应: {response}")
            
            return response
        except Exception as e:
            print(f"大模型处理失败: {str(e)}")
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"
    
    async def response_to_speech(self, response, output_file="output_response.mp3"):
        """
        响应转语音（edge-TTS）
        
        Args:
            response (str): 大模型的响应
            output_file (str): 输出音频文件路径
            
        Returns:
            bool: 是否成功
        """
        if not response:
            print("响应文本为空，无法进行语音合成")
            return False
        
        try:
            print("开始语音合成（edge-TTS）...")
            start_time = time.time()
            
            # 使用edge-tts生成语音
            communicate = edge_tts.Communicate(
                response, 
                self.edge_tts_voice,
                rate=self.edge_tts_rate,
                volume=self.edge_tts_volume
            )
            
            # 写入音频文件
            with open(output_file, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
            
            end_time = time.time()
            print(f"语音合成完成，耗时: {end_time - start_time:.2f}秒")
            print(f"输出文件: {output_file}")
            print(f"文件大小: {os.path.getsize(output_file)} 字节")
            
            # 自动播放合成的语音
            if self.audio_playback_available and os.path.exists(output_file):
                try:
                    print("正在播放合成的语音...")
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    
                    # 等待播放完成
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    print("语音播放完成")
                    
                    # 播放完成后删除音频文件
                    if os.path.exists(output_file):
                        os.remove(output_file)
                        print(f"音频文件已删除: {output_file}")
                except Exception as e:
                    print(f"播放语音失败: {str(e)}")
            
            return True
        except Exception as e:
            print(f"语音合成失败: {str(e)}")
            return False
    
    async def run(self, duration=5):
        """
        运行完整的ASR→LLM→edge-TTS流程
        
        Args:
            duration (int): 录音时长（秒）
            
        Returns:
            dict: 包含整个流程结果的字典
        """
        print("\n开始运行ASR→LLM→edge-TTS流程...")
        start_time = time.time()
        
        result = {
            "success": False,
            "asr_text": "",
            "llm_response": "",
            "error": ""
        }
        
        try:
            # 1. 语音转文本
            asr_text = self.speech_to_text(duration=duration)
            if not asr_text:
                result["error"] = "语音识别失败或未检测到有效语音"
                print(result["error"])
                return result
            result["asr_text"] = asr_text
            
            # 2. 文本转响应
            llm_response = self.text_to_response(asr_text, self.conversation_history)
            if not llm_response:
                result["error"] = "大模型处理失败"
                print(result["error"])
                return result
            result["llm_response"] = llm_response
            
            # 3. 响应转语音
            output_file = f"edge_tts_output_{int(time.time())}.mp3"
            tts_success = await self.response_to_speech(llm_response, output_file)
            if not tts_success:
                result["error"] = "语音合成失败"
                print(result["error"])
                return result
            
            # 4. 更新对话历史
            self.conversation_history.append({"role": "user", "content": asr_text})
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # 限制对话历史长度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            result["success"] = True
            print("ASR→LLM→edge-TTS流程执行成功")
            
        except Exception as e:
            error_msg = f"流程执行失败: {str(e)}"
            print(error_msg)
            result["error"] = error_msg
        
        end_time = time.time()
        print(f"总耗时: {end_time - start_time:.2f}秒")
        
        return result

async def main():
    """
    主函数，用于测试级联系统
    """
    print("=== ASR+LLM+edge-TTS级联系统测试 ===")
    print("使用女声: zh-CN-XiaoyiNeural\n")
    
    try:
        # 初始化级联系统
        pipeline = ASRLLMEdgeTTSPipeline(
            asr_model_dir="./models/SenseVoice",
            edge_tts_voice="zh-CN-XiaoyiNeural",  # 使用女声
            edge_tts_rate="+0%",
            edge_tts_volume="+0%"
        )
        
        round_count = 1
        while True:
            print(f"\n=== 对话轮次 {round_count} ===")
            print("请说话，系统将识别您的语音并生成响应...")
            print("（提示：录音将持续5秒，请在这段时间内完成说话）")
            
            # 运行完整流程
            result = await pipeline.run(duration=5)
            
            if result["success"]:
                print("\n✅ 流程执行成功！")
                print(f"语音识别结果: {result['asr_text']}")
                print(f"大模型响应: {result['llm_response']}")
            else:
                print(f"\n❌ 流程执行失败: {result['error']}")
            
            round_count += 1
            print("\n--- 准备下一轮对话 ---")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n用户终止对话")
    except Exception as e:
        print(f"\n程序执行失败: {str(e)}")
    finally:
        print("\n测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
