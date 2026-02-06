# coding=utf-8
# 实时VAD检测测试（版本5）

import pyaudio
import webrtcvad
import numpy as np
import wave
import os
import time

class RealTimeVAD:
    """
    实时VAD检测类
    使用webrtcvad进行实时语音活动检测
    """
    
    def __init__(self, 
                 sample_rate=16000, 
                 channels=1, 
                 chunk_duration=20,  # 每个检测chunk的时长（毫秒）
                 vad_window=500,     # VAD检测窗口时长（毫秒）
                 activation_threshold=0.4,  # 有效语音激活率阈值
                 vad_mode=3):        # VAD模式（0-3，3最敏感）
        """
        初始化VAD检测器
        
        Args:
            sample_rate (int): 采样率
            channels (int): 声道数
            chunk_duration (int): 每个检测chunk的时长（毫秒）
            vad_window (int): VAD检测窗口时长（毫秒）
            activation_threshold (float): 有效语音激活率阈值
            vad_mode (int): VAD模式（0-3，3最敏感）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration  # 20ms
        self.vad_window = vad_window          # 500ms
        self.activation_threshold = activation_threshold  # 40%
        
        # 计算参数
        self.chunk_size = int(sample_rate * chunk_duration / 1000)
        self.window_chunks = int(vad_window / chunk_duration)  # 25个chunk
        self.activation_count = int(self.window_chunks * activation_threshold)  # 10个激活chunk
        
        # 初始化VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)
        
        # 初始化PyAudio
        self.pa = pyaudio.PyAudio()
        
        # 测试麦克风
        self._test_microphone()
        
        # 音频缓冲区
        self.audio_buffer = []
        self.vad_history = []
        
        print(f"实时VAD检测器初始化完成")
        print(f"配置参数：")
        print(f"  采样率: {sample_rate}Hz")
        print(f"  声道数: {channels}")
        print(f"  检测chunk: {chunk_duration}ms")
        print(f"  检测窗口: {vad_window}ms")
        print(f"  激活阈值: {activation_threshold*100}%")
        print(f"  窗口chunk数: {self.window_chunks}")
        print(f"  激活所需chunk数: {self.activation_count}")
        print(f"  VAD模式: {vad_mode} (0-3，3最敏感)")
    
    def _test_microphone(self):
        """
        测试麦克风是否正常工作
        """
        try:
            print("正在测试麦克风...")
            
            # 打开临时流测试麦克风
            test_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # 读取一小段数据测试
            test_data = test_stream.read(self.chunk_size)
            test_stream.stop_stream()
            test_stream.close()
            
            # 检查数据是否有声音
            audio_data = np.frombuffer(test_data, dtype=np.int16)
            energy = np.sqrt(np.mean(np.square(audio_data)))
            
            print(f"麦克风测试完成，当前能量: {energy:.2f}")
            if energy > 100:
                print("✅ 麦克风正常，检测到声音")
            else:
                print("⚠️  麦克风能量较低，请检查麦克风是否开启")
                
        except Exception as e:
            print(f"❌ 麦克风测试失败: {e}")
            print("请检查麦克风是否正确连接")
    
    def is_speech(self, audio_chunk):
        """
        检测音频chunk是否包含语音
        
        Args:
            audio_chunk (bytes): 音频数据
            
        Returns:
            bool: 是否包含语音
        """
        try:
            # 计算音频能量，用于调试
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(np.square(audio_data)))
            
            # VAD检测
            result = self.vad.is_speech(audio_chunk, self.sample_rate)
            
            # 每10个chunk打印一次调试信息
            if len(self.vad_history) % 10 == 0:
                print(f"能量: {energy:.2f}, VAD: {result}", end="\r")
                
            return result
        except Exception as e:
            print(f"VAD检测失败: {e}")
            return False
    
    def process_audio_stream(self, duration=30, output_file="vad_output.wav"):
        """
        处理音频流，实时VAD检测
        
        Args:
            duration (int): 最大录音时长（秒）
            output_file (str): 输出音频文件路径
            
        Returns:
            bool: 是否成功
        """
        print(f"\n=== 实时VAD检测开始 ===")
        print(f"最大录音时长: {duration}秒")
        print(f"输出文件: {output_file}")
        print("请开始说话...")
        
        try:
            # 打开音频流
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            start_time = time.time()
            total_frames = []
            active_frames = []
            silence_count = 0
            max_silence = 3  # 最大静音时间（秒）
            
            # 开始录音和VAD检测
            while time.time() - start_time < duration:
                # 读取音频数据
                data = stream.read(self.chunk_size)
                total_frames.append(data)
                
                # VAD检测
                is_speech = self.is_speech(data)
                self.vad_history.append(is_speech)
                
                # 保持VAD历史不超过窗口大小
                if len(self.vad_history) > self.window_chunks:
                    self.vad_history.pop(0)
                
                # 计算当前窗口的激活率
                if len(self.vad_history) == self.window_chunks:
                    active_count = sum(self.vad_history)
                    activation_rate = active_count / self.window_chunks
                    
                    # 检查是否为有效语音
                    if activation_rate >= self.activation_threshold:
                        # 有效语音，加入缓存
                        active_frames.extend([data])
                        silence_count = 0
                        print(f"VAD激活: {active_count}/{self.window_chunks} ({activation_rate:.2f}) 有效语音", end="\r")
                    else:
                        # 静音
                        silence_count += self.chunk_duration / 1000
                        print(f"VAD静音: {active_count}/{self.window_chunks} ({activation_rate:.2f}) 静音时间: {silence_count:.1f}s", end="\r")
                        
                        # 如果静音时间过长，停止录音
                        if silence_count >= max_silence and len(active_frames) > 0:
                            print(f"\n检测到长时间静音，停止录音")
                            break
            
            # 停止流
            stream.stop_stream()
            stream.close()
            
            # 计算统计信息
            total_duration = time.time() - start_time
            active_duration = len(active_frames) * self.chunk_duration / 1000
            active_ratio = active_duration / total_duration * 100 if total_duration > 0 else 0
            
            print(f"\n=== VAD检测完成 ===")
            print(f"总录音时长: {total_duration:.2f}秒")
            print(f"有效语音时长: {active_duration:.2f}秒")
            print(f"有效语音比例: {active_ratio:.1f}%")
            print(f"总帧数: {len(total_frames)}")
            print(f"有效语音帧数: {len(active_frames)}")
            
            # 保存有效语音
            if len(active_frames) > 0:
                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(active_frames))
                
                print(f"\n✅ 有效语音已保存到: {output_file}")
                print(f"文件大小: {os.path.getsize(output_file)} 字节")
                
                # 播放并删除音频文件
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    print("正在播放音频...")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    print("音频播放完成")
                    pygame.mixer.quit()
                    
                    # 播放完成后删除音频文件
                    if os.path.exists(output_file):
                        os.remove(output_file)
                        print(f"音频文件已删除: {output_file}")
                except Exception as e:
                    print(f"播放音频失败: {e}")
                
                return True
            else:
                print("\n❌ 未检测到有效语音")
                return False
                
        except Exception as e:
            print(f"\n❌ 处理音频流失败: {e}")
            return False
        finally:
            # 清理资源
            try:
                self.pa.terminate()
            except:
                pass
    
    def __del__(self):
        """
        析构函数，清理资源
        """
        try:
            self.pa.terminate()
        except:
            pass

class VADWithASRPipeline:
    """
    VAD+ASR级联系统
    """
    
    def __init__(self, 
                 asr_model_dir="./models/SenseVoice",
                 vad_sample_rate=16000,
                 vad_channels=1,
                 vad_chunk_duration=20,
                 vad_window=500,
                 vad_activation_threshold=0.4,
                 vad_mode=3):
        """
        初始化VAD+ASR级联系统
        
        Args:
            asr_model_dir (str): SenseVoice模型目录
            vad_sample_rate (int): VAD采样率
            vad_channels (int): VAD声道数
            vad_chunk_duration (int): VAD检测chunk时长
            vad_window (int): VAD检测窗口时长
            vad_activation_threshold (float): VAD激活阈值
            vad_mode (int): VAD模式
        """
        # 初始化VAD
        self.vad = RealTimeVAD(
            sample_rate=vad_sample_rate,
            channels=vad_channels,
            chunk_duration=vad_chunk_duration,
            vad_window=vad_window,
            activation_threshold=vad_activation_threshold,
            vad_mode=vad_mode
        )
        
        # 初始化ASR
        self._init_asr(asr_model_dir)
        
        print("VAD+ASR级联系统初始化完成")
    
    def _init_asr(self, model_dir):
        """
        初始化ASR组件
        """
        try:
            from src.asr.asr_model import ASRModule
            print("正在加载SenseVoice模型...")
            self.asr = ASRModule(model_dir=model_dir)
            print("SenseVoice模型加载完成")
        except Exception as e:
            print(f"初始化ASR组件失败: {str(e)}")
            self.asr = None
    
    def run_vad_asr(self, max_duration=30):
        """
        运行VAD+ASR流程
        
        Args:
            max_duration (int): 最大录音时长
            
        Returns:
            str: 识别的文本
        """
        print("\n=== 运行VAD+ASR流程 ===")
        
        # 第一步：VAD检测
        temp_file = "temp_vad_output.wav"
        vad_success = self.vad.process_audio_stream(duration=max_duration, output_file=temp_file)
        
        if not vad_success:
            print("VAD检测失败，无法进行语音识别")
            return ""
        
        # 第二步：ASR识别
        if not self.asr:
            print("ASR组件未初始化，无法进行语音识别")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return ""
        
        try:
            print("\n开始语音识别...")
            start_time = time.time()
            
            # 使用ASR模块识别音频文件
            # 注意：这里需要修改asr_model.py，添加从文件识别的方法
            # 暂时使用录音方法，后续可以优化
            text = self.asr.record_and_recognize(duration=1, use_vad=False)
            
            end_time = time.time()
            print(f"语音识别完成，耗时: {end_time - start_time:.2f}秒")
            print(f"识别结果: {text}")
            
            return text
        except Exception as e:
            print(f"语音识别失败: {str(e)}")
            return ""
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"临时文件已删除: {temp_file}")

def test_realtime_vad():
    """
    测试实时VAD功能
    """
    print("\n=== 测试1: 实时VAD检测 ===")
    vad = RealTimeVAD()
    vad.process_audio_stream(duration=10, output_file="vad_test_output.wav")

def test_vad_asr_pipeline():
    """
    测试VAD+ASR级联系统
    """
    print("\n=== 测试2: VAD+ASR级联系统 ===")
    pipeline = VADWithASRPipeline()
    text = pipeline.run_vad_asr(max_duration=10)
    if text:
        print(f"\n最终识别结果: {text}")
    else:
        print("\n未识别到有效语音")

def main():
    """
    主测试函数
    """
    print("=== 实时VAD检测测试（版本5） ===")
    
    try:
        # 测试1: 实时VAD检测
        test_realtime_vad()
        
        # 测试2: VAD+ASR级联系统
        test_vad_asr_pipeline()
        
    except KeyboardInterrupt:
        print("\n用户终止测试")
    except Exception as e:
        print(f"\n测试失败: {e}")
    finally:
        print("\n测试完成！")

if __name__ == "__main__":
    main()
