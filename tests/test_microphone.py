# coding=utf-8
# 测试麦克风功能

import pyaudio
import wave
import time

def test_microphone():
    """
    测试麦克风功能
    """
    print("=== 麦克风功能测试 ===")
    
    # 录音参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "test_microphone_output.wav"
    
    try:
        # 初始化PyAudio
        print("正在初始化音频设备...")
        audio = pyaudio.PyAudio()
        
        # 列出可用的输入设备
        print("\n可用的音频输入设备:")
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                print(f"设备 {i}: {device_info.get('name')}")
        
        # 打开音频流
        print(f"\n开始录制 {RECORD_SECONDS} 秒音频...")
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        frames = []
        
        # 录制音频
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            # 显示录制进度
            if i % 10 == 0:
                print(f"录制中... {int((i / (RATE / CHUNK * RECORD_SECONDS)) * 100)}%")
        
        print("录制完成！")
        
        # 停止流
        stream.stop_stream()
        stream.close()
        
        # 保存音频文件
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        # 关闭PyAudio
        audio.terminate()
        
        print(f"\n音频已保存到: {WAVE_OUTPUT_FILENAME}")
        print(f"文件大小: {len(b''.join(frames))} 字节")
        
        # 分析音频数据
        audio_data_size = len(b''.join(frames))
        if audio_data_size > 0:
            print("\n✅ 麦克风测试成功！")
            print("提示：您可以播放生成的音频文件验证录制效果")
        else:
            print("\n⚠️  麦克风测试警告：未捕获到音频数据")
            print("请检查麦克风连接和设置")
            
    except Exception as e:
        print(f"\n❌ 麦克风测试失败: {str(e)}")
        print("请检查麦克风连接和权限设置")
    finally:
        print("\n麦克风测试完成！")

if __name__ == "__main__":
    test_microphone()
