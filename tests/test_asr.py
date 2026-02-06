# coding=utf-8
# 测试ASR语音识别功能

from src.asr.asr_model import ASRModule
import time

def test_asr():
    """
    测试ASR语音识别功能
    """
    print("=== ASR语音识别功能测试 ===")
    
    try:
        # 初始化ASR模块
        print("正在初始化ASR模块...")
        asr = ASRModule(model_dir="./SenseVoice")
        print("ASR模块初始化完成！")
        
        # 测试录音和识别
        print("\n请说话，系统将识别您的语音...")
        print("（提示：录音将持续10秒，请在这段时间内完成说话）")
        
        start_time = time.time()
        
        # 使用固定长度录音（10秒）
        text = asr.record_and_recognize(duration=10, use_vad=False)
        
        end_time = time.time()
        
        print(f"\n识别完成，耗时: {end_time - start_time:.2f}秒")
        print(f"识别结果: {text}")
        
        if text:
            print("\n✅ ASR语音识别测试成功！")
        else:
            print("\n❌ ASR语音识别测试失败：未识别到有效语音")
            
    except Exception as e:
        print(f"\n❌ ASR测试失败: {str(e)}")
    finally:
        print("\nASR测试完成！")

if __name__ == "__main__":
    test_asr()
