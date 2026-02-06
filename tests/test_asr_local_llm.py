# coding=utf-8
# ASR + 本地LLM 整合测试

import torch
import time
import os
from src.asr.asr_model import ASRModule
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

def get_available_device():
    """
    获取可用的设备
    """
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU 可用，内存: {gpu_mem:.2f}GB")
            return "cuda:0"
        except Exception as e:
            print(f"GPU 检查失败: {e}")
            return "cpu"
    else:
        print("GPU 不可用，使用 CPU")
        return "cpu"

def load_local_llm(model_path):
    """
    加载本地LLM模型
    """
    print(f"\n开始加载本地LLM模型: {model_path}")
    start_time = time.time()
    
    device = get_available_device()
    
    try:
        if device == "cuda:0":
            try:
                print("尝试使用 INT8 量化加载模型...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map=device,
                    trust_remote_code=True
                )
                print("模型加载成功 (INT8 量化)")
            except Exception as e:
                print(f"INT8 量化加载失败: {e}")
                print("尝试使用 BF16 精度加载模型...")
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
                print("模型加载成功 (BF16 精度)")
        else:
            print("使用 CPU 模式加载模型...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            print("模型加载成功 (CPU 模式)")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("分词器加载成功")
        
        end_time = time.time()
        print(f"LLM模型加载完成，耗时: {end_time - start_time:.2f}秒")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"LLM模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_response(model, tokenizer, device, text, max_new_tokens=100):
    """
    使用本地LLM生成响应
    """
    try:
        print(f"\nLLM处理中...")
        start_time = time.time()
        
        # 尝试使用chat接口
        if hasattr(model, "chat"):
            print("使用chat接口...")
            response = model.chat(
                tokenizer,
                text,
                history=[],
                max_new_tokens=max_new_tokens
            )
        else:
            # 使用generate方法
            print("使用generate方法...")
            
            # 改进prompt格式，添加简洁回答的指示
            prompt = f"{text}\n\n请用简短的一句话回答，不超过50字。"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 设置停止token
            if tokenizer.eos_token_id is not None:
                eos_token_id = tokenizer.eos_token_id
            else:
                eos_token_id = tokenizer.pad_token_id
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id
                )
            
            # 只解码新生成的部分
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        end_time = time.time()
        print(f"LLM生成完成，耗时: {end_time - start_time:.2f}秒")
        
        return response
    except Exception as e:
        print(f"LLM生成失败: {e}")
        import traceback
        traceback.print_exc()
        return "抱歉，生成响应失败。"

def test_asr_llm():
    """
    测试ASR + 本地LLM整合
    """
    print("=== ASR + 本地LLM 整合测试 ===")
    
    # 1. 初始化ASR
    print("\n1. 初始化ASR模块...")
    try:
        asr = ASRModule(model_dir="./SenseVoice")
        print("✅ ASR模块初始化成功")
    except Exception as e:
        print(f"❌ ASR模块初始化失败: {e}")
        return
    
    # 2. 加载本地LLM
    print("\n2. 加载本地LLM模型...")
    model_path = "./models/qwen_vl/qwen/Qwen2-VL-2B"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    model, tokenizer, device = load_local_llm(model_path)
    
    if not model or not tokenizer:
        print("❌ LLM模型加载失败")
        return
    
    print("✅ LLM模型加载成功")
    
    # 3. 开始测试
    print("\n3. 开始语音识别和LLM处理...")
    print("请说话，系统将识别您的语音并使用本地LLM生成响应...")
    print("（录音将持续5秒）")
    
    try:
        # ASR语音识别
        print("\n--- 开始录音 ---")
        start_time = time.time()
        recognized_text = asr.record_and_recognize(duration=5, use_vad=False)
        end_time = time.time()
        
        print(f"\n--- ASR识别完成 ---")
        print(f"识别耗时: {end_time - start_time:.2f}秒")
        print(f"识别结果: {recognized_text}")
        
        if not recognized_text:
            print("❌ 未识别到有效语音")
            return
        
        # LLM生成响应
        print(f"\n--- LLM生成响应 ---")
        response = generate_response(model, tokenizer, device, recognized_text)
        
        print(f"\n--- 最终结果 ---")
        print(f"用户输入: {recognized_text}")
        print(f"LLM响应: {response}")
        print("\n✅ 测试成功！")
        
    except KeyboardInterrupt:
        print("\n🔄 用户终止测试")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_asr_llm()