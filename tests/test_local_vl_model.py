# coding=utf-8
# 测试本地 Qwen2-VL-2B 模型

import torch
import time
import os
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

def get_available_device():
    """
    获取可用的设备
    """
    if torch.cuda.is_available():
        # 检查GPU内存
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

def load_model(model_path):
    """
    加载模型
    """
    print(f"开始加载模型: {model_path}")
    start_time = time.time()
    
    device = get_available_device()
    
    try:
        # 尝试使用INT8量化（如果GPU内存不足）
        if device == "cuda:0":
            try:
                from transformers import BitsAndBytesConfig
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
            # CPU 模式
            print("使用 CPU 模式加载模型...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            print("模型加载成功 (CPU 模式)")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("分词器加载成功")
        
        end_time = time.time()
        print(f"模型加载完成，耗时: {end_time - start_time:.2f}秒")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_model(model, tokenizer, device):
    """
    测试模型
    """
    if not model or not tokenizer:
        print("模型或分词器未加载，无法测试")
        return
    
    print("\n=== 开始测试模型 ===")
    
    # 测试文本输入
    print("\n1. 测试文本输入:")
    text_input = "你好，请问你是谁？"
    print(f"输入: {text_input}")
    
    try:
        start_time = time.time()
        
        # 构建输入
        if hasattr(model, "chat"):
            # 使用chat接口
            response = model.chat(
                tokenizer,
                text_input,
                history=[],
                max_new_tokens=100
            )
        else:
            # 使用generate方法
            print("使用generate方法生成响应...")
            inputs = tokenizer(text_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = time.time()
        print(f"输出: {response}")
        print(f"生成耗时: {end_time - start_time:.2f}秒")
        
    except Exception as e:
        print(f"文本测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试多轮对话
    print("\n2. 测试多轮对话:")
    try:
        history = []
        
        # 第一轮
        user_input1 = "你好，我叫小明"
        print(f"用户: {user_input1}")
        
        if hasattr(model, "chat"):
            assistant_response1 = model.chat(
                tokenizer,
                user_input1,
                history=history,
                max_new_tokens=50
            )
            history.append((user_input1, assistant_response1))
            print(f"助手: {assistant_response1}")
        
        # 第二轮
        user_input2 = "你能帮我做什么？"
        print(f"用户: {user_input2}")
        
        if hasattr(model, "chat"):
            assistant_response2 = model.chat(
                tokenizer,
                user_input2,
                history=history,
                max_new_tokens=50
            )
            print(f"助手: {assistant_response2}")
            
    except Exception as e:
        print(f"多轮对话测试失败: {e}")
    
    print("\n=== 测试完成 ===")

def main():
    """
    主函数
    """
    print("=== 测试本地 Qwen2-VL-2B 模型 ===")
    
    # 模型路径
    model_path = "./models/qwen_vl/qwen/Qwen2-VL-2B"
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先下载模型")
        return
    
    # 加载模型
    model, tokenizer, device = load_model(model_path)
    
    # 测试模型
    if model and tokenizer:
        test_model(model, tokenizer, device)
    
    print("\n测试结束！")

if __name__ == "__main__":
    main()