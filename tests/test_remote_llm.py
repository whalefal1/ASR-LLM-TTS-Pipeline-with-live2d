# coding=utf-8
# 测试远程LLM功能

import os
from dotenv import load_dotenv
from llmModel import call_llm

def test_remote_llm():
    """
    测试远程LLM调用
    """
    print("\n=== 测试远程LLM ===")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查API密钥
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("❌ 未配置ZHIPU_API_KEY环境变量")
        print("请在.env文件中配置ZHIPU_API_KEY")
        return
    
    print(f"✅ API密钥已配置: {api_key[:10]}...")
    
    # 测试查询
    test_queries = [
        "你好，请问你是谁？",
        "今天天气怎么样？",
        "你能帮我做什么？"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n--- 测试 {i+1} ---")
        print(f"用户输入: {query}")
        
        try:
            response = call_llm(query)
            print(f"模型响应: {response}")
        except Exception as e:
            print(f"❌ 调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_remote_llm()