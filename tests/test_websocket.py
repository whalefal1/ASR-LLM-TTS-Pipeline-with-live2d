#!/usr/bin/env python3
# coding=utf-8
"""
WebSocket功能测试
"""

import asyncio
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.websocket.live2d_ws_server import get_ws_server, send_llm_response

async def test_websocket():
    """测试WebSocket服务器功能"""
    print("=== WebSocket功能测试 ===")
    print()
    
    # 启动WebSocket服务器
    print("正在启动WebSocket服务器...")
    server = await get_ws_server()
    print("✅ WebSocket服务器已启动（ws://localhost:8765）")
    print()
    
    # 等待客户端连接
    print("等待客户端连接...")
    print("请在浏览器中打开: http://localhost:8000/demo/demo.html")
    print()
    
    # 发送测试消息
    test_messages = [
        "你好！这是第一条测试消息。",
        "这是第二条测试消息。",
        "这是第三条测试消息。",
        "WebSocket通信测试成功！",
        "测试完成，即将结束。"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"发送第 {i} 条消息: {message}")
        success = await send_llm_response(message, duration=5000)
        if success:
            print(f"✅ 消息发送成功")
        else:
            print(f"❌ 消息发送失败（可能没有连接的客户端）")
        print()
        await asyncio.sleep(6)
    
    print("🎉 WebSocket功能测试完成！")

if __name__ == "__main__":
    asyncio.run(test_websocket())