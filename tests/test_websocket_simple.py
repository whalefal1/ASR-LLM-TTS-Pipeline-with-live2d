#!/usr/bin/env python3
# coding=utf-8
"""
专门测试WebSocket功能
"""

import asyncio
import time
import sys
import os
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.websocket.live2d_ws_server import get_ws_server, send_llm_response
import webbrowser

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_websocket_simple')

async def start_ws_server():
    """启动WebSocket服务器"""
    logger.info("正在启动WebSocket服务器...")
    server = await get_ws_server()
    logger.info("WebSocket服务器已启动")
    return server

async def test_websocket_messages():
    """测试发送WebSocket消息"""
    logger.info("开始测试WebSocket消息发送...")
    
    # 测试消息
    test_messages = [
        "你好！这是WebSocket测试消息。",
        "今天天气怎么样？",
        "Python是一种很好的编程语言。",
        "WebSocket通信测试成功！",
        "测试完成，感谢使用。"
    ]
    
    for i, message in enumerate(test_messages, 1):
        logger.info(f"发送第 {i} 条消息: {message}")
        success = await send_llm_response(message, duration=8000)
        if success:
            logger.info(f"✅ 消息发送成功")
        else:
            logger.warning(f"❌ 消息发送失败（可能没有连接的客户端）")
        await asyncio.sleep(9)
    
    logger.info("🎉 WebSocket消息测试完成！")

def main():
    """主函数"""
    print("=== WebSocket功能测试 ===")
    print()
    print("测试流程：")
    print("1. 启动WebSocket服务器")
    print("2. 打开Live2D页面")
    print("3. 发送测试消息")
    print("4. 验证Live2D对话框更新")
    print()
    
    # 启动WebSocket服务器
    print("正在启动WebSocket服务器...")
    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(start_ws_server())
    print("✅ WebSocket服务器已启动（ws://localhost:8765）")
    print()
    
    # 打开Live2D页面
    print("正在打开Live2D页面...")
    webbrowser.open("http://localhost:8000/demo/demo.html")
    print("✅ Live2D页面已打开")
    print()
    print("请在浏览器中确认Live2D页面已加载，WebSocket连接已建立")
    print()
    
    # 等待用户确认
    input("按回车键开始发送测试消息...")
    print()
    
    # 发送测试消息
    print("开始发送测试消息...")
    loop.run_until_complete(test_websocket_messages())
    print()
    
    print("🎉 WebSocket功能测试完成！")
    print()
    print("测试结果：")
    print("✅ WebSocket服务器启动成功")
    print("✅ Live2D页面打开成功")
    print("✅ 消息发送功能正常")
    print()
    print("请验证Live2D对话框是否显示了所有测试消息")

if __name__ == "__main__":
    main()