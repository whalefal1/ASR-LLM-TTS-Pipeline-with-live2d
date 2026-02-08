"""
WebSocket客户端，用于Live2D对话框更新
"""
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Live2DWebSocketClient:
    """Live2D WebSocket客户端"""
    
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.connected = False
    
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            logger.info(f"已连接到WebSocket服务器: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接WebSocket服务器失败: {e}")
            return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("已断开WebSocket连接")
    
    async def send_message(self, message):
        """发送消息到服务器"""
        if not self.connected:
            logger.warning("未连接到WebSocket服务器")
            return False
        
        try:
            await self.websocket.send(message)
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def send_llm_response(self, response, duration=8000):
        """发送LLM回复消息"""
        message = {
            "type": "llm_response",
            "content": response,
            "duration": duration
        }
        return await self.send_message(json.dumps(message))
    
    async def send_status(self, status):
        """发送状态消息"""
        message = {
            "type": "status",
            "content": status
        }
        return await self.send_message(json.dumps(message))

# 全局WebSocket客户端实例
ws_client = None

async def get_ws_client():
    """获取WebSocket客户端实例（单例模式）"""
    global ws_client
    if ws_client is None:
        ws_client = Live2DWebSocketClient()
        await ws_client.connect()
    return ws_client

async def send_llm_response(response, duration=8000):
    """发送LLM回复（便捷函数）"""
    client = await get_ws_client()
    return await client.send_llm_response(response, duration)

async def send_status(status):
    """发送状态消息（便捷函数）"""
    client = await get_ws_client()
    return await client.send_status(status)

async def disconnect():
    """断开连接（便捷函数）"""
    global ws_client
    if ws_client:
        await ws_client.disconnect()
        ws_client = None

if __name__ == "__main__":
    async def test():
        client = Live2DWebSocketClient()
        if await client.connect():
            await client.send_llm_response("测试消息", 5000)
            await asyncio.sleep(2)
            await client.disconnect()
    
    asyncio.run(test())