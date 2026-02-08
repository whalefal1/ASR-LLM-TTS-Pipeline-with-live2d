"""
WebSocket服务器，用于实时更新Live2D对话框
"""
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Live2DWebSocketServer:
    """Live2D WebSocket服务器"""
    
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
    
    async def register(self, websocket):
        """注册新的客户端连接"""
        self.clients.add(websocket)
        logger.info(f"客户端已连接: {websocket.remote_address}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logger.info(f"客户端已断开: {websocket.remote_address}")
    
    async def broadcast_message(self, message):
        """向所有客户端广播消息"""
        if not self.clients:
            logger.warning("没有连接的客户端")
            return False
        
        # 移除断开的客户端
        disconnected = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                disconnected.append(client)
        
        # 清理断开的客户端
        for client in disconnected:
            self.clients.remove(client)
        
        if disconnected:
            logger.warning(f"移除了 {len(disconnected)} 个断开的客户端")
        
        return len(self.clients) > 0
    
    async def send_llm_response(self, response, duration=8000):
        """发送LLM回复消息"""
        message = {
            "type": "llm_response",
            "content": response,
            "duration": duration
        }
        return await self.broadcast_message(json.dumps(message))
    
    async def send_status(self, status):
        """发送状态消息"""
        message = {
            "type": "status",
            "content": status
        }
        return await self.broadcast_message(json.dumps(message))
    
    async def start(self):
        """启动WebSocket服务器"""
        logger.info(f"启动WebSocket服务器: ws://{self.host}:{self.port}")
        self.server = await websockets.serve(
            self.register,
            self.host,
            self.port
        )
        logger.info("WebSocket服务器已启动")
    
    async def stop(self):
        """停止WebSocket服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket服务器已停止")

# 全局WebSocket服务器实例
ws_server = None

async def get_ws_server():
    """获取WebSocket服务器实例（单例模式）"""
    global ws_server
    if ws_server is None:
        ws_server = Live2DWebSocketServer()
        await ws_server.start()
    return ws_server

async def send_llm_response(response, duration=8000):
    """发送LLM回复（便捷函数）"""
    server = await get_ws_server()
    return await server.send_llm_response(response, duration)

async def send_status(status):
    """发送状态消息（便捷函数）"""
    server = await get_ws_server()
    return await server.send_status(status)

def run_ws_server():
    """运行WebSocket服务器（阻塞模式）"""
    async def main():
        server = await get_ws_server()
        logger.info("WebSocket服务器运行中，按Ctrl+C停止...")
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            await server.stop()
    
    asyncio.run(main())

if __name__ == "__main__":
    run_ws_server()