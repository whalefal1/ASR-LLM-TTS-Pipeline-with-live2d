# WebSocket实时通信功能说明

## 功能概述

本项目已实现WebSocket实时通信功能，用于在Python后端和Live2D前端之间进行实时消息传递，避免了每次对话都打开新标签页的问题。

## 架构说明

### 1. WebSocket服务器 (`src/websocket/live2d_ws_server.py`)
- 监听端口：8765
- 功能：接收Python后端的LLM回复，广播给所有连接的客户端
- 主要方法：
  - `get_ws_server()`: 获取WebSocket服务器实例（单例模式）
  - `send_llm_response(response, duration)`: 发送LLM回复消息
  - `send_status(status)`: 发送状态消息

### 2. WebSocket客户端 (`src/websocket/live2d_ws_client.py`)
- 连接地址：ws://localhost:8765
- 功能：向WebSocket服务器发送消息
- 主要方法：
  - `get_ws_client()`: 获取WebSocket客户端实例（单例模式）
  - `send_llm_response(response, duration)`: 发送LLM回复消息
  - `send_status(status)`: 发送状态消息
  - `disconnect()`: 断开连接

### 3. Live2D前端 (`live2d-widget/demo/demo.html`)
- WebSocket连接：自动连接到ws://localhost:8765
- 功能：接收WebSocket消息并实时更新Live2D对话框
- 主要特性：
  - 自动重连机制（最多5次）
  - 支持LLM回复消息和状态消息
  - 保持最高优先级显示8秒，不被其他提示打断

## 使用方法

### 方法1：独立运行WebSocket服务器

#### 启动WebSocket服务器
```bash
python -m src.websocket.live2d_ws_server
```

#### 启动Live2D HTTP服务器
```bash
cd live2d-widget
python -m http.server 8000
```

#### 在浏览器中打开Live2D页面
```
http://localhost:8000/demo/demo.html
```

#### 运行测试脚本
```bash
python tests/test_websocket.py
```

### 方法2：集成到ASR-LLM-TTS流程

#### 运行多轮对话测试
```bash
python tests/test_live2d_integration.py
```

该脚本会自动：
1. 启动WebSocket服务器（后台线程）
2. 打开Live2D页面（只打开一次）
3. 进行多轮对话
4. 通过WebSocket实时更新Live2D对话框

## 消息格式

### LLM回复消息
```json
{
  "type": "llm_response",
  "content": "LLM回复内容",
  "duration": 8000
}
```

### 状态消息
```json
{
  "type": "status",
  "content": "状态信息"
}
```

## 代码示例

### Python后端发送LLM回复
```python
import asyncio
from src.websocket.live2d_ws_server import send_llm_response

async def send_response():
    response = "你好！这是LLM的回复。"
    success = await send_llm_response(response, duration=8000)
    if success:
        print("消息发送成功")
    else:
        print("消息发送失败（可能没有连接的客户端）")

asyncio.run(send_response())
```

### 前端接收消息（自动处理）
Live2D页面会自动连接WebSocket并接收消息，无需额外代码。

## 优势

1. **避免重复打开标签页**：只在第一次打开Live2D页面，后续对话在同一页面中更新
2. **实时通信**：WebSocket提供低延迟的双向通信
3. **自动重连**：前端具有自动重连机制，提高稳定性
4. **优先级控制**：LLM回复以最高优先级显示，不被其他提示打断
5. **易于扩展**：支持多种消息类型，便于扩展功能

## 故障排除

### 问题1：WebSocket连接失败
- 检查WebSocket服务器是否启动
- 检查端口8765是否被占用
- 查看浏览器控制台的错误信息

### 问题2：消息发送失败
- 检查是否有客户端连接到WebSocket服务器
- 确认Live2D页面已打开并成功连接
- 查看服务器日志

### 问题3：Live2D对话框不更新
- 检查浏览器控制台是否有JavaScript错误
- 确认WebSocket连接状态
- 查看消息格式是否正确

## 技术栈

- **后端**：Python + websockets库
- **前端**：原生WebSocket API
- **通信协议**：WebSocket（ws://）
- **消息格式**：JSON

## 性能优化

1. **单例模式**：WebSocket服务器和客户端使用单例模式，避免重复创建
2. **异步处理**：使用async/await提高性能
3. **连接池**：支持多个客户端同时连接
4. **自动清理**：自动移除断开的客户端

## 未来扩展

1. 支持更多消息类型（如：表情控制、动作触发）
2. 添加消息确认机制
3. 支持消息队列
4. 添加消息加密
5. 支持跨域通信