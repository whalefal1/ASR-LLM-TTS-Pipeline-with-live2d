from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import threading
import time
import sys

# 加载环境变量
load_dotenv()

# 创建ChatOpenAI实例（用于调用智谱GLM-4模型）
model = ChatOpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model_name="glm-4",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.7
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，用中文回答用户的问题。"),
    ("user", "{query}")
])

# 创建输出解析器
output_parser = StrOutputParser()

# 构建链式结构
chain = prompt | model | output_parser

# 定义一个用于外部调用的函数
def call_llm(query):
    """
    调用大模型获取回复
    :param query: 用户的查询文本
    :return: 大模型的回复文本
    """
    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"调用大模型出错：{str(e)}")
        return "抱歉，我暂时无法回答您的问题，请稍后再试。"

# 超时管理类
class TimeoutManager:
    def __init__(self, timeout_seconds=300):  # 默认5分钟超时
        self.timeout_seconds = timeout_seconds
        self.last_activity_time = time.time()
        self.timeout_event = threading.Event()
        self.timeout_thread = threading.Thread(target=self._timeout_monitor, daemon=True)
    
    def reset_timeout(self):
        """重置超时计时器"""
        self.last_activity_time = time.time()
        self.timeout_event.clear()
    
    def start(self):
        """启动超时监控线程"""
        self.timeout_thread.start()
    
    def stop(self):
        """停止超时监控"""
        self.timeout_event.set()
        if self.timeout_thread.is_alive():
            self.timeout_thread.join()
    
    def _timeout_monitor(self):
        """超时监控线程函数"""
        while not self.timeout_event.wait(1):  # 每秒检查一次
            if time.time() - self.last_activity_time > self.timeout_seconds:
                print("\n\n程序已超过5分钟无活动，自动退出！")
                self.stop()
                sys.exit(0)

# 定义主函数
def main():
    print("LangChain大模型调用示例")
    print("请输入您的问题（输入'退出'、'exit'、'quit'或'q'结束程序，按Ctrl+C也可退出）：")
    
    # 创建并启动超时管理器
    timeout_manager = TimeoutManager()
    timeout_manager.start()
    
    try:
        while True:
            query = input("\n用户：")
            # 重置超时计时器（用户有活动）
            timeout_manager.reset_timeout()
            
            # 支持多种退出命令
            exit_commands = ["退出", "exit", "quit", "q"]
            if query.strip().lower() in exit_commands:
                print("程序结束，感谢使用！")
                break
            
            try:
                # 调用大模型
                result = chain.invoke({"query": query})
                print(f"助手：{result}")
            except Exception as e:
                print(f"调用出错：{str(e)}")
                print("请检查API密钥是否正确配置，或网络连接是否正常。")
    except KeyboardInterrupt:
        # 处理Ctrl+C键盘中断
        print("\n\n程序已通过键盘中断退出，感谢使用！")
    finally:
        # 无论如何都要停止超时监控
        timeout_manager.stop()

if __name__ == "__main__":
    main()