# -*- coding: utf-8  -*-


from utils import path, OPENAI_CONFIG

import base64
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify
# 导入 LangChain 相关库
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

app = Flask(__name__)

# --- 1. 自定义 API 配置 ---


# 初始化 LLM (使用你的自定义网关和 Key)
# 这里的 max_retries 对应你代码中的重试逻辑，LangChain 内部已封装
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_CONFIG["apiKey"],
    base_url=OPENAI_CONFIG["baseURL"],
    temperature=0,
    max_retries=3
)


# --- 2. RAG 逻辑实现 ---
def setup_rag_agent():
    # 注意：Embeddings 同样需要指向你的自定义网关
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_CONFIG["apiKey"],
        base_url=OPENAI_CONFIG["baseURL"]
    )

    # 模拟用户上传的个人资料数据
    mock_data = ["用户姓名：张三", "职业：高级架构师", "偏好：喜欢使用 Python 和极简 UI 设计",
                 "项目经历：开发过多个多智能体协作系统"]

    # 创建向量库
    vectorstore = FAISS.from_texts(mock_data, embeddings)

    # 创建 RAG 检索链
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return rag_chain


rag_tool_executor = setup_rag_agent()

# --- 3. 定义多 Agent 工具库 ---
tools = [
    Tool(
        name="User_Profile_QA",
        func=lambda q: rag_tool_executor.invoke(q),
        description="当需要了解用户的个人背景、职业、资料或偏好时，调用此工具。"
    ),
    Tool(
        name="Logic_Expert",
        func=lambda q: llm.invoke(f"请作为逻辑专家分析以下问题：{q}"),
        description="当需要深度逻辑推理或架构建议时，调用此专家 Agent。"
    )
]

# --- 4. 初始化主 Agent (调度中心) ---
# 使用 ConversationBufferMemory 来实现记忆功能
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_manager = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # 在终端查看 Agent 相互调用的过程
    memory=memory,
    handle_parsing_errors=True
)


# --- 5. Flask 路由 ---
@app.route('/')
def index():
    # 假设你的 html 文件名为 index.html
    try:
        with open(f"{path}/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "请确保 index.html 文件在当前目录下。"


# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_input = data.get("message")

#     try:
#         # Agent 会根据输入决定是直接回答，还是调用 RAG 工具查找资料
#         response = agent_manager.run(input=user_input)
#         return jsonify({"response": response})
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"response": "抱歉，我遇到了一些逻辑处理错误。"}), 500


# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_input = data.get("message")
#     mode = data.get("mode") # 'pe', 'rag', or 'agent'


#     if mode == 'pe':
#         # Simple GPT call
#         response = llm.invoke(user_input).content
#     elif mode == 'rag':
#         # RAG logic
#         response = rag_tool_executor.invoke(user_input)
#     else:
#         # Multi-Agent Orchestrator
#         response = agent_manager.run(input=user_input)


#     return jsonify({"response": response})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    is_new_user = data.get("is_new_user", True)

    # 根据是否是新用户，动态调整语气
    tone_instruction = "Be polite and introduce your capabilities briefly." if is_new_user \
        else "Be warm, use friendly language like 'good to see you again', and avoid repetitive introductions."

    # 结合之前的角色设置
    system_prompt = f"{data['config']['role']}. {tone_instruction}"

    # ... 后续调用 LLM 的逻辑
    user_message = data.get("message")
    mode = data.get("mode")

    # Extract configurations from the new UI fields
    config = data.get("config", {})
    user_role = config.get("role", "You are a helpful assistant.")
    restrictions = config.get("restrictions", "None")
    max_tokens = config.get("max_tokens", 512)
    max_tools = config.get("max_tools", 5)

    # Construct the final internal prompt
    combined_system_prompt = (
        f"System Role: {user_role}\n"
        f"Topic Restrictions: {restrictions}\n"
        f"Constraint: If the user asks about restricted topics, politely decline."
    )

    if mode == 'pe':
        # Direct call with custom settings
        response = llm.invoke(f"{combined_system_prompt}\nUser: {user_message}").content
    elif mode == 'agent':
        # Apply tool limit to the agent executor
        agent_manager.max_iterations = max_tools
        # Run agent with the custom role prepended
        response = agent_manager.run(input=f"Context: {combined_system_prompt}. User Query: {user_message}")

    return jsonify({"response": response})


# --- 6. 启动配置 ---
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == '__main__':
    # 自动打开界面
    Timer(1.5, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False)


