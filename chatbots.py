import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

st.title("Your personal chatbot is here")

api_key = st.secrets["api_key"]

model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Hello! I am your personal chatbot. How can I help you today?")
    ]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        role = "**You:**"
    elif isinstance(msg, AIMessage):
        role = "**Bot:**"
    else:
        role = "**System:**"
    st.write(role, msg.content)

text = st.text_input("Enter what you want to query")
button = st.button("Send")

if button and text.strip():
    st.session_state.messages.append(HumanMessage(content=text))
    
    # Pass only the latest user message
    bot_response = model.invoke(text).strip()
    
    st.session_state.messages.append(AIMessage(content=bot_response))
    st.rerun()
