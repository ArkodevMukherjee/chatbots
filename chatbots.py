import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_core.prompts import PromptTemplate


st.title("Your personal chatbot is here")

if "api_key" not in st.secrets:
    st.error("Hugging Face API key is missing! Please add it in Streamlit Secrets.")
else:
    api_key = st.secrets["api_key"]

model = HuggingFaceEndpoint(
    repo_id="DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    huggingfacehub_api_key=api_key
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Hello! I am your personal chatbot. How can I help you today?"),
        AIMessage(content="I am a chatbot.")
    ]

# Display chat history
for msg in st.session_state.messages:
    role = "**You:**" if isinstance(msg, HumanMessage) else "**Bot:**"
    st.write(role, msg.content)


text = st.text_input("Enter what you want to query")
button = st.button("Send")


if button and text != "":
    st.session_state.messages.append(HumanMessage(content=text))
    text =""
    botMessages = model.invoke(st.session_state.messages).split("AI:")[-1].strip()
    st.session_state.messages.append(AIMessage(content=botMessages))
    st.rerun()
