import streamlit as st
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

st.title("Your Personal Chatbot")

# Get API key from Streamlit secrets
api_key = st.secrets["api_key"]

# Initialize the chat model
model = ChatHuggingFace(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

# Initialize chat history if not present
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

# User input
text = st.text_input("Enter your message")
button = st.button("Send")

if button and text.strip():
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=text))

    # Get AI response using chat history
    bot_response = model.invoke(st.session_state.messages)

    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=bot_response))

    # Refresh the page to show new messages
    st.rerun()
