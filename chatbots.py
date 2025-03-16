import streamlit as st
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

st.title("Your Personal Chatbot")

# Get API key from Streamlit secrets
api_key = st.secrets["api_key"]

# Load the model properly
model = ChatHuggingFace(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Hello! I am your personal chatbot. How can I help you today?")
    ]

# Display chat history
for msg in st.session_state.messages:
    role = "**You:**" if isinstance(msg, HumanMessage) else "**Bot:**"
    st.write(role, msg.content)

# User input
text = st.text_input("Enter your query")
button = st.button("Send")

# Process user input
if button and text.strip():
    st.session_state.messages.append(HumanMessage(content=text))

    # Call model and get response
    bot_response = model.invoke(st.session_state.messages)

    # Append AI response
    st.session_state.messages.append(AIMessage(content=bot_response))

    st.rerun()
