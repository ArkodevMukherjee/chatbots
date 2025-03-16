import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

st.title("Your Personal Chatbot")

# Ensure API key exists in secrets
if "api_key" not in st.secrets:
    st.error("Missing Hugging Face API key! Add it to Streamlit secrets.")
    st.stop()

api_key = st.secrets["api_key"]

# Properly initialize the Hugging Face model
llm_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

# Pass `llm_model` inside `ChatHuggingFace`
model = ChatHuggingFace(llm=llm_model)

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
