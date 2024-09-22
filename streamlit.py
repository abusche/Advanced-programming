import os
import sqlite3
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Load the environment and the API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load a local database
db = SQLDatabase.from_uri("sqlite:///")

# Configuration of the Web page
st.set_page_config(page_title ="Crous menu ChatBot")

# Give a title
st.title("Crous menu ChatBot")

# Load the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a chain with the LangChain tools
chain = SQLDatabaseToolkit(db=db, llm=llm)
tools = chain.get_tools()

# Load the prompt
with open('prompt.txt', 'r', encoding='utf-8') as prompt:
    prompt = prompt.read().format(
            table_names=db.get_usable_table_names()
        )

# Send the prompt
system_message = SystemMessage(content=prompt)

# Create the agent
agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

# Initialization of session state and first bot message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I help you today?"}
    ]

# Setting up a section with sample questions
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with st.sidebar:
    st.image("img/", width=280)
    st.subheader("Pre-selected Questions :")
    preselected_questions = [
        "",
        "",
        ""
    ]

    for question in preselected_questions:
        if st.button(question):
            st.session_state.selected_question = question

# Viewing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for question
query = st.chat_input("Ask me a question...")

# Use the pre-selected question if available
if st.session_state.selected_question:
    query = st.session_state.selected_question
    st.session_state.selected_question = None

# Otherwise we take the question we ask
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Question display
    with st.chat_message("user"):
        st.markdown(query)
        
    # Some aspects of visualization
    with st.spinner("Loading..."):
        with st.chat_message("assistant"):
            
            # Query generation and response generation
            for s in agent_executor.stream(
                    {"messages": [HumanMessage(content=query)]}
                ):
                    if 'agent' in s and 'messages' in s['agent']:
                        for message in s['agent']['messages']:
                            if isinstance(message, AIMessage) and message.content.strip():
                                response = message.content
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# JavaScript code for automatic scrolling down
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column-reverse;
    }
    </style>
    <script>
    const chatContainer = parent.document.querySelector('.main');
    chatContainer.scrollTop = chatContainer.scrollHeight;
    </script>
""", unsafe_allow_html=True)