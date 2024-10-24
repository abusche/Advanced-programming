import os
import streamlit as st
from ScrapeMenuToday import ScrapeMenu
from hf_functions import get_link
from ScrapeMenuToday import ScrapeMenu
from hf_functions import get_link

# Configuration of the Web page
st.set_page_config(page_title ="Crous menu ChatBot")

# Give a title
st.title("Crous menu ChatBot")



# Initialization of session state and first bot message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I help you today?"}
    ]

# Setting up a section with sample questions
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with st.sidebar:
    st.subheader("Pre-selected Questions :")
    preselected_questions = [
        "What is the menu at Esplanade ?",
        "What kind of food can I find at Gallia ?",
        "I'm at Koeningshoffen, what is the menu ?"
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
            response = ScrapeMenu(get_link(query)[0])["lunch"][0]
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