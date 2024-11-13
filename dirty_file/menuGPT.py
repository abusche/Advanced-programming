import streamlit as st




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