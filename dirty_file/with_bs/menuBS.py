import os
import streamlit as st
from langchain_openai import ChatOpenAI
from menu_functions import rag
from dotenv import load_dotenv

# Chemin du fichier .env
ENV_PATH = "C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced Programming/projet/environment/"

# Charger les variables d'environnement
load_dotenv(f"{ENV_PATH}.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

st.set_page_config(page_title ="AI’m Hungry", page_icon="img/crous.png")

# Configurer l'application Streamlit
def main():
    # Personnalisation du thème Streamlit
    st.markdown(
        """
        <style>
        body {
            background-color: #ffe6e6;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            color: #b30000;
            text-align: center;
            font-size: 3em;
            margin-bottom: 20px;
        }
        .question-input {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #ff6666;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 15px;
            font-size: 16px;
            cursor: pointer;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #e63946;
        }
        .response-box {
            background-color: #ffcccc;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
            color: #990000;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .response-header {
            color: #b30000;
            font-size: 1.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Titre de l'application
    st.markdown("<h1 class='main-header'>-🍴 AI’m Hungry -</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='main-header'>Assistant for Strasbourg university menus</h4>", unsafe_allow_html=True)
    st.write("Ask your questions about university restaurant menus. 😊")

    # Variables globales
    if "context_loaded" not in st.session_state:
        st.session_state["context_loaded"] = False
        st.session_state["rag_chain"] = None

    # Initialisation de l'état de la session et du premier message du bot
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, Feeling hungry? Let me help you 😊"}
        ]
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None
    
    # Section des questions pré-sélectionnées
    with st.sidebar:
        st.image("img/crous.png", width=180)
        st.subheader("Pre-selected Questions :")
        preselected_questions = [
            "Esplanade",
            "Gallia",
            "Pege",
            "Paul Appell",
            "32",
            "Lannexe",
            "Illkirch"
            "Mini R",
            "Cronenbourg",
            "Cristal Shop" 
        ]
    
        for question in preselected_questions:
            if st.button(question):
                st.session_state.selected_question = question

    # Affichage des messages de chat à partir de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.session_state["context_loaded"] = False
    
    # Entrée utilisateur
    question = st.chat_input("💬 Enter your question here...")

    # Utiliser la question pré-sélectionnée si elle existe
    if st.session_state.selected_question:
        question = "Give me the menu at " + st.session_state.selected_question
        st.session_state.selected_question = None
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        # Si le contexte n'est pas encore chargé
        if not st.session_state["context_loaded"]:
            with st.spinner("⏳ Loading..."):

                rag_chain = rag(question, llm)

                # Sauvegarder l'état dans la session
                st.session_state["context_loaded"] = True
                st.session_state["rag_chain"] = rag_chain

        # Si le contexte est chargé, répondre à la question
        if st.session_state["context_loaded"]:
            with st.spinner("⏳ Loading..."):
                rag_chain = st.session_state["rag_chain"]
                results = rag_chain.invoke({"input": question})
                answer = results["answer"]

                if answer.lower() == "no context.":
                    st.session_state["context_loaded"] = False
                    st.error("❌ Contexte non trouvé. Essayez de reformuler votre question.")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

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

# Lancer l'application Streamlit
if __name__ == "__main__":
    main()