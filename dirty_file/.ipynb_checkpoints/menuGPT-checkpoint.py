import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain

# Charger l'environnement
load_dotenv("C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced programming/projet/environment/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Interface utilisateur Streamlit
st.title("Crous Chatbot")

##################
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I help you today?"}
    ]
##################
    
if 'context' not in st.session_state:
    st.session_state.context = False
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

question = st.text_input("Ask me a question...")

if question:
    if question == "Exit":
        st.write("Merci d'avoir utilisé l'application !")
    else:
        if st.session_state.context == False:
            # Scraping du menu du restaurant
            menu_to_pdf(question)

            # Connection au contexte
            file_path = "C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced programming/projet/dirty_file/menu.pdf"
            loader = PyPDFLoader(file_path)

            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = InMemoryVectorStore.from_documents(
                documents=splits, embedding=OpenAIEmbeddings()
            )

            retriever = vectorstore.as_retriever()

            # Prompt
            system_prompt = (
                "You are an assistant for question-answering tasks about the menu of a specific restaurant. "
                "I give you a menu, just answer my question, and don't take attention to the name that I specified. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            st.session_state.context = True

        if st.session_state.rag_chain:
            # Write an answer from the LLM
            results = st.session_state.rag_chain.invoke({"input": question})

            # Print the answer
            st.write(results["answer"])

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