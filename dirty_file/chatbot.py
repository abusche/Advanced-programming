import getpass
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory,)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment and API Key
load_dotenv("C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced programming/projet/environment/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def chatbot():

    # Connexion to the LLM
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # Initialize the promt
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an assistant.",),
         MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create the chain
    chain = prompt | model
    
    # Load historical environment
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Integrate the chain with the history
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    
    # Initialize a session ( an environment to save history )
    config = {"configurable": {"session_id": "session1"}}
    
    # Initialize the question
    question = ""
    
    # Interact with the LLM
    while question != "Exit":
        question = input() # Save the question of the user
        
        # Write an answer from the LLM
        response = with_message_history.invoke(
            [HumanMessage(content=question)],
            config=config,
        )

        # Print the answer
        print(response.content)