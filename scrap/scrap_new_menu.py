# Import necessary functions and modules
from menu_functions import get_menu_bs_unique_all_day  # Custom function to retrieve menus for restaurants
import getpass  # Module to handle password input
import os  # Module for interacting with the operating system
from dotenv import load_dotenv  # Library to load environment variables from a .env file
import pandas as pd  # Library for data manipulation and analysis

# Importing specific classes and functions from langchain and langchain_core modules
from langchain_openai import ChatOpenAI  # To interact with OpenAI models
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)  # Handle chat message history
from langchain_core.runnables.history import RunnableWithMessageHistory  # Enable chat history integration in pipelines
from langchain_core.messages import HumanMessage  # Represent a human-sent message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Manage and format prompts

# List of restaurant names
restaurants_name = ["Pege", "Gallia", "Esplanade", "Paul appell", "32", "Lannexe", 
                     "Illkirch", "Mini r", "Cronenbourg", "Cristal shop"]

# Lists of common allergens in different languages
h_gluten = [...]  # List of gluten-containing keywords in multiple languages
h_shellfish = [...]  # Shellfish allergens
h_eggs = [...]  # Egg allergens
h_peanut = [...]  # Peanut allergens
h_soia = [...]  # Soy allergens
h_lactose = [...]  # Dairy allergens
h_nut = [...]  # Nut allergens
h_celery = [...]  # Celery allergens
h_mustard = [...]  # Mustard allergens
h_sesamo = [...]  # Sesame allergens
h_lupins = [...]  # Lupin allergens
h_Molluschi = [...]  # Mollusk allergens

# Links to restaurant menu pages
links = [
    'https://www.crous-strasbourg.fr/restaurant/cafeteria-le-pege-2/',
    'https://www.crous-strasbourg.fr/restaurant/resto-u-gallia-2/',
    ...
]

# Retrieve menus for lunch and dinner using the custom function
menus = get_menu_bs_unique_all_day(links, restaurants_name, "lunch") + get_menu_bs_unique_all_day(links, restaurants_name, "diner")

# List of keywords to ignore in menus
error = [...]  

# Load environment variables and set API key for OpenAI
path = "C:/Users/busch/.../environment/"
load_dotenv(f"{path}.env")  # Load variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set API key for OpenAI

# Define a function to get ingredients for a dish using an LLM
def assist_culinaire(plat):
    """
    Retrieves a list of ingredients for a given dish using an LLM.

    Args:
        plat (str): Name of the dish in French.

    Returns:
        str: Ingredients list with quantities.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo")  # Initialize the OpenAI model

    # Define the prompt template for the assistant
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Tu es un assistant culinaire, je te donne des noms de recette et tu dois me donner la liste des ingrédients avec les quantités entre parenthèse. Voici un exemple : 'Tarte Flambée' : 'Pâte (250 g), crème fraîche (200 g), fromage blanc (200 g), oignons (2 moyens), lardons fumés (150 g), sel (une pincée), poivre'",),
         MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create a chain of operations
    chain = prompt | model

    # Set up message history storage
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Combine chain with history
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    
    # Configure session
    config = {"configurable": {"session_id": "session1"}}
    
    # Get the response from the model
    response = with_message_history.invoke(
            [HumanMessage(content=plat)],
            config=config,
        )
    
    return response.content  # Return the model's response

# Define a translation function using Hugging Face models
def translator(language_from, language_to, text):
    """
    Translates text from one language to another using specific Hugging Face translation models.

    Args:
        language_from (str): Source language code (e.g., "fr" for French, "it" for Italian).
        language_to (str): Target language code (e.g., "en" for English, "fr" for French).
        text (str): The text to translate.

    Returns:
        str: The translated text.
    """
    if language_from == "fr" and language_to == "en":
        api_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    ...
    return api_translator(text)[0]['translation_text']

# Extract dishes from the menus, excluding unwanted keywords
new_dishes = []
for menu in menus:
    menu = menu.split("\n")
    for m in menu:
        t = 0
        if m != "":
            if m[0] == "-":
                for e in error:
                    if m.replace("- ", "") == e:
                        t += 1
                if t == 0:
                    new_dishes.append(m.replace("- ", "").replace("ou ", ""))

# Remove duplicates
new_dishes = list(set(new_dishes))

# Get ingredients for each dish using the assistant
new_ingredient = []
for plat in list(set(new_dishes)):
    new_ingredient.append(assist_culinaire(plat))

# Create a DataFrame with dishes and ingredients
df = pd.DataFrame({
    'Recipes (French)': list(set(new_dishes)),
    'Ingredients (French)': new_ingredient,
})

# Save the DataFrame to a CSV file
df.to_csv('new_menu.csv', index=False)
