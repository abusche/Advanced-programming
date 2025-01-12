from menu_functions import get_menu_bs_unique_all_day  # Custom function to retrieve menus for restaurants
import getpass 
import os 
from dotenv import load_dotenv  
import pandas as pd 

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
h_gluten = ["grano", "segale", "orzo", "avena", "farro", "kamut", "wheat", "rye", "barley", "oats", "spelt", "kamut", "pâtes", "pasta", "pâte", "blé", "seigle", "orge", "avoine", "épeautre", "kamut", 'pain', "bread", 
            "dolore", 'farine', "farina", "flour", 'céréales', 'céréale', "cereali", "cereale", "cereals", "cereal", 'pâtes', 'pâte', "pasta", "pastas", 'cracker', 'crackers', 'sandwich', 'sandwichs', 'dessert', 'desserts', 
            "dolce", "dolci"]
h_shellfish = ["gamberi", "gamberetti", "scampi", "granchi", "aragoste", "astici", "shrimps", "prawns", "langoustines", "crabs", "lobsters", "spiny lobsters", "crevettes", "langoustine", "crabe", "homard", "langouste"]
h_eggs = ["uova", "eggs", "œufs", "oeuf", "oeufs", "uova", "uovo", "egg", "eggs"]
h_peanut = ["arachide", "peanut", "cacahuète"]
h_soia = ["soia", "soya", "soja"]
h_lactose = ["latte", "formaggio", "burro", "panna", "yogurt", "crema di latte", "ricotta", "mozzarella", "parmigiano", "gorgonzola","milk", "cheese", "butter", "cream", "yogurt", "heavy cream", "ricotta", "mozzarella", 
             "parmesan","chèvre", "chèvre", "lait", "fromage", "beurre", "crème", "yaourt", "crème épaisse", "ricotta", "mozzarella", "parmesan", "lait", "crème", "crèmes", 'yaourt', 'yaourts', 'fromage', 'fromages', 
             'beurre', 'mozzarella', "milk", "cream", "yoghurt", "cheese", "butter", "latte", "panna", "yogurt", "formaggio", "burro"]
h_nut = ["noci", "nocciole", "mandorle", "pistacchi", "anacardi", "pinoli", "arachidi", "castagne", "pecan", "noci del Brasile", "noci macadamia", "walnuts", "hazelnuts", "almonds", "pistachios", "cashews", "pine nuts", 
         "peanuts", "chestnuts", "pecans", "brazil nuts", "macadamia nuts","noix", "noisette", "amande", "pistaches", "pignons", "arachides", "châtaignes", "pecan"]
h_celery = ["sedano", "celery", "céleri"]
h_mustard = ['moutarde', "mustard", "senape", "mostarda"]
h_sesamo =["sesamo", "sesame", "sésame"]
h_lupins = ["lupini", "lupins", "lupin"]
h_Molluschi = ["canestrello", "cannolicchio", "capasanta", "dattero di mare", "fasolaro","garagolo", "lumachino", "cozza", "murice", "ostrica", "patella", "tartufo di mare", "tellina", "vongola", "scallop", "razor clam", 
               "queen scallop", "date mussel", "smooth clam", "whelk", "periwinkle", "mussel", "murex", "oyster", "limpet", "sea truffle", "bean clam", "clam", "pétoncle", "couteau", "coquille Saint-Jacques", 
               "dattier de mer", "praire",  "buccin", "bigorneau", "moule", "murex", "huître", "patelle", "truffe de mer", "telline", "palourde"]

# Links to restaurant menu pages
links = ['https://www.crous-strasbourg.fr/restaurant/cafeteria-le-pege-2/',
         'https://www.crous-strasbourg.fr/restaurant/resto-u-gallia-2/',
         'https://www.crous-strasbourg.fr/restaurant/resto-u-esplanade-2/',
         'https://www.crous-strasbourg.fr/restaurant/resto-u-paul-appell/',
         'https://www.crous-strasbourg.fr/restaurant/le-32-2/',
         'https://www.crous-strasbourg.fr/restaurant/lannexe-2/',
         'https://www.crous-strasbourg.fr/restaurant/resto-u-illkirch-2/',
         'https://www.crous-strasbourg.fr/restaurant/cafeteria-mini-r-2/',
         'https://www.crous-strasbourg.fr/restaurant/resto-u-cronenbourg-2/',
         'https://www.crous-strasbourg.fr/restaurant/le-cristal-shop-ru-esplanade-2/']

# Retrieve menus for lunch and dinner using the custom function
menus = get_menu_bs_unique_all_day(links, restaurants_name, "lunch") + get_menu_bs_unique_all_day(links, restaurants_name, "diner")

# List of keywords to ignore in menus
error = ['Café & Viennoiseries','7h30 -10h30','MENU 100% VEGETARIEN', 'TOUS LES MERCREDIS MIDIS MENU 100% VEGETARIEN', 'JOURNEE VERTE', '1er étage :', 'ou', 'Petits-déjeuners à partir de 7h45',
         'Sandwichs à partir de 9h45', 'Formules à emporter', 'Snacking chaud', 'Desserts variés', 'Boissons chaudes et froides', 'Chaine étu : Menu du jour', 'menu non communiqué',
         '4,50 € HT/4,95 € TTC', '+ prix de la Box 5,70 € TTC', 'Formule avec frites', 'Formule Be Fit', 'Sandwich, eau plate ou gazeuse, fruit ou yaourt']

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
    if language_from == "it" and language_to == "fr":
        api_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-it-fr")
    if language_from == "it" and language_to == "en":
        api_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")
    if language_from == "fr" and language_to == "it":
      api_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-it")
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
