import torch
import os
import time
import functools
import requests
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime

from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from google.cloud import translate_v2 as translate

# Load environment variables from a .env file
path = "C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced Programming/projet/environment/"
load_dotenv(f"{path}.env")
hf_token = os.getenv("HUGGING_FACE_KEY")
custom_cache_dir = "/home/peltouz/Documents/pretrain"

# Configure environment variables for Hugging Face operations
os.environ['HF_HOME'] = custom_cache_dir  # Hugging Face home directory for all HF operations
os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir  # Transformers-specific cache directory
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir  # Datasets-specific cache directory
os.environ['HF_METRICS_CACHE'] = custom_cache_dir  # Metrics-specific cache directory
os.environ['HF_TOKEN'] = hf_token  # Hugging Face API token

def HF_model(model, question):
    """
    Loads a specific Hugging Face model for named entity recognition (NER) based on the input model type.
    Processes a question string to extract named entities.
    """
    if model == "hours":
        m = AutoModelForTokenClassification.from_pretrained("DAMO-NLP-SG/roberta-time_identification")
        tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/roberta-time_identification")
        nlp = pipeline("ner", model=m, tokenizer=tokenizer, aggregation_strategy="simple")
        ner_results = nlp(question)
    if model == "loc":
        m = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        nlp = pipeline("ner", model=m, tokenizer=tokenizer, aggregation_strategy="simple")
        ner_results = nlp(question)
    
    return ner_results

def resto_link(resto):
    """
    Matches restaurant names from a list against predefined vectors using cosine similarity.
    Generates URLs for restaurant information.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vec_resto = ["cafeteria le pege", "resto u gallia", "resto u esplanade", "resto u paul appell", "le 32", "lannexe", 
                 "resto u illkirch", "cafeteria mini r", "resto u cronenbourg", "le cristal shop ru esplanade"]
    link = []
    for i in range(len(resto)):
        sim_score = []
        for j in range(len(vec_resto)):
            embeddings1 = model.encode([resto[i]], convert_to_tensor=True)
            embeddings2 = model.encode([vec_resto[j]], convert_to_tensor=True)
            sim_score.append(torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item())
        if max(sim_score) < 0.5:
            link.append("Error")
        else:
            k = sim_score.index(max(sim_score))
            if vec_resto[k] == "resto u paul appell":
                link.append("https://www.crous-strasbourg.fr/restaurant/" + vec_resto[k].replace(" ", "-") + "/")
            else:
                link.append("https://www.crous-strasbourg.fr/restaurant/" + vec_resto[k].replace(" ", "-") + "-2/")
    
    return link

def get_link(question):
    """
    Extracts restaurant-related entities from a question string and generates corresponding links using the functions resto_link and HF_model.
    """
    restaurants_name = []
    ent = HF_model("loc", question)
    for i in range(len(ent)):
        entity_group = ent[i]['entity_group']
        if entity_group == "ORG" or entity_group == "LOC":
            restaurants_name.append(ent[i]['word'])
    if restaurants_name == []:
        restaurants_name = ["pege", "gallia", "esplanade", "paul appell", "32", "lannexe", 
                 "illkirch", "cronenbourg", "cristal shop"]
    link = resto_link(restaurants_name)
    return link, restaurants_name

def get_hours_meal(question):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    repas = ["diner", "lunch"]
    word = question.split(" ")
    sim_score = []
    r = []
    for i in range(len(question.split(" "))):
        for j in range(len(repas)):
            embeddings1 = model.encode([word[i]], convert_to_tensor=True)
            embeddings2 = model.encode([repas[j]], convert_to_tensor=True)
            sim_score.append(torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item())
            r.append(repas[j])
    max_ = max(sim_score)
    if max_ < 0.6:
        current_time = datetime.now().strftime('%H:%M:%S')
        if '14:00:00' < current_time and '21:00:00' > current_time:
            r_ = "diner"
        else:
            r_ = "lunch"
    else:
        r_ = r[sim_score.index(max(sim_score))]
    
    return r_

def get_day(question):
    t = HF_model("hours", question)
    if t == []:
        day = 'today'
    else:
        day = t[0]["word"].replace(" ", "")
    return day

@functools.cache
def page(urlpage): 
    """
    Récupération du HTML d'un site internet via Beautifulsoup
    """
    user_agent = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'}
    time.sleep(0 + np.random.rand()/5)
    res = requests.get(urlpage, headers = user_agent)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def get_menu_bs_unique_today(links, restaurants_name, meal):
    all_menus = []
    for j in range(len(links)):
        soup = page(links[j])
        raw_menus = soup.find_all("ul", class_="meal_foodies")
        if soup.find_all("time", class_="menu_date_title"):
            date = soup.find_all("time", class_="menu_date_title")[0].text
            if meal == "lunch":
                m = 0
            else:
                m = 1
            if len(raw_menus) < m + 1:
                all_menus.append(f"{date} à {restaurants_name[j]} pour le {meal} \n\n menu non available")
            else:
                raw_menu = raw_menus[m].find_all("li")
                items = []
                for i in range(len(raw_menu)):
                    if raw_menu[i].find("ul") != None:
                        items.append(raw_menu[i].text.replace(raw_menu[i].find("ul").text, ""))
                    else:
                        if raw_menu[i].text == "-":
                            items.append(" ")
                        else:
                            items.append("- " + raw_menu[i].text)
        
                menu = f"{date} à {restaurants_name[j]} pour le {meal} \n\n"
                for i in range(len(items)):
                    if i != len(items)-1:
                        if items[i+1][0] != "-" and items[i+1][0] != " ":
                            menu += items[i] + "\n\n"
                        else:
                            menu += items[i] + "\n"
                    else:
                        menu += items[i] + "\n"
    
                origine_index = menu.lower().find("origine")
                if origine_index != -1:
                    menu = menu[:origine_index].strip()
                    
                all_menus.append(menu)

    return all_menus

def get_menu_bs_unique_all_day(links, restaurants_name, meal):
    all_menus = []
    for j in range(len(links)):
        soup = page(links[j])
        raw_menus = soup.find_all("ul", class_="meal_foodies")
        k=0
        if soup.find_all("time", class_="menu_date_title"):
            date = soup.find_all("time", class_="menu_date_title")[k].text
            for i in range(len(raw_menus)):
                if i % 2 == 0 and i != 0:
                    k+=1
                    date = soup.find_all("time", class_="menu_date_title")[k].text
                if soup.find_all("div", class_="meal_title")[i].text == "Déjeuner":
                    meal_ = "lunch"
                else :
                    meal_ = "diner"
                if meal_ == meal:
                    raw_menu = raw_menus[i].find_all("li")

                    items = []
                    for i in range(len(raw_menu)):
                        if raw_menu[i].find("ul") != None:
                            items.append(raw_menu[i].text.replace(raw_menu[i].find("ul").text, ""))
                        else:
                            if raw_menu[i].text == "-":
                                items.append(" ")
                            else:
                                items.append("- " + raw_menu[i].text)

                    menu = f"{date} à {restaurants_name[j]} pour le {meal} \n\n"
                    for i in range(len(items)):
                        if i != len(items)-1:
                            if items[i+1][0] != "-" and items[i+1][0] != " ":
                                menu += items[i] + "\n\n"
                            else:
                                menu += items[i] + "\n"
                        else:
                            menu += items[i] + "\n"

                    origine_index = menu.lower().find("origine")
                    if origine_index != -1:
                        menu = menu[:origine_index].strip()

                    all_menus.append(menu)
    return all_menus  

def add_allergens(menu, similarity_threshold=0.5):
    """
    Adds allergens to dishes in a given menu and includes a summary of allergens at the end of each menu.

    Args:
        menu (str): String containing the menu items
        similarity_threshold (float): Similarity threshold to consider a dish similar

    Returns:
        str: Menu modified with allergens summary appended after each menu section
    """
    # Load model and data
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("db/full_translated_data.csv")
    recipe_names = df['Recipes (French)'].tolist()

    # Calculate embeddings of recipes
    recipe_embeddings = model.encode(recipe_names)

    def find_allergens(dishe):
        # Find the most similar dish
        embeddings = model.encode([dishe])
        similarity_scores = cosine_similarity(embeddings, recipe_embeddings)[0]
        most_similar_index = similarity_scores.argmax()
        max_similarity_score = similarity_scores[most_similar_index]

        if max_similarity_score < similarity_threshold:
            return None

        # Recover allergens from similar dishes
        most_similar_recipe = df.iloc[most_similar_index]
        allergies = {
            "Gluten": most_similar_recipe['Gluten'], "Eggs": most_similar_recipe['Eggs'],
            "Peanut": most_similar_recipe['Peanut'], "Lactose": most_similar_recipe['Lactose'],
            "Soy": most_similar_recipe['Soy'], "Nut": most_similar_recipe['Nut'],
            "Celery": most_similar_recipe['Celery'], "Mustard": most_similar_recipe['Mustard'],
            "Sesame": most_similar_recipe['Sesame'], "Lupins": most_similar_recipe['Lupins'],
            "Molluscs": most_similar_recipe['Molluscs']
        }
        return ", ".join([k for k, v in allergies.items() if v == 1])

    def process_menu_section(section):
        allergens_summary = []

        def add_allergen(match):
            dishe = match.group(1).strip()
            allergenes = find_allergens(dishe)
            if allergenes:
                allergens_summary.append(f"- {dishe}: {allergenes}")
            return f"- {dishe}"

        # Add allergens to menu items in the section
        pattern = r"- ([^\n]+)"
        section_with_dishes = re.sub(pattern, add_allergen, section)

        # Append allergens summary to the section
        summary_text = "Allergènes :\n" + "\n".join(allergens_summary) + "\n\n"
        return section_with_dishes + summary_text

    # Split the menu into sections based on "Menu du"
    menu_sections = re.split(r"(?=Menu du)", menu)
    processed_sections = [process_menu_section(section) for section in menu_sections if section.strip()]

    return "\n".join(processed_sections)


def get_menu(question):
    """
    Retrieves and cleans the menu for a specific restaurant mentioned in the question.
    """
    error = False
    links, restaurants_name = get_link(question)
    for link in links:
        if link == "Error":
            dirty_menu = "Restaurant not find"
            error = True
            break
            break
    meal = get_hours_meal(question)
    if error == False:  
        if get_day(question) == 'today':
            scrap_menus = get_menu_bs_unique_today(links, restaurants_name, meal)
        else: 
            scrap_menus = get_menu_bs_unique_all_day(links, restaurants_name, meal)
        dirty_menu = ""
        for sm in scrap_menus:
            dirty_menu += "\n \n" + sm
        cleaned_menu = add_allergens(dirty_menu.replace(",", "\n-"))
        
    return cleaned_menu, restaurants_name

def change_context(question, restaurant_name):

    vec_resto = ["Pege", "Gallia", "Esplanade", "Paul appell", "32", "Lannexe", 
                     "Illkirch", "Mini r", "Cronenbourg", "Cristal shop"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    mod = HF_model("loc", question)
    if mod:
        loc = []
        for m in mod:
            if m["score"] > 0.6:
                sim_score = []
                for j in range(len(vec_resto)):
                    embeddings1 = model.encode(m["word"], convert_to_tensor=True)
                    embeddings2 = model.encode([vec_resto[j]], convert_to_tensor=True)
                    sim_score.append(torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item())
                if max(sim_score) < 0.5:
                    loc = None
                else:
                    k = sim_score.index(max(sim_score))
                    loc.append(vec_resto[k])
            else:
                loc = None
    else:
        loc = None

    if loc != None and loc != restaurant_name:
        menu, restaurant_name = get_menu(question)
        context = False
    else:
        context = True

    return context

def rag(question, llm, language):
    
    menu, restaurant_name = get_menu(question)
    menus = translate_text(menu, language)
    
    # Traitement brut du texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(menus)

    # Transformer le texte en objets Document
    documents = [Document(page_content=split) for split in splits]

    # Création du VectorStore
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    # Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks about the menu of university restaurant. "
        "If you don't specify a specific dish, you should always give today's meal or the nearest one. "
        "If I ask you a question about a restaurant and you don't have the menu of this restaurant, say: 'No context'."
        "If you are unable to provide specific menu information for a restaurant for a date, say : 'No context'."
        "Use the following pieces of retrieved context to answer the question. "
        "Use bullet points when it's necessary. "
        "The menu will be sent to you in markdown text format. After the menu, you find the allergies part. Never show the allergen part in your answer. "
        "You're also an allergy specialist. Allergies are explained in brackets. If there are allergens, you must write a message at the end, for example: ‘Warning! Allergens such as lactose or hazelnuts may be present in this menu'. "
        "When I ask you about allergies, always answer about the allergies of the menu that we talked about before, never all the allergies of all menus."
    )

    # Ajouter la variable context dans le modèle
    prompt_template = PromptTemplate(
        input_variables=[system_prompt, "input"],
        template="{context}\n\nHuman: {input}"
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def translate_text(text, target_language):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "db/translate-project-447409-d76e1b17f268.json"
    
    translate_client = translate.Client()  # Usa il file di credenziali
    result = translate_client.translate(text, target_language=target_language)
    return result["translatedText"]


def detect_language(text):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "db/translate-project-447409-d76e1b17f268.json"

    translate_client = translate.Client()
    result = translate_client.detect_language(text)['language']
    return result