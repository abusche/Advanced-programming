from ScrapeMenuToday import ScrapeMenu
from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv

load_dotenv("C:/Users/busch/OneDrive/Documents/Fac/M2/UE1 - Advanced programming and data visualization/Advanced programming/projet/environment/.env")
hf_token = os.getenv("HUGGING_FACE_KEY")
custom_cache_dir = "/home/peltouz/Documents/pretrain"

os.environ['HF_HOME'] = custom_cache_dir  # Hugging Face home directory for all HF operations
os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir  # Transformers-specific cache directory
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir  # Datasets-specific cache directory
os.environ['HF_METRICS_CACHE'] = custom_cache_dir  # Metrics-specific cache directory
os.environ['HF_TOKEN'] = hf_token  # Hugging Face API token

def HF_model(model, question):
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
        k = sim_score.index(max(sim_score))
        link.append("https://www.crous-strasbourg.fr/restaurant/" + vec_resto[k].replace(" ", "-") + "-2/")
    
    return link

def get_link(question):
    resto = []
    ent = HF_model("loc", question)
    for i in range(len(ent)):
        entity_group = ent[i]['entity_group']
        if entity_group == "ORG" or entity_group == "LOC":
            resto.append(ent[i]['word'])
    link = resto_link(resto)
    return link, resto[0]

def translator(language_from, language_to, text):
    if language_from == "fr" and language_to == "en":
        api_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        translation = api_translator(text)[0]['translation_text']
    return translation

def clean_menu(dirty_menu):
    menu = ""
    el_menu_cleaned_str = ""
    dirty_menu_2 = []
    elements = dirty_menu.split("none")
    for k in range(1,len(elements)):
        if k != 0: # and k != len(elements)-1:
            el_menu_cleaned_str += "- " + translator("fr", "en", elements[k].replace("- ", "").split("\n")[0]) + "\n\n"
            el_menu_dirty = elements[k].replace("- ", "").split("\n")[1:]
            el_menu_dirty_2 = [item.strip() for item in el_menu_dirty if item.strip()]
            el_menu_cleaned = []
            for item in el_menu_dirty_2:
                item_trad = translator("fr", "en", item)
                if item_trad == '-' or item_trad not in el_menu_cleaned:
                    el_menu_cleaned.append(item_trad)
            valid = False
            while valid == False:
                if el_menu_cleaned[len(el_menu_cleaned)-1] == "-":
                    el_menu_cleaned = el_menu_cleaned[:-1]
                else:
                    valid = True
            for i in range(len(el_menu_cleaned)):
                if el_menu_cleaned[i] == "-":
                    el_menu_cleaned[i] = ""
            for mu in el_menu_cleaned:
                el_menu_cleaned_str += mu + "\n"
            el_menu_cleaned_str += "\n"
    return el_menu_cleaned_str

def get_menu(question):
    links, resto = get_link(question)
    dirty_menu = ""
    for link in links:
        dirty_menu += "\n \n" + ScrapeMenu(link)["lunch"][0]
    menu = "Menu " + resto + " : \n \n" + clean_menu(dirty_menu)
    
    return menu



