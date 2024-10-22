from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import os

hf_token = "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"
custom_cache_dir = "/home/peltouz/Documents/pretrain"

os.environ['HF_HOME'] = custom_cache_dir  # Hugging Face home directory for all HF operations
os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir  # Transformers-specific cache directory
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir  # Datasets-specific cache directory
os.environ['HF_METRICS_CACHE'] = custom_cache_dir  # Metrics-specific cache directory
os.environ['HF_TOKEN'] = hf_token  # Hugging Face API token

def HF_model(model, message):
    if model == "hours":
        m = AutoModelForTokenClassification.from_pretrained("DAMO-NLP-SG/roberta-time_identification")
        tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/roberta-time_identification")
        nlp = pipeline("ner", model=m, tokenizer=tokenizer, aggregation_strategy="simple")
        ner_results = nlp(message)
    if model == "loc":
        m = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        nlp = pipeline("ner", model=m, tokenizer=tokenizer, aggregation_strategy="simple")
        ner_results = nlp(message)
    
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

def get_link(message):
    resto = []
    ent = HF_model("loc", message)
    for i in range(len(ent)):
        entity_group = ent[i]['entity_group']
        if entity_group == "ORG" or entity_group == "LOC":
            resto.append(ent[i]['word'])
    link = resto_link(resto)
    return link






