import random
import functools
import time
import requests
from bs4 import BeautifulSoup

@functools.cache
def page(urlpage): 
    """
    Récupération du HTML d'un site internet via Beautifulsoup
    """
    user_agent = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'}
    time.sleep(1 + np.random.rand()/5) # Retarder le téléchargement pour pas se faire ban
    res = requests.get(urlpage, headers = user_agent)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup