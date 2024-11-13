from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import pandas as pd

def ScrapeMenu(link):
    
    menuInfo = {"day": [], "lunch": [], "dinner": []}

    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)
    
    driver.get(link)
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    try:
        reject_all = driver.find_element(By.XPATH, "/html/body/div[6]/div/div[2]/div/div[2]/button[1]")
        reject_all.click()
    except Exception:
        try:
            reject_all = driver.find_element(By.ID, "tru_deselect_btn")
            reject_all.click()
        except Exception:
            pass

    date_elements = driver.find_elements(By.CSS_SELECTOR, 'time.menu_date_title')
    scraped_day = ' '.join(date_elements[0].text.split()[2:5])

    meal_div = driver.find_elements(By.CSS_SELECTOR, 'div.meal')

    meals = []
    for meal in meal_div:
        if meal.text != "":
            meal_text = meal.text.replace('\n', ' ')

            list_items = meal.find_elements(By.CSS_SELECTOR, 'li')
            formatted_meal = ""
            for item in list_items:
                pseudo_before = driver.execute_script(
                    "return window.getComputedStyle(arguments[0], '::before').getPropertyValue('content');", item
                )
                pseudo_before = pseudo_before.replace('"', '')
                formatted_meal += pseudo_before + item.text + '\n'
            
            meals.append(formatted_meal.strip())

    if len(meals) == 2:
        lunch = meals[0]
        dinner = meals[1]
    elif len(meals) == 1:
        lunch = meals[0]
        dinner = ""
    else:
        lunch = ""
        dinner = ""
    
    menuInfo["day"].append(scraped_day)
    menuInfo["lunch"].append(lunch)
    menuInfo["dinner"].append(dinner)
    
    driver.quit()
    
    return pd.DataFrame(menuInfo)