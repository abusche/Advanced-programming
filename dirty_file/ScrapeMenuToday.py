from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


def ScrapeMenu(link):
    menuInfo = {"day": [], "lunch": [], "dinner": []}
    
    driver = webdriver.Firefox()
    driver.get(link)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    
    # Cookies button
    try:
        reject_all = driver.find_element(By.XPATH, "/html/body/div[6]/div/div[2]/div/div[2]/button[1]")
        reject_all.click()
    except Exception:
        try:
            reject_all = driver.find_element(By.ID, "tru_deselect_btn")
            reject_all.click()
        except Exception:
            pass
    
    # Find the right day
    date_elements = driver.find_elements(By.CSS_SELECTOR, 'time.menu_date_title')
    scraped_day = ' '.join(date_elements[0].text.split()[2:5])

    # Scrape meal
    meal_div = driver.find_elements(By.CSS_SELECTOR, 'div.meal')
    meals = []
    for meal in meal_div:
        if meal.text != "":
            meals.append(meal.text)
    
    if len(meals) == 2:
        lunch = meals[0].replace('\n', ' ')  # Ensure correct format for lunch
        dinner = meals[1].replace('\n', ' ')  # Ensure correct format for dinner
    elif len(meals) == 1:
        lunch = meals[0].replace('\n', ' ')
        dinner = ""
    else:
        lunch = ""
        dinner = ""
    
    # Do dataframe
    menuInfo["day"].append(scraped_day)
    menuInfo["lunch"].append(lunch)
    menuInfo["dinner"].append(dinner)

    driver.quit()
    
    return pd.DataFrame(menuInfo)
