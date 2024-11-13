from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

def ScrapeMenu(link, day):
    options = Options() 
    options.add_argument("-headless") 
    driver = webdriver.Firefox(options=options)
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
    date_elements = driver.find_elements(By.CSS_SELECTOR, "time.menu_date_title")
    scraped_day = " ".join(date_elements[0].text.split()[2:5])   
    scraped_day

    while day != scraped_day:
        for i in range(len(date_elements)):
            try:
                next_button = driver.find_element(By.XPATH, f"/html/body/article/div/section[1]/div/div/div[{i+1}]/div[1]/div[2]")
                next_button.click()
                break
            except Exception:
                pass
        
        for i in range(len(date_elements)):        
            date_elements = driver.find_elements(By.CSS_SELECTOR, "time.menu_date_title")
            a = " ".join(date_elements[i].text.split()[2:5])
            if a != "":
                scraped_day = " ".join(date_elements[i].text.split()[2:5])

    # Scrape meal
    meal_div = driver.find_elements(By.CSS_SELECTOR, 'div.meal')
    meals = []
    for meal in meal_div:
        if meal.text != "":
            text = meal.text
            try:
                text = text.split("Plat du jour", 1)[1].strip()
            except Exception:
                pass
            try:
                text = text.split("Origine", 1)[0].strip()
            except Exception:
                pass
            meals.append(text)
    
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
    text = f"Lunch:\n{lunch}\nDinner:{dinner}"

    driver.quit()
    
    return text

print(ScrapeMenu("https://www.crous-strasbourg.fr/restaurant/resto-u-gallia-2/","lundi 4 novembre"))