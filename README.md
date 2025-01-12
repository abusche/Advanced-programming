# Project Overview :

"AI’m Hungry" is an interactive chatbot application designed to assist users in discovering daily menus at the Crous restaurants of the University of Strasbourg. This project leverages web scraping, natural language processing, and machine learning to provide relevant menu information, including allergens and meal schedules, through an intuitive interface. The application supports multilingual queries and offers an enriched user experience via Dash, Streamlit, or WhatsApp.

# Project Objectives :
- Menu Retrieval: Scrape and store menu information from Crous restaurant websites
- Interactive Chatbot: Enable users to query menus using natural language
- Advanced Features: Include allergen alerts, meal timing relevance (lunch/dinner), and multilingual support

# File Structure and Functionality :

1) ```requirements.txt``` : Lists the required Python libraries and dependencies, including:

- NLP frameworks: langchain, transformers, sentence-transformers
- Web scraping: beautifulsoup4, requests
- Other utilities: pandas, streamlit, numpy, google-cloud-translate

2) ```menu_functions.py``` : Core logic for menu retrieval and processing:

- ```HF_model()```: Utilizes Hugging Face models for named entity recognition (NER)
- ```get_link()```: Extracts restaurant-related links based on user queries
- ```get_menu()```: Retrieves and cleans menus for specific restaurants
- ```add_allergens()```: Identifies allergens in menu items
- ```translate_text()``` and ```detect_language()```: Handle multilingual support via Google Cloud Translate

3) ```menu.py``` : Streamlit application file:
- Sets up the chatbot UI with a custom theme and question-answering interface
- Manages the session states and connects the chatbot to menu functions
- Provides preselected questions for quick access to specific restaurants

4) ```Application.ipynb``` : A Jupyter Notebook file that likely outlines initial prototyping or testing of chatbot features, showcasing the integration of APIs and logic

5) ```Giallozafferano.ipynb```: A supplementary notebook for experiments, possibly for additional functionalities like advanced recipe handling or exploratory tasks

6) ```scrap_new_menu.py```: Script for advanced menu scraping and ingredient extraction:
- Uses get_menu_bs_unique_all_day() for fetching full-day menus
- Extracts unique dishes from menus and queries an AI assistant for ingredient details
- Translates and stores menus in a structured CSV format

# Key Freatures :

1) Web Scraping:
    - Retrieves menu data using BeautifulSoup and predefined restaurant links
    - Cleans and structures data for user-friendly access

2) AI-Powered Chatbot:
    - Employs LangChain for document-based question answering.
    - Supports conversational responses enhanced with context and natural language understanding

3) Allergen Detection:
    - Matches dishes with a predefined allergen database to alert users about potential allergens

4) Multilingual Support:
    - Detects user language and translates menus using Google Translate and Hugging Face models

5) User-Friendly Interface:
    - Streamlit application with preselected questions and responsive chat elements

# How to Run :

1) Install Dependencies: ```pip install -r requirements.txt```

2) Set Up Environment Variables: Add necessary API keys and configurations to a ```.env``` file.

3) Launch the Application: ```streamlit run menu.py```

4) Interact: Ask questions about menus or allergens, or select pre-suggested queries.

# Future Enhancements :

- Integration with WhatsApp for mobile-friendly interaction.
- Advanced allergen prediction using external datasets.
- Dynamic support for new restaurants and menu updates.
