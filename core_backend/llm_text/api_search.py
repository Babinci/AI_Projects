from dotenv import load_dotenv
import os
import requests
from .models import CypherArenaPerplexityDeepResearch

# Configuration
SEARCH_TYPES = {
    "general_news": "general_news_search.md",
    "polish_showbiznes": "polish_showbiznes_search.md",
}

# Load environment variables
load_dotenv(dotenv_path="/home/wojtek/AI_Projects/credentials/.env")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY_WALKOWO")
BASE_PATH = "/home/wojtek/AI_Projects/core/llm_text/prompts/perplexity_deep_research"

def search_internet(start_date, end_date, search_model="sonar-pro", name="general_news"):
    """Search the internet for news in a given date range and save results to database"""
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    print(f"Searching: {name} from {start_date_str} to {end_date_str} using {search_model}")
    
    # Get prompt template
    template_file = SEARCH_TYPES.get(name)
    if not template_file:
        raise ValueError(f"Unknown search type: {name}")
    
    with open(f"{BASE_PATH}/{template_file}", "r", encoding="utf-8") as file:
        user_prompt = file.read().replace("{{start_date_str}}", start_date_str).replace("{{end_date_str}}", end_date_str)
    
    # Create system prompt with date placeholders replaced
    system_prompt = f"Jesteś specjalistycznym asystentem przeprowadzającym przegląd wydarzeń z okresu od {start_date_str} do {end_date_str}."
    
    # Make API request
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"},
            json={
                "model": search_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Save to database
        search_record = CypherArenaPerplexityDeepResearch.objects.create(
            start_date=start_date,
            end_date=end_date,
            data_response=response_data,
            search_type=search_model,
            news_source=name
        )
        
        print(f"Search record created with ID: {search_record.id}")
        return response_data
        
    except Exception as e:
        print(f"Error during search: {e}")
        raise