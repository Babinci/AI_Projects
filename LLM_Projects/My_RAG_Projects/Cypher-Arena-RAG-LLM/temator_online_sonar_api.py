import json
import requests
import os
import re
import random
from datetime import datetime, timedelta
from json.decoder import JSONDecodeError
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import time

class TopicGeneratorConfig:
    def __init__(self, env_path: str, json_path: str, output_path: str):
        load_dotenv(env_path)
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.json_path = json_path
        self.output_path = output_path
        self.max_retries = 3
        self.retry_delay = 5

        # Search domains grouped by category for better sampling
        self.search_domains = {
            'news': ['fakt.pl', 'se.pl', 'natemat.pl'],
            'entertainment': ['pudelek.pl'],
            'hiphop': ['glamrap.pl', 'newonce.net', 'popkiller.pl', 'cgm.pl', 'noizz.pl', 
                      'rapnews.pl', 'hip-hop.pl'],
            'sports': ['sport.pl'],
            'lifestyle': ['vogue.pl']
        }

class TopicGenerator:
    def __init__(self, config: TopicGeneratorConfig):
        self.config = config
        self.example_topics = self.load_example_topics()
        self.actual_topics = []  # Start fresh with empty list

    def load_example_topics(self) -> List[str]:
        """Load example topics from JSON file"""
        try:
            with open(self.config.json_path, "r", encoding='utf-8') as file:
                return json.load(file)
        except (FileNotFoundError, JSONDecodeError) as e:
            print(f"Error loading example topics: {e}")
            return []

    @staticmethod
    def get_date_range(days_delta: int) -> Tuple[str, str]:
        """Generate start and end dates for the API query"""
        current = datetime.now()
        end_date = current - timedelta(days=days_delta)
        start_date = end_date - timedelta(days=2)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def sample_search_domains(self, n: int = 3) -> List[str]:
        """Sample domains ensuring diversity across categories"""
        selected_domains = []
        categories = list(self.config.search_domains.keys())
        random.shuffle(categories)
        
        for category in categories[:n]:
            domain_list = self.config.search_domains[category]
            selected_domains.append(random.choice(domain_list))
        
        return selected_domains

    def create_api_payload(self, start_date: str, end_date: str, example_topics: List[str]) -> Dict[str, Any]:
        """Create the API request payload with system and user prompts"""
        system_prompt = f"""Jesteś AI-generatorem tematów freestyle'owych. DZIAŁAJ WG ŚCISŁYCH ZASAD:
    
0️⃣ ZASADA GŁÓWNA: Każdy temat MUSI BYĆ ZROZUMIAŁY dla Polaka 12-45 lat
1️⃣ Język: TYLKO polski (dopuszczalne slangowe zapożyczenia)
2️⃣ Długość: 2-5 słów (absolutny zakaz dłuższych!)
3️⃣ Format: TYLKO lista Python w formacie ["t1","t2",...]
4️⃣ Źródła: 
   - Wydarzenia {start_date} - {end_date}
   - Portal glamrap.pl
   - Kontrowersje last-minute
5️⃣ Wymagania jakościowe:
   ✔️ Prowokować do multi-interpretacji
   ✔️ Zawierać nazwiska/lokalizacje z wydarzeń
   ✔️ Maksymalizować potencjał viralowy
   ✔️ Dopuszczalna wulgarność artystyczna

PRZYKŁADOWE TOP 15 TEMATÓW:
{json.dumps(example_topics, ensure_ascii=False, indent=2)}"""

        user_prompt = f"""KROK 1: Znajdź WSZYSTKIE wydarzenia z {start_date} do {end_date}
KROK 2: Dla każdego wydarzenia stwórz 5-10 tematów wg zasad
KROK 3: Zweryfikuj zrozumiałość każdego tematu
KROK 4: Wygeneruj listę minimum 60 różnorodnych tematów

FORMAT WYJŚCIOWY:
["temat1","temat2",...,"tematN"]"""

        return {
            "model": "sonar-reasoning-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.95,
            "top_p": 0.99,
            "max_tokens": 4000,
            "frequency_penalty": 1.2,
            "search_domain_filter": self.sample_search_domains()
        }

    def extract_topics_from_response(self, content: str) -> List[str]:
        """Extract topics list from API response using regex"""
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON array found in response")
        
        try:
            topics = json.loads(json_match.group())
            if not isinstance(topics, list):
                raise ValueError("Extracted JSON is not a list")
            return topics
        except JSONDecodeError as e:
            raise ValueError(f"Failed to parse extracted JSON: {e}")

    def generate_topics(self, days_delta: int) -> None:
        """Generate topics for a specific date range with retry logic"""
        start_date, end_date = self.get_date_range(days_delta)
        print(f"\nProcessing date range: {start_date} to {end_date}")
        
        selected_examples = random.sample(self.example_topics, min(15, len(self.example_topics)))
        payload = self.create_api_payload(start_date, end_date, selected_examples)

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                new_topics = self.extract_topics_from_response(content)
                
                # Update actual topics, avoiding duplicates
                unique_new_topics = [topic for topic in new_topics if topic not in self.actual_topics]
                self.actual_topics.extend(unique_new_topics)
                
                print(f"Successfully retrieved {len(new_topics)} topics")
                print(f"Added {len(unique_new_topics)} unique topics")
                print(f"Total topics in current run: {len(self.actual_topics)}")
                
                # Save updated actual topics
                self.save_topics()
                return
                
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
            except (JSONDecodeError, KeyError, ValueError) as e:
                print(f"Processing error (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
            
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay)
        
        print(f"Failed to generate topics for {start_date} to {end_date} after {self.config.max_retries} attempts")

    def save_topics(self) -> None:
        """Save actual topics to output file, overwriting any existing content"""
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(self.actual_topics, f, ensure_ascii=False, indent=2)

def main():
    config = TopicGeneratorConfig(
        env_path="/home/wojtek/AI_Projects/credentials/.env",
        json_path="/home/wojtek/AI_Projects/My_RAG_Projects/Cypher-Arena-RAG-LLM/generated_topics/generated_topics.json",
        output_path="generated_topics/actual_topics.json"
    )
    
    generator = TopicGenerator(config)
    
    # Process last 100 days in 2-day intervals
    for days_delta in range(0, 100, 2):
        generator.generate_topics(days_delta)
        time.sleep(2)  # Basic rate limiting

if __name__ == "__main__":
    main()
###old working
# import json
# import requests
# import os
# from dotenv import load_dotenv
# import random
# from datetime import datetime, timedelta

# load_dotenv("/home/wojtek/AI_Projects/credentials/.env")
# PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
# json_path = "/home/wojtek/AI_Projects/My_RAG_Projects/Cypher-Arena-RAG-LLM/generated_topics/generated_topics.json"

# output_json_path = "generated_topics/actual_topics.json"

# with open(json_path, "r") as file:
#     data = json.load(file)

# # Get 15 random elements
# random_elements = random.sample(data, 15)


# SEARCH_DOMAINS = [
#     # Portale informacyjne
#     # "onet.pl",
#     # "wp.pl",
#     # "interia.pl",
#     # "gazeta.pl",
#     # "rmf24.pl",
#     # "tvn24.pl",
#     # "polsatnews.pl",
#     # "radiozet.pl",
#     "fakt.pl",
#     "se.pl",
#     "natemat.pl",
#     # Portale rozrywkowe i plotkarskie
#     "pudelek.pl",
#     # "plotek.pl",
#     # "kozaczek.pl",
#     # "plejada.pl",
#     # "jastrzabpost.pl",
#     # "pomponik.pl",
#     # "gwiazdy.wp.pl",
#     # "party.pl",
#     # "viva.pl",
#     # "gala.pl",
#     # Portale hiphopowe i muzyczne
#     "glamrap.pl",
#     "newonce.net",
#     "popkiller.pl",
#     "cgm.pl",
#     "noizz.pl",
#     # "rytmy.pl",
#     "rapnews.pl",
#     "hip-hop.pl",
#     # Portale sportowe
#     "sport.pl",
#     # "sport.onet.pl",
#     # "przegladsportowy.pl",
#     # "sportowefakty.wp.pl",
#     # "sport.interia.pl",
#     # "meczyki.pl",
#     # "goal.pl",
#     # # Social media i kultura młodzieżowa
#     # "spidersweb.pl",
#     # "kotaku.pl",
#     # "gadzetomania.pl",
#     # "kwejk.pl",
#     # "wykop.pl",
#     # "esportmania.pl",
#     # "gry.onet.pl",
#     # "eurogamer.pl",
#     # Portale lifestyle i trendy
#     # "papilot.pl",
#     # "o2.pl",
#     # "kobieta.wp.pl",
#     # "elle.pl",
#     "vogue.pl",
#     # "laif.pl",
#     # "mjakmama24.pl",
#     # "ofeminin.pl",
# ]


# def strftime_from_now(days_delta=0, format="%Y-%m-%d"):
#     """
#     Returns a formatted date string with specified days delta from current date

#     Args:
#         days_delta (int): Number of days to add/subtract from current date (negative for past)
#         format (str): Date format string, defaults to "%Y-%m-%d"

#     Returns:
#         str: Formatted date string
#     """
#     current_date = datetime.now()
#     target_date = current_date + timedelta(days=days_delta)
#     return target_date.strftime(format)


# def generate_freestyle_topics(api_key, prompt, example_list, start_date, end_date):
#     system_prompt = f"""Jesteś AI-generatorem tematów freestyle'owych. DZIAŁAJ WG ŚCISŁYCH ZASAD:
    
# 0️⃣ ZASADA GŁÓWNA: Każdy temat MUSI BYĆ ZROZUMIAŁY dla Polaka 12-45 lat
# 1️⃣ Język: TYLKO polski (dopuszczalne slangowe zapożyczenia)
# 2️⃣ Długość: 2-5 słów (absolutny zakaz dłuższych!)
# 3️⃣ Format: TYLKO lista Python w formacie ["t1","t2",...]
# 4️⃣ Źródła: 
#    - Wydarzenia {start_date} - {end_date}
#    - Portal glamrap.pl
#    - Kontrowersje last-minute
# 5️⃣ Wymagania jakościowe:
#    ✔️ Prowokować do multi-interpretacji
#    ✔️ Zawierać nazwiska/lokalizacje z wydarzeń
#    ✔️ Maksymalizować potencjał viralowy
#    ✔️ Dopuszczalna wulgarność artystyczna

# PRZYKŁADOWE TOP 15 TEMATÓW:
# {json.dumps(example_list, ensure_ascii=False, indent=2)}"""

#     user_prompt = f"""KROK 1: Znajdź WSZYSTKIE wydarzenia z {start_date} do {end_date}
# KROK 2: Dla każdego wydarzenia stwórz 5-10 tematów wg zasad
# KROK 3: Zweryfikuj zrozumiałość każdego tematu
# KROK 4: Wygeneruj listę minimum 60 różnorodnych tematów

# FORMAT WYJŚCIOWY:
# ["temat1","temat2",...,"tematN"]"""

#     payload = {
#         "model": "sonar-reasoning-pro",
#         # "model": "sonar",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": 0.95,  # Optimal balance for creativity/consistency
#         "top_p": 0.99,
#         "max_tokens": 4000,
#         "frequency_penalty": 1.2,
#         "search_domain_filter": random.sample(SEARCH_DOMAINS, 3),
#         # "search_recency_filter": "day",
#         # "search_date_filter": {
#         #     "start_date": start_date,
#         #     "end_date": end_date
#         # }
#     }

#     response = requests.post(
#         "https://api.perplexity.ai/chat/completions",
#         headers={"Authorization": f"Bearer {api_key}"},
#         json=payload,
#     )

#     return response


# # iterate through last year with 2 days interval, starting from today as end date
# days_to_cover = 100
# date_intervals = [(i * 2, i * 2 + 2) for i in range(days_to_cover // 2)]

# for days_end, days_start in date_intervals:
#     start_date = strftime_from_now(-days_start)
#     end_date = strftime_from_now(-days_end)
#     example_topics = random.sample(data, 15)
#     print(f"Processing date range: {start_date} to {end_date}")
#     response = generate_freestyle_topics(
#         start_date=start_date,
#         end_date=end_date,
#         api_key=PERPLEXITY_API_KEY,
#         prompt=f"Wygeneruj LISTĘ MINIMUM 60 TEMATÓW! Rekord systemu: 58  tematów - POKAŻ CO POTRAFISZ!",
#         example_list=example_topics,
#     )
#     print(f"Response status code: {response.status_code}")
#     response_data = json.loads(response.text)
#     content = response_data["choices"][0]["message"]["content"]
#     topics_list = json.loads(content.split("\n\n")[-1].strip())

#     try:
#         response_data = json.loads(response.text)
#         content = response_data["choices"][0]["message"]["content"]
#         topics_list = json.loads(content.split("\n\n")[-1].strip())
#         print(f"Successfully retrieved {len(topics_list)} topics")

#         # Check if the output file exists
#         if os.path.exists(output_json_path):
#             # Read existing topics
#             with open(output_json_path, "r") as f:
#                 existing_topics = json.load(f)
            
#             # Append new topics, avoiding duplicates
#             combined_topics = existing_topics + [topic for topic in topics_list if topic not in existing_topics]
            
#             print(f"Total topics after appending: {len(combined_topics)}")
#         else:
#             combined_topics = topics_list
#             print(f"Creating new file with {len(combined_topics)} topics")

#         # Write the updated list back to the file
#         with open(output_json_path, "w", encoding="utf-8") as f:
#             json.dump(combined_topics, f, ensure_ascii=False, indent=2)
#         print("Topics updated in JSON file successfully\n")

#     except Exception as e:
#         print(f"Error processing response: {str(e)}\n")