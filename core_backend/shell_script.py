from llm_text.api_search import search_internet
from datetime import datetime, timedelta


end_date = datetime.now()
start_date = end_date - timedelta(days=7)

now=  datetime.now() - timedelta(days=21)
days_interval = 7
for _ in range(5):
    
    end_date = now - timedelta(days=days_interval)
    start_date = end_date - timedelta(days=7)
    for name in ["general_news", "polish_showbiznes"]:
        search_internet(start_date, end_date, search_model="sonar-pro", name=name)

    days_interval +=7
