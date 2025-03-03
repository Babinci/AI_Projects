from perplexityai import Perplexity
import requests
import json
try:
    client = Perplexity()
except requests.exceptions.RequestException as e:
    print(f"Connection error: {str(e)}")
except json.JSONDecodeError:
    print("Received invalid JSON from API server")