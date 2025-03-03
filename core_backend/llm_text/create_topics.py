from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

load_dotenv(dotenv_path="/home/wojtek/AI_Projects/credentials/.env")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY)

openrouter_models_list = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


google_gemini_models_list = [
    "gemini-2.0-pro-exp-02-05",  # Rate limits 5 RPM Free 2 RPM 50 req/day
    "gemini-2.0-flash-thinking-exp-01-21",  # Rate limits 10 RPM Free 10 RPM 1500 req/day
    "gemini-2.0-flash-exp",  # Rate limits 10 RPM Free 10 RPM 1500 req/day
]


def generate_gemini_content(api_key, model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=data)
    return response.json()


# generate_gemini_content(
#     api_key=GOOGLE_AI_STUDIO_API_KEY,
#     model_name="gemini-2.0-pro-exp-02-05",
#     prompt=prompt,
# )


##openrouter limits: 20 per minute, 200 per day
completion = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-preview-02-05:free",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)

print(completion.choices[0].message.content)
