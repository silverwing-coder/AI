
from datetime import date
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN")
)

GPT_MODEL = "gpt-4o-mini"  # Exact string ID for GitHub Models

prompt = "Translate the following English text to French: 'I have a small dog named Champ.'"
# " Do not include any additional text. "

response = client.completions.create(
    model=GPT_MODEL,
    prompt=prompt,
    temperature=0.7,
    max_tokens=100, 
    stop=None
)


print("Prompt:")
print(prompt)

print("Translation:")
print(response.choices[0].text.strip())