import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN")
)

prompt = "Suggest three names and a tagline which is at least 3 sentences for a new pet salon business"\
    "THe generated name ideas shouuld evoke positive emotions and the following key features: "\
    "professional, friendly, and personalized service."

for response in client.completions.create(
    model="gpt-4o-mini",
    prompt=prompt,
    temperature=0.8,
    max_tokens=500,
    stream=True,
    # suffix="\nThat all folks!"
    stop=None
): 
    for choice in response.choices:
        sys.stdout.write((choice.text) + "\n")
        sys.stdout.flush()