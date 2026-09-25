import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN")
)

prompt = "Suggest three names for a new pet salon business. The generated name ideas " \
"should evoke positive emotions and the following key features: professional, friendly, and personalized service."

response = client.completions.create(
    model="gpt-4o-mini",
    prompt=prompt,
    temperature=0.7,
    max_tokens=100,
    # suffix="\nThat all folks!"
    stop=None
)

# response_text = response.choices[0].text.strip()
# print("Prompt:\n")
# print(prompt)
print("Generated Business Names:")
print(response)