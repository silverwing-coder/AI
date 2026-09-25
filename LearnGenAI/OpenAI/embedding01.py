"""
pip install openai
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def get_embedding(text):
    response = client.embeddings.create(
        # model="text-embedding-ada-002",
        model="text-embedding-3-small",
        input=[text]
    )
    print(response.data[0].embedding)
    return response.data[0].embedding

if __name__ == "__main__":
    # client = OpenAI(api_key=os.environ.get("GITHUB_TOKEN"))    
    # client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ.get("GITHUB_TOKEN"))
    client = OpenAI(base_url="https://azure.com", api_key=os.environ.get("GITHUB_TOKEN"))
    text = "Hello world, I am learning to create embeddings!"
    # text = "I have a white dog named Champ."
    embedding = get_embedding(text)
    print(f"Embedding Length: {len(embedding)}")
    # print(f"Embedding: {embedding[:50]}...")  # Print the first 50 dimensions for brevity
