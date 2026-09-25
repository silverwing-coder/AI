'''
Edited in May. 2026 by Sangmork Park at VMI

1. Go to the GitHub Marketplace Models catalog.
2. Sign in with a standard, free GitHub account.
3. Create a personal access token (PAT) in your GitHub developer settings.
4. Route your Python openai library calls to the Azure-hosted GitHub server instead of OpenAI's endpoint

Option-1: 
os.environ.get("GITHUB_TOKEN", "your-github-token-here")

Option-2:
$ export GITHUB_TOKEN="your-github-token-here"

Option-3: Use .env file
$ pip install python-dotenv
$ nano .env
    ---
    GITHUB_TOKEN="your-github-token-here"
    ---

from dotenv import load_dotenv
load_dotenv()
os.environ.get("GITHUB_TOKEN")

-------------------------------------------
$ pip install openAI

'''

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. Initialize the client using GitHub's inference server endpoint
# Note: Ensure you replace 'your_github_token_here' with your real token!
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN")
)

try:
    # 2. Call the chat completion endpoint using the exact model ID
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Exact string ID for GitHub Models
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that explains things simply."
            },
            {
                "role": "user", 
                "content": "Why is the sky blue?"
            }
        ],
        temperature=0.7,
        max_tokens=300
    )

    # 3. Cleanly parse the text response
    print("🤖 GPT-4o Mini Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"❌ An error occurred: {e}")
