
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN")
)

GPT_MODEL = "gpt-4o-mini"  # Exact string ID for GitHub Models

conversation = [{ 
    "role": "system", "content": "You are an AI assistant that helps people find information."
    "You can only talk about pets and nothing else. If you dont't knlw the answer, say \"Sorry bud,"
    "I don't know that.\" And If you cannot answer it, say \"Sorry mate, can't answer that - I am"
    "not allowed to\"."
     }]

print("Please enter what you want to talk about: ")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    conversation.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=conversation,
            temperature=0.7,
            max_tokens=300
        )
        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")
        conversation.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        print(f"Error: {e}")