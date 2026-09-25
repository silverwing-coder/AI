
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
    "role": "system", "content": "You are an AI assistant that extract entities from text as JSON."
    "\nHere is an example of your output format: \n{\n\"the_name\":\"\", \n\"the_company\":\":\":,\n\"a_phone_number\":"
    "\"\"\n}. "
    # "Only fill in the fields outlined in th output format and not additional fields."
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
        print("AI:" + assistant_reply + "\n")
        # conversation.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        print(f"Error: {e}")