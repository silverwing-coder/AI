"""
install transformers, langchain, langchain-huggingface
$ pip install -r requirments.txt
$ pip install python-dotenv
$ pip install torch

downloaded model files are saved to ~/.cache//huggingface/hub/
"""
# setup for HF_TOKEN injection
from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
import torch

text = "Three people were dead at the scene and another was taken to a hospital in unknown condition, the department in a statement. At least one of the deceased appeared to be a pedestrian on the ground when the crash happened. Fire Capt. Branden Silverman said at the scene.\
The helicopter crash occurred after two people were killed and others were injured in a crash between a Los Angeles Metro bus and a passenger vehicle on Nordhoff Street, according to the fire department.It said that the bus crash happened at 5:03 p.m. and that firefighters worked to free the trapped people and assess the bus passengers."
summarizer = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")

messages = [
    {
        "role": "user",
        "content": "Summarize the following text in 3 bullet points:\n\n" + text
    }
]

out = summarizer(messages, max_new_tokens=200)
print(out[0]["generated_text"][-1]["content"])



