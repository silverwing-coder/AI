"""
install transformers, langchain, langchain-huggingface
$ pip install -r requirments.txt
$ pip install python-dotenv
# $ pip install torch

downloaded model files are saved to ~/.cache//huggingface/hub/
"""

# This code is to  
from dotenv import load_dotenv
load_dotenv()

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Explain about GPT2 Tokenizer."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)