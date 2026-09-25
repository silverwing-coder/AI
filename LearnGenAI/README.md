<h2>Reference directory for learning Generative AI implementation</h2>
<h4>/** Edited by Sangmork Park at VMI, Last update: Sep. 2026 </h4>

The applications provided in this directory are to provide reference for learning LLMs with Hugging Face models and OpenAI models. 

<h3>Programming environment setup</h3>

``` sh
$ pip install -r requiremts.txt
# List dependencies in requirements.txt
#   : python-dotenv, transformers, langchain, langchain-huggingface, torch, torchvideo, torchaudio, gradio, openai, ...
# Install torch if you have a GPU
```

---
<br>

``` py
'''
# token delivery setup
1. Generate token from hugging face: Settings -> Access tokens -> Create new Access Token ....
2. Copy and save the token in .env file: HF_TOKEN="your-token"

'''
from dotenv import load_dotenv
load_dotenv()
```
