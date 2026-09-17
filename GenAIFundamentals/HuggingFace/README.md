<h2>: Hugging Face Model Applications Practice Page</h2>
<h4>/** Edited by Sangmork Park at VMI, Last update: Sep. 2026 </h4>

---

<h3>Programming environment setup</h3>

``` sh
$ pip install -r requiremts.txt
# List dependencies in requirements.txt
#   : python-dotenv, transformers, langchain, langchain-huggingface, torch, torchvideo, torchaudio, ...
# Install torch if you have a GPU
```

---
``` py
'''
# token delivery setup
1. Generate token from hugging face: Settings -> Access tokens -> Create new Access Token ....
2. Copy and save the token in .env file: HF_TOKEN="your-token"

'''
from dotenv import load_dotenv
load_dotenv()
```