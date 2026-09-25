<h2>Reference repository for AI and ML applications implementation</h2>
<h4>/** Edited by Sangmork Park at VMI, Last update: Sep. 2026 </h4>

This repository is designed to be a resource for the students and novice researchers entering the domains of AI and ML.
The applications provided in this repository functioned effectively at the time of their development, however, some of them outdated due to the rapid advancement of theories and technologies. 

<h3>Programming environment setup</h3>

``` sh
$ pip install -r requirements.txt
# List dependencies in requirements.txt
#   : mediaipe, ultralytics, python-dotenv, transformers, langchain, langchain-huggingface, torch, torchvideo, torchaudio, gradio, openai, ...
# Install torch if you have a GPU
```

---
<br>

``` py
'''
# token delivery setup
1. Generate token from hugging face: Settings -> Access tokens -> Create new Access Token ....
2. Copy and save the token in .env file: XX_TOKEN="your-token"

'''
from dotenv import load_dotenv
load_dotenv()
```
