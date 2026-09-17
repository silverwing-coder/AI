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

--- 
<h3>When memory exhaustion problem ...</h3>

``` py
""" High-level pipeline() loads the model into 32-bits precision (FP32) 
    which forces 4-billion parameter model to consume nearly 16 GB of memory. 
    We need explicitly pass down quantization and data-type arguments 
    to the underlying model config within the pipeline 
    $ pip install bitsandbytes accelerate    
"""

# 1. Configure 4-bit quantization to fit your 7.53 GiB GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Pass quantization settings directly via model_kwargs
model = pipeline(
    task="text-generation", 
    model=model_id,
    model_kwargs={
        "quantization_config": quantization_config,
        "torch_dtype": torch.float16,
        "device_map": "auto"    # Automatically maps layers to GPU
    }
)

```