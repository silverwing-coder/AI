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

from transformers import pipeline, BitsAndBytesConfig

import torch

torch.cuda.empty_cache()
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())

# Insert cache-cleaning command at the beginning of training loop or validation step
# to make PyTorch release unused cached memory back to GPU

''' Example-1: Summarization '''

text = "Three people were dead at the scene and another was taken to a hospital in unknown condition, the department in a statement. At least one of the deceased appeared to be a pedestrian on the ground when the crash happened. Fire Capt. Branden Silverman said at the scene.\
The helicopter crash occurred after two people were killed and others were injured in a crash between a Los Angeles Metro bus and a passenger vehicle on Nordhoff Street, according to the fire department.It said that the bus crash happened at 5:03 p.m. and that firefighters worked to free the trapped people and assess the bus passengers."

# summarization task became obsolete -> use text-generation
model_id = "Qwen/Qwen3-4B-Instruct-2507"

""" High-level pipeline() loads the model into 32-bitn precision (FP32) which forces
    4-billion parameter model to consume nearly 16 GB of memory. 
    Wd need explicitly pass down quantization and data-type arguments to the underlying 
    model config within the pipeline 
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

# model = pipeline(task="text-generation", model=model_id)

messages = [
    {
        "role": "user",
        "content": "Summarize the following text in 3 bullet points:\n\n" + text
    }
]

out = model(messages, max_new_tokens=200)
print(out[0]["generated_text"][-1]["content"])



