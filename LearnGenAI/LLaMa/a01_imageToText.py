"""
This code does not work on RTX-4060 model since the current Llama vision models require 11B
    - RTX 4060 vRAM size: 8GB
    - Llama-3.2-11B-Vision weights: 11B
"""

import torch
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# 1. Authenticate with Hugging Face (Replace with your actual token)
import dotenv
from dotenv import load_dotenv
load_dotenv()

from PIL import Image

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# 2. Configure 4-bit quantization to prevent VRAM spillover
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3. Load the processor and the model cleanly into GPU memory
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto" # Automatically handles proper mapping
)

# 4. Open and convert the file into a PIL Image object
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

image_path = "images/weddingcouple.jpg"
image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# 5. Apply the official template to generate a clean string prompt
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 6. Pass the clean string prompt and the image into the processor
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
).to("cuda") # Ensure it routes to your RTX 4060

# 5. Generate text
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))