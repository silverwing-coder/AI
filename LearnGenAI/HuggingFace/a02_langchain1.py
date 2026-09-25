from dotenv import load_dotenv
load_dotenv()

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# $ pip install langchain-core 
# // langchain.prompts was moved to a dedicated core package
from langchain_core.prompts import PromptTemplate 

from transformers import pipeline, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = pipeline(
    task = "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    # device = 0,
    max_length = 256,
    truncation = True,
    model_kwargs={
        "quantization_config": quantization_config,
        "torch_dtype": torch.float16,
        "device_map": "auto"    # Automatically maps layers to GPU
    }
)

# LangChain components cannot directly talk to raw Hugging Face objects.
# We use HuggingFacePipeline wrapper. 
llm = HuggingFacePipeline(pipeline=model)

# Create the prompt template
template = PromptTemplate.from_template("Explain {topic} in detail for {age} year old to understand.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

# Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)