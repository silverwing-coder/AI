'''
Edited May. 2026 by Sangmork Park at VMI

This code is for generating images using the Stability AI API. It allows you 
to create images based on text prompts and save them to a specified directory.

1. Sign up for an API key at https://platform.stability.ai
2. Get your API key and set it as an environment variable named 'STABILITY_API_KEY'
3. Install the required library:
   $ pip install stability-sdk
   
   ## AI consultation will be most helpful if you encounter any issues during installation or usage.
   $ pip install --upgrade pip setuptools wheel
   $ pip install grpcio grpcio-tools
   $ pip install stability-sdk --no-deps
   $ pip install Pillow python-dotenv

4. Run the code to generate images based on your prompts.

## Make sure to replace the example prompts with your own creative ideas to generate unique images!
'''

import os
# from stability_sdk import client
# import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv
load_dotenv()

import requests
import json

api_host = "https://api.stability.ai"
url = f"{api_host}/v1/engines/list"

response = requests.get(url, headers={"Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"})
# print(response.json())

if response.status_code == 200:
    engines = response.json()
    print("Available engines:")
    for engine in engines:
        print(f"- {engine['id']}: {engine['name']}")
else:
    print(f"Failed to retrieve engines: {response.status_code} - {response.text}")

