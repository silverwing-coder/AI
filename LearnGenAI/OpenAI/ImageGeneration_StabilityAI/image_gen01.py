import os
import base64
import requests
import datetime
import re
from dotenv import load_dotenv
load_dotenv()

# from stability_sdk import client
# import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
# from dotenv import load_dotenv
# load_dotenv()
engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = "https://api.stability.ai"

prompt = "A fantasy landscape with mountains, a river, and a castle in the background, in the style of Studio Ghibli"
# prompt = "Laughing panda in the cloudes eating bamboo"

img_dir = os.path.join(os.getcwd(), "generated_images")
if not os.path.exists(img_dir):
    os.makedirs(img_dir)    

def validate_filename(filename):
    # Remove invalid characters and trim the filename
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Limit the filename length to 255 characters
    return filename.strip()

response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/text-to-image",
    headers={"Content-type": "application/json", 
             "Accept": "application/json",
             "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"
    },
    json = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 50,
    },
)

if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
else:
    data = response.json()
    for i, image in enumerate(data["artifacts"]):
        img_data = base64.b64decode(image["base64"])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{validate_filename(prompt)[:50]}_{timestamp}_{i}.png"
        filepath = os.path.join(img_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img_data)
        print(f"Image saved to {filepath}")
