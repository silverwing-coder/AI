import os
import base64
import requests
import datetime
import re
from dotenv import load_dotenv
load_dotenv()

# original_image = "../generated_images/inplane.jpg"
original_image = "generated_images/inplane_sketch.jpeg"
engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = "https://api.stability.ai"

prompt = "A happy pilot thumbs up in the cockpit of a plane"
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
    f"{api_host}/v1/generation/{engine_id}/image-to-image",
    headers={"Accept": "application/json",
             "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"
    },
    files={"init_image": open(original_image, "rb")},
    data = {
        "image_strength": 0.35,
        "init_image_mode": "IMAGE_STRENGTH",
        "text_prompts[0][text]": prompt,
        # "text_prompts": [
        #     {
        #         "text": "A happy pilot thumbs up in the cockpit of a plane", 
        #         "weight": 1
        #         }
        # ],
        # "width": 1024,
        # "height": 1024,
        "cfg_scale": 7,
        "samples": 1,
        "steps": 50,
        "sampler": "K_DPMPP_2M",
    },
)

if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
else:
    data = response.json()
    for i, image in enumerate(data["artifacts"]):
        # img_data = base64.b64decode(image["base64"])
        file_name = f"{validate_filename(prompt)[:50]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"{validate_filename(prompt)[:50]}_{timestamp}_{i}.png"
        filepath = os.path.join(img_dir, file_name)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
        print(f"Image saved to {filepath}")
