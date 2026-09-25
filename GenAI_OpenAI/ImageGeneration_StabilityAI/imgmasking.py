import os
import base64
import requests
import datetime
import re
from dotenv import load_dotenv
load_dotenv()

# engine_id = "stable-inpainting-512-v2-0"
engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = "https://api.stability.ai"
api_key = os.getenv('STABILITY_API_KEY')

original_image = "generated_images/serene_vacation.jpeg"
mask_image = "generated_images/mask_serene_vacation.jpeg"
prompt = " boat with a person fishing and a dog in the boat."

img_dir = os.path.join(os.getcwd(), "generated_images")
if not os.path.exists(img_dir):
    os.makedirs(img_dir)    

def validate_filename(filename):
    # Remove invalid characters and trim the filename
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Limit the filename length to 255 characters
    return filename.strip()

response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/image-to-image/masking",
    headers={"Accept": "application/json",
             "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"
    },
    files={"init_image": open(original_image, "rb"), 
           "mask_image": open(mask_image, "rb")},
    data = {
        "mask_source": "MASK_IMAGE_BLACK",
        "text_prompts[0][text]": prompt,
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "samples": 4,
        "steps": 50,
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
