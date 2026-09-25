import os
import re
import base64
import argparse
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

API_HOST = "https://api.stability.ai"
ENGINE_ID = "stable-diffusion-xl-1024-v1-0"
OUTPUT_DIR = os.path.join(os.getcwd(), "generated_images")


def validate_filename(filename: str) -> str:
    """Strip invalid filename characters and trim the result."""
    clean = re.sub(r'[<>:"/\\|?*]', '', filename)
    return clean.strip()[:200]


def generate_image(prompt: str, output_path: str = None, width: int = 1024, height: int = 1024, steps: int = 50, cfg_scale: int = 7):
    """Generate an image from a text prompt using Stability AI and save it."""
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable STABILITY_API_KEY is required")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = validate_filename(prompt).replace(' ', '_') or "stability_image"
        output_path = os.path.join(OUTPUT_DIR, f"{filename}_{timestamp}.png")

    url = f"{API_HOST}/v1/generation/{ENGINE_ID}/text-to-image"
    payload = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": cfg_scale,
        "clip_guidance_preset": "FAST_BLUE",
        "height": height,
        "width": width,
        "samples": 1,
        "steps": steps,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Stability API error {response.status_code}: {response.text}")

    data = response.json()
    if "artifacts" not in data or not data["artifacts"]:
        raise RuntimeError("No image artifact returned from Stability AI")

    image_base64 = data["artifacts"][0]["base64"]
    image_data = base64.b64decode(image_base64)

    with open(output_path, "wb") as f:
        f.write(image_data)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a prompt using Stability AI")
    parser.add_argument("prompt", nargs="?", help="Text prompt for the image generation")
    parser.add_argument("--output", "-o", help="Path to save the generated image file")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image")
    parser.add_argument("--steps", type=int, default=50, help="Number of generation steps")
    parser.add_argument("--cfg-scale", type=int, default=7, help="CFG scale for image generation")
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        prompt = input("Enter image prompt: ")

    output_file = generate_image(
        prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )

    print(f"Image saved to: {output_file}")


if __name__ == "__main__":
    main()
