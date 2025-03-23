import base64
import requests
from urllib.parse import urlparse
import json

def image_file_to_base64(image_path):
    """Convert an image file to Base64 string, supporting local paths, S3, and GCS URLs."""
    parsed_url = urlparse(image_path)

    if parsed_url.scheme in ["http", "https"]:  # For public URLs (GCS, S3 with public access)
        response = requests.get(image_path)
        if response.status_code == 200:
            base64_string = base64.b64encode(response.content).decode('utf-8')
        else:
            raise Exception(f"Failed to fetch image from {image_path}")
    else:  # Local file path
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    return base64_string

def save_base64_to_json(base64_image, output_path):
    """Save Base64 image string to a JSON file."""
    data = {"base64_image": base64_image}
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


base64_image = image_file_to_base64(r'inputs/Autumn/Z_AUTM_RT_P_141.JPG')
save_base64_to_json(base64_image,r'outputs/base64.json')
print(base64_image)