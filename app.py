import cv2
import json
import os
import base64
import requests
import numpy as np
from flask import Flask, jsonify, request
from scipy.spatial import distance as dist
import face_recognition
from supabase import create_client, Client
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()
app = Flask(__name__)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
X_API_KEY = os.getenv('X_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def base64_to_image(base64_string):
    """Convert Base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_string)  # Decode Base64
    np_arr = np.frombuffer(img_data, np.uint8)  # Convert to NumPy array
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode image
    return image

def image_file_to_base64(image_path):
    """Convert an image file to Base64 string."""
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


def image_file_to_base64(image_path):
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

@app.route("/store_face", methods=["POST"])
def store_face():
    """API to store or update face encoding and email in Supabase."""
    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401

    data = request.json
    email_id = data.get("email")
    face_encoding = data.get("face_encoding")

    if not email_id:
        return jsonify({"error": "Missing email"}), 400

    # Check if email already exists
    existing_record = supabase.table("dml_face_coordinates").select("*").eq("email", email_id).execute()

    if existing_record.data:
        # Update face encoding if email exists
        response = supabase.table("dml_face_coordinates").update({"face_encoding": face_encoding}).eq("email", email_id).execute()
        return jsonify({"message": "Face encoding updated successfully!", "supabase_response": response.data})
    else:
        # Insert new record if email does not exist
        response = supabase.table("dml_face_coordinates").insert({
            "email": email_id,
            "face_encoding": face_encoding
        }).execute()

        return jsonify({"message": "Face encoding stored successfully!", "supabase_response": response.data})

@app.route('/convert_imagepath', methods=['POST'])
def convert_image_to_base64():
    data = request.get_json()
    image_path = data.get('image_path')

    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    if not image_path:
        return jsonify({'error': 'Image path is required'}), 400

    try:
        base64_string = image_file_to_base64(image_path)
        return jsonify({'base64_string': base64_string}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    """API to recognize a face based on email_id and Base64 image."""
    data = request.json
    email_id = data.get("email")
    base64_image = data.get("image")

    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    if not email_id or not base64_image:
        return jsonify({"error": "Missing email_id or image"}), 400

    # Query Supabase for face encoding
    response = supabase.table("dml_face_coordinates").select("face_encoding").eq("email", email_id).execute()

    if not response.data:
        return jsonify({"error": "Email ID not found in database"}), 404

    # Get stored encoding from Supabase
    stored_encoding = json.loads(response.data[0]["face_encoding"])  # Convert JSON string back to list
    stored_encoding = np.array(stored_encoding)  # Convert list to NumPy array

    # Convert Base64 image to OpenCV format
    image = base64_to_image(base64_image)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Detect face and extract encoding
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        return jsonify({"error": "No face detected"}), 400

    # Compare with stored encoding
    for encoding in face_encodings:
        matches = face_recognition.compare_faces([stored_encoding], encoding, tolerance=0.5)
        if True in matches:
            return jsonify({"message": f"{email_id} is recognized!"})

    return jsonify({"message": "Face not recognized"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
