import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif

# Load multiple reference images
import os
import json
import numpy as np
import face_recognition
import pillow_heif

def load_known_faces(reference_folder, save_path="outputs/faces.json"):
    encodings_list = []

    # Extract folder name as the person's name
    person_name = os.path.basename(reference_folder.rstrip("/\\"))

    for filename in os.listdir(reference_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".heic")):
            img_path = os.path.join(reference_folder, filename)

            # Convert HEIC to RGB image
            if filename.lower().endswith(".heic"):
                heif_image = pillow_heif.open_heif(img_path)
                image = heif_image.to_pillow()  # Convert to Pillow Image
                image = np.array(image)  # Convert to NumPy array
            else:
                image = face_recognition.load_image_file(img_path)

            encodings = face_recognition.face_encodings(image)
            if encodings:  # Ensure face is detected
                encodings_list.append(encodings[0])

    if not encodings_list:
        print(f"No valid face encodings found in folder: {reference_folder}")
        return None

    # Average all encodings to create a stronger representation of the person
    mean_encoding = np.mean(encodings_list, axis=0)

    # Save to JSON
    face_data = {
        "name": person_name,
        "encoding": mean_encoding.tolist()  # Convert NumPy array to list for JSON serialization
    }

    with open(save_path, "w") as f:
        json.dump(face_data, f)

    print(f"Saved face encoding for {person_name} to {save_path}")

    return mean_encoding, person_name



# Compare a test image
def recognize_person_in_video(video_path, json_path="outputs/faces.json"):
    # Load face encoding from JSON
    with open(json_path, "r") as f:
        face_data = json.load(f)
    
    person_name = face_data["name"]
    known_encoding = np.array(face_data["encoding"])  # Convert list back to NumPy array

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
         return "Error: Could not open video file."
        

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Prevent division by zero
        fps = 30  

    frame_interval = int(fps)  
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            matches = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.5)
            if True in matches:
                return f"{person_name} found in video!"
                video_capture.release()
                return True  

    video_capture.release()
    return "No match found in video."


# Load reference images
# reference_folder = "inputs/Autumn"  # Folder containing images of the person
# mean_encoding, person_name = load_known_faces(reference_folder)

# # Test with a new image
test_video_path = "comparisons/Autumn.mp4"
recognize_person_in_video(test_video_path)
