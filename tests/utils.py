import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif
import json
import dlib
from scipy.spatial import distance as dist

# Load dlib's facial landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Download this file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect blinks"""
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

def capture_face_from_camera(save_path="outputs/camera_face.json"):
    cap = cv2.VideoCapture(0)  # Open webcam
    print("Capturing face... Follow the liveness test.")

    blink_detected = False
    EAR_THRESHOLD = 0.2  # Adjust if needed
    BLINK_FRAMES = 3  # Number of frames with closed eyes required to confirm blink

    while not blink_detected:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            
            # Get eye landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0  # Average of both eyes

            # Check if EAR is below threshold for a few frames (confirm blink)
            if avg_ear < EAR_THRESHOLD:
                BLINK_FRAMES -= 1
                if BLINK_FRAMES == 0:
                    print("Blink detected! Liveness test passed.")
                    blink_detected = True
                    break
            else:
                BLINK_FRAMES = 3  # Reset counter if eyes are open

        cv2.imshow("Look at the camera and blink", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    if not blink_detected:
        print("Liveness test failed. Try again.")
        return None

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract face encoding
    encodings = face_recognition.face_encodings(rgb_frame)

    if not encodings:
        print("No face encoding detected, please try again.")
        return None

    face_encoding = encodings[0]

    # Save encoding to JSON
    face_data = {
        "name": "Captured_Face",
        "encoding": face_encoding.tolist()
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(face_data, f)

    print(f"Saved face encoding to {save_path}")
    return face_encoding

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
    return "No match found in video."


# Load reference images
# reference_folder = "inputs/Autumn"  # Folder containing images of the person
# mean_encoding, person_name = load_known_faces(reference_folder)

# # Test with a new image
# test_video_path = "comparisons/Autumn.mp4"
# recognize_person_in_video(test_video_path)

capture_face_from_camera(save_path="outputs/camera_face.json")