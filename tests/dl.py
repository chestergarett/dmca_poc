import urllib.request
import bz2
import shutil

# Download the file
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
filename = "shape_predictor_68_face_landmarks.dat.bz2"

print("Downloading landmark predictor model...")
urllib.request.urlretrieve(url, filename)

# Extract the file
with bz2.BZ2File(filename) as f_in, open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

print("Download complete! The model is ready to use.")
