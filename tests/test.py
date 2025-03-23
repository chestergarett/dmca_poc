
import streamlit as st
import tempfile
import os
from utils import recognize_person_in_video

# Streamlit App
st.title("🎥 Face Recognition in Video")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file as a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.video(temp_file.name)

    # Run face recognition
    if st.button("🔍 Run Face Recognition"):
        with st.spinner("Processing..."):
            result = recognize_person_in_video(temp_file.name)
        st.success(result)

    # Cleanup temp file after processing
    os.remove(temp_file.name)
