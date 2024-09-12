import streamlit as st
import cv2
import easyocr
import numpy as np
import time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Title and description
st.title("OCR Model Comparison: GPU vs CPU")
st.write("Upload a video file to see the OCR results and performance comparison between GPU and CPU models.")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Initialize EasyOCR reader for both GPU and CPU
@st.cache_resource
def initialize_reader(use_gpu):
    return easyocr.Reader(['en'], gpu=use_gpu)

# Function to process video and extract text
def process_video(video_path, use_gpu):
    reader = initialize_reader(use_gpu)
    
    # Capture the video
    capture = cv2.VideoCapture(video_path)
    
    prev_frame_time = 0
    total_frames = 0
    total_fps = 0
    extracted_text = []
    
    # Processing video frames
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        total_frames += 1
        
        # Resize frame (optional)
        frame = cv2.resize(frame, (420, 320))
        
        # OCR on the frame
        result = reader.readtext(frame)
        
        # Extract text from the OCR result
        for detection in result:
            extracted_text.append(detection[1])
        
        # FPS calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        total_fps += fps

    avg_fps = total_fps / total_frames if total_frames > 0 else 0
    accuracy = 0.9178  # Mock accuracy value for demonstration
    
    return avg_fps, accuracy, extracted_text

# Function to filter and include only meaningful distinct words
def filter_distinct_words(text_list, ignore_phrases):
    ignore_phrases_set = set(ignore_phrases)
    words = []
    for text in text_list:
        # Split text into words and filter
        words.extend(word for word in text.split() if word not in ignore_phrases_set)
    return ' '.join(sorted(set(words)))

# Run when video is uploaded
if uploaded_file:
    # Save the uploaded video file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Define the phrases to exclude
    ignore_phrases = [
        'VEED', 'SUPPORT', 'DEVELOPMENT', 'INFORMATION TECHNOLOGY', 
        'CONSULTING', 'STOCK', 'AStocConsulting', 'Infrastructure'
    ]
    
    # Process the video for GPU model
    st.write("Processing video using **GPU** model...")
    avg_fps_gpu, accuracy_gpu, extracted_text_gpu = process_video("uploaded_video.mp4", use_gpu=True)
    
    # Process the video for CPU model
    st.write("Processing video using **CPU** model...")
    avg_fps_cpu, accuracy_cpu, extracted_text_cpu = process_video("uploaded_video.mp4", use_gpu=False)
    
    # Display the extracted and filtered results
    st.subheader("Extracted Text (GPU):")
    st.write(' '.join(extracted_text_gpu))
    
    st.subheader("Distinct Words (GPU):")
    st.write(filter_distinct_words(extracted_text_gpu, ignore_phrases))
    
    st.subheader("Extracted Text (CPU):")
    st.write(' '.join(extracted_text_cpu))
    
    st.subheader("Distinct Words (CPU):")
    st.write(filter_distinct_words(extracted_text_cpu, ignore_phrases))
    
    # Display the FPS and Accuracy for both GPU and CPU
    st.subheader("Performance Comparison")
    st.write(f"**GPU** - FPS: {avg_fps_gpu:.2f}, Accuracy: {accuracy_gpu * 100:.2f}%")
    st.write(f"**CPU** - FPS: {avg_fps_cpu:.2f}, Accuracy: {accuracy_cpu * 100:.2f}%")
    
    # Visualization of the FPS comparison
    fps_data = {
        "Model": ["GPU", "CPU"],
        "FPS": [avg_fps_gpu, avg_fps_cpu]
    }
    df_fps = pd.DataFrame(fps_data)
    
    st.subheader("FPS Comparison Chart")
    fig_fps, ax_fps = plt.subplots()
    df_fps.plot(x="Model", y="FPS", kind="bar", ax=ax_fps, color=["#0d88e6", "#00b7c7"])
    ax_fps.set_ylabel("FPS")
    ax_fps.set_title("FPS Comparison: GPU vs CPU")
    st.pyplot(fig_fps)
    
    # Visualization of the Accuracy comparison
    accuracy_data = {
        "Model": ["GPU", "CPU"],
        "Accuracy (%)": [accuracy_gpu * 100, accuracy_cpu * 100]
    }
    df_accuracy = pd.DataFrame(accuracy_data)
    
    st.subheader("Accuracy Comparison Chart")
    fig_accuracy, ax_accuracy = plt.subplots()
    df_accuracy.plot(x="Model", y="Accuracy (%)", kind="bar", ax=ax_accuracy, color=["#0d88e6", "#00b7c7"])
    ax_accuracy.set_ylabel("Accuracy (%)")
    ax_accuracy.set_title("Accuracy Comparison: GPU vs CPU")
    st.pyplot(fig_accuracy)
