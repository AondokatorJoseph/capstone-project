# streamlit_app.py
import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Define project root for file paths
project_root = r"C:\Users\aondo\OneDrive\Documents\semester three fanshawe college\INFO 6156 Capstone Project\main_capstone"

# Import your modules
from src.models.model import WasteClassifier
from src.utils.helpers import load_and_preprocess_image

class WasteClassificationApp:
    def __init__(self):
        self.classifier = WasteClassifier()
        self.model = self.classifier.build_model()
        
        # Define categories and recyclable materials
        self.categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        self.recyclable = ['glass', 'paper', 'cardboard', 'plastic', 'metal']
    
        # Load model weights
        weights_path = os.path.join(project_root, 'models', 'saved_models', 'waste_classifier.weights.h5')
        self.model.load_weights(weights_path)
        
    def predict(self, image):
        # Preprocess the image
        processed = cv2.resize(image, (224, 224))
        if len(processed.shape) == 2:  # Convert grayscale to RGB
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        prediction = self.model.predict(processed)
        results = []
        
        # Get top 3 predictions
        for i in np.argsort(prediction[0])[-3:][::-1]:
            confidence = prediction[0][i]
            material = self.categories[i]
            is_recyclable = material in self.recyclable
            results.append((material, confidence, is_recyclable))
            
        return results

def main():
    st.set_page_config(page_title="CyclotekAI - Waste Classification", page_icon="♻️")
    
    # Set up page structure
    header = st.container()
    chat_area = st.container()
    input_area = st.container()
    
    # Initialize session state for chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize processed files tracking
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    # Header Section
    with header:
        st.title("CyclotekAI - Waste Classification")
        st.write("Upload an image or take a photo to identify waste material and check recyclability")
    
    # Chat Area - Display conversation history
    with chat_area:
        # Display all previous messages
        for message in st.session_state.messages:
            if message['role'] == 'user':
                with st.chat_message('user'):
                    st.image(message['content'], use_container_width=True)
            else:
                with st.chat_message('assistant'):
                    for material, confidence, is_recyclable in message['content']:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{material.title()}** ({confidence:.1%})")
                            st.progress(float(confidence))
                        with col2:
                            if is_recyclable:
                                st.success("♻️ Recyclable")
                            else:
                                st.error("🗑️ Non-recyclable")
    
    # Input Area at the bottom
    with input_area:
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
        
        with col2:
            camera_button = st.button("Take a Photo", key="camera_button")
            if camera_button:
                camera_input = st.camera_input("Take a picture")
                if camera_input:
                    uploaded_file = camera_input
        
        # Process the image if uploaded
        if uploaded_file is not None:
            # Create unique file ID
            file_id = hash(uploaded_file.name + str(uploaded_file.size))
            
            # Only process if we haven't seen this file before
            if file_id not in st.session_state.processed_files:
                app = WasteClassificationApp()
            
                # Load and process image
                image = np.array(Image.open(uploaded_file))
            
                # Add user message to chat
                st.session_state.messages.append({
                    'role': 'user',
                    'content': image
                })
            
                # Process and add assistant response
                with st.spinner("Classifying..."):
                    results = app.predict(image)
                
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': results
                    })
                
                # Mark this file as processed
                st.session_state.processed_files.add(file_id)
                
                # Rerun to update the chat interface
                st.rerun()

if __name__ == "__main__":
    main()