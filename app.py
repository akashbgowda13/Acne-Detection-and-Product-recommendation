import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tempfile


# Set up the page
st.set_page_config(page_title="Acne Detection & Product Recommendation System", page_icon=":rose:", layout="wide")

# Load Skincare Dataset
dff = pd.read_csv("result.csv")

# Load YOLO model
def load_yolo_model():
    model_path = "best.pt"
    if not model_path or not os.path.exists(model_path):
        st.error("Model not found! Please ensure the correct path.")
    model = YOLO(model_path)
    return model

# Get Acne Type
def get_acne_type(image_path):
    model = load_yolo_model()
    labels = {0: 'Blackheads', 1: 'Cysts', 2: 'Papules', 3: 'Pustules', 4: 'Whiteheads'}
    results = model(image_path)
    top_prediction = max(zip(results[0].probs.top5, results[0].probs.top5conf), key=lambda x: x[1])
    acne_type = labels[top_prediction[0]]
    # Standardize the detected acne type to match keys in acne_subtype_to_ingredients
    acne_type = acne_type.lower()
    return acne_type

# Option Menu function with horizontal layout
def streamlit_menu(example=1):
    if example == 1:
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Home", "Get Recommendation", "Skin Care Tips", "Contact Us"],  # Added "Contact Us" here
                icons=["house", "stars", "book", "envelope"],  # Corresponding icon for Contact Us
                menu_icon="cast",
                default_index=0,
            )
        return selected
    elif example == 2:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Get Recommendation", "Skin Care Tips", "Contact Us"],  # Added "Contact Us" here
            icons=["house", "stars", "book", "envelope"],  # Corresponding icon for Contact Us
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",  # This makes the menu horizontal
        )
        return selected

# Call the function to get the selected page
selected = streamlit_menu(example=2)

if selected == "Home":
    st.title("Welcome to Acne Detection & Product Recommendation System 🌹")
    st.write('---') 

    st.write("""
    ### This application uses Machine Learning to recommend skincare products based on your skin type and concerns.
    """)
    
    # Display video or welcome image
    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time=1)
    
    st.write("""
    #### Get personalized skincare recommendations from a wide variety of brands. Choose 'Get Recommendation' to start or 'Skin Care Tips' to learn more about skincare.
    """)
    st.info('Credit: Akash H B')

# Upload and save the image
if selected == "Get Recommendation":
    st.title("Get Your Personalized Skincare Recommendation")
    

    user_image = st.file_uploader("Upload a clear photo of your face with acne for analysis", type=["jpg", "png", "jpeg"])
    if user_image:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(user_image.read())
            img_path = tmp_file.name  # Save the path of the temp file

        st.image(user_image, caption="Uploaded Image", use_column_width=True)

        # Pass the path to the YOLO model for detection
        acne_type = get_acne_type(img_path)
        st.write(f"**Detected Acne Type:** {acne_type}")
        

        st.write("Fetching personalized product recommendations...")

        # Import the recommender system and display recommendations
        from mp_skin_care_recommender_system import recommend_products_based_on_acne_type
        recommendations = recommend_products_based_on_acne_type(acne_type, dff)
        
        # Fix: Check if recommendations is valid
        if isinstance(recommendations, pd.DataFrame):
            st.dataframe(recommendations)
        else:
            st.error(recommendations)


if selected == "Skin Care Tips":
    st.title("Skincare Tips and Advice")
    st.write('---') 

    st.write("""
    ### Tips to Maximize the Use of Your Skincare Products
    """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Dot&Key')

    st.write("""
    #### 1. Facial Wash
    - Use a facial wash recommended for your skin type.
    - Wash your face a maximum of twice a day (morning and night) to avoid stripping natural oils.
    - Use gentle, circular motions for 30-60 seconds with your fingertips.

    #### 2. Toner
    - Apply toner using a cotton pad or your hands for better absorption.
    - Avoid products with fragrance if you have sensitive skin.
    
    #### 3. Serum
    - Apply serum after cleaning for optimal absorption.
    - Choose a serum based on your specific needs, such as acne scars or anti-aging.

    #### 4. Moisturizer
    - Use a moisturizer that suits your skin type to lock in hydration.
    - Apply different moisturizers for day and night for better protection and regeneration.
    
    #### 5. Consistency
    - Consistency is key for skincare effectiveness. Stick to a routine for the best results.

    #### 6. Avoid Switching Products Frequently
    - Changing products often can cause stress to the skin. Stick to a product for a few months to see results.
    """)

if selected == "Contact Us":
    st.title("Contact Us")

    st.write("""
    ### Meet the Team
    """)
    team_members = {
        "Akash H B": "akashbgowda13@gmail.com",
        "Rahul Sah": "rahulsah@example.com",
        "Sooraj Suresh": "sooraj@example.com",
        "Nishan Menezes": "nishanmenezes@example.com"
    }

    for name, email in team_members.items():
        st.markdown(f"[{name}]({f'mailto:{email}'})")

# Footer
st.write("---")
st.write("Acne Detection & Product Recommendation System © 2024")
