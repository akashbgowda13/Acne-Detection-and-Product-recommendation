import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO


# Import skincare dataset
dff = pd.read_csv("result.csv")

# Load YOLO Model
def load_yolo_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

# Header
st.set_page_config(page_title="Acne Detection & Product Recommendation App", page_icon=":blossom:", layout="wide")

# Display the main page
st.title("Acne Detection & Product Recommendation App :sparkles:")
st.write('---')

# Displaying a local video file
video_file = open("skincare.mp4", "rb").read()
st.video(video_file, start_time=1)  # Displaying the video 

# Upload Image
user_image = st.file_uploader("Upload a photo of your face", type=["jpg", "png", "jpeg"])
if user_image:
    img_path = "/tmp/user_uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(user_image.read())
    st.image(user_image, caption="Uploaded Image", use_column_width=True)
    acne_type = YOLO("best.pt")(img_path).names[0]
    st.write(f"**Acne Type:** {acne_type}")

    # Allow user to select their skin type

    st.write("**Recommended Products:**")
    from mp_skin_care_recommender_system import recommend_products_based_on_acne_type
    recommendations = recommend_products_based_on_acne_type(acne_type , dff)

    # Display recommendations
    if isinstance(recommendations, pd.DataFrame):
        st.dataframe(recommendations)
    else:
        st.error("No valid recommendations found!")


# More filters and recommendations based on user input...
