import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------- Configuration --------
MODEL_PATH = "Src/model_v2.keras"          # <--- Updated path
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# -------- Page Config (must be first) --------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")

# -------- Custom CSS for Dark/Light Mode --------
def set_theme(theme):
    if theme == "dark":
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
        border = "#404040"
    else:  # light
        bg_color = "#ffffff"
        text_color = "#262730"
        card_bg = "#f0f2f6"
        border = "#d0d0d0"

    st.markdown(f"""
    <style>
        /* Main background and text */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp .stMarkdown {{
            color: {text_color} !important;
        }}
        /* Card style for each image prediction */
        .pred-card {{
            background-color: {card_bg};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {card_bg};
        }}
    </style>
    """, unsafe_allow_html=True)

# -------- Load Model --------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please place your trained model in the 'Src' folder.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -------- Preprocessing Function --------
def preprocess_image(image):
    """Convert PIL image to normalized numpy array ready for the model."""
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------- Theme Toggle (in Sidebar) --------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio(
    "Choose Theme",
    ("🌞 Light", "🌙 Dark"),
    index=0,
    help="Switch between light and dark mode."
)
# Map radio selection to theme string
theme_mode = "dark" if theme == "🌙 Dark" else "light"
set_theme(theme_mode)

# -------- Main UI --------
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload one or more MRI scans to get predictions (Glioma, Meningioma, Pituitary, or No Tumor).")

if model is None:
    st.stop()

# -------- File Uploader (multiple files) --------
uploaded_files = st.file_uploader(
    "Choose MRI images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="You can select multiple images at once."
)

# Clear button to reset uploaded files
if uploaded_files:
    if st.button("🗑️ Clear All Images"):
        uploaded_files = None
        st.rerun()  # Forces refresh to clear the uploaded list

if uploaded_files:
    st.subheader(f"📋 {len(uploaded_files)} Image(s) Uploaded")
    
    # Create a grid layout: 3 images per row
    cols_per_row = 3
    rows = [uploaded_files[i:i+cols_per_row] for i in range(0, len(uploaded_files), cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, file in enumerate(row):
            with cols[idx]:
                # Display the image and its prediction inside a card
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)
                
                # Preprocess and predict
                with st.spinner(f"Classifying {file.name}..."):
                    processed = preprocess_image(image)
                    predictions = model.predict(processed)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    confidence = np.max(predictions, axis=1)[0]
                
                # Show results in a styled card
                st.markdown(f"""
                <div class="pred-card">
                    <b>Prediction:</b> {CLASS_NAMES[predicted_class]}<br>
                    <b>Confidence:</b> {confidence:.2%}
                </div>
                """, unsafe_allow_html=True)
                
                # Show probability bar chart (small)
                prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(4)}
                st.bar_chart(prob_dict, use_container_width=True)
                
                # Add a small separator
                st.divider()

else:
    st.info("📤 Please upload one or more MRI images to begin.")

# -------- Footer Disclaimer --------
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ This is a demo model and should not be used for real medical diagnosis.")
