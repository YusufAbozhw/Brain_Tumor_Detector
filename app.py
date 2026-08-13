import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("⚙️ About System")
    st.info("""
    This application utilizes a Deep Learning model built with Keras to assist in detecting and classifying brain tumors from MRI scans.
    """)
    st.markdown("---")
    st.markdown("**Target Classes:**")
    st.markdown("""
    - 🔴 **Glioma** (Class 0)
    - 🟠 **Meningioma** (Class 1)
    - 🔵 **Pituitary** (Class 2)
    - 🟢 **No Tumor** (Class 3)
    """)
    st.caption("Note: This system is built for educational and demonstration purposes only and should not replace professional medical diagnosis.")

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain Magnetic Resonance Imaging (**MRI Scan**) to perform real-time classification using the deep learning model.")

st.divider()

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('Src/model_v2.keras')

with st.spinner("Loading AI Model..."):
    model = load_keras_model()

def preprocess_and_predict(pil_image, model):
    img_resized = pil_image.resize((128, 128))
    
    img_rgb = np.array(img_resized.convert('RGB'))
    
    img_bgr = img_rgb[:, :, ::-1]
    
    img_normalized = img_bgr.astype('float32') / 255.0
    
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    predictions = model.predict(img_batch)
    return predictions

uploaded_file = st.file_uploader(
    "Select an MRI image (JPG, JPEG, PNG)...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
    with col2:
        st.subheader("📊 Analysis Results")
        st.write("Click the button below to initiate model prediction.")
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing scan patterns..."):
                predictions = preprocess_and_predict(image, model)
                
                class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
                
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = predictions[0][predicted_index] * 100

                if predicted_class == "No Tumor":
                    st.success(f"✅ **Predicted Class:** {predicted_class}")
                else:
                    st.error(f"⚠️ **Predicted Class:** {predicted_class}")
                
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                st.markdown("**Probability Distribution:**")
                for i, name in enumerate(class_names):
                    prob = float(predictions[0][i])
                    st.write(f"- {name}: `{prob*100:.1f}%`")
                    st.progress(prob)
else:
    st.info("👆 Please upload an MRI scan to begin.")