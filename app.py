
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# PREMIUM BLUE & WHITE CSS
st.markdown("""
<style>

.stApp {
    background: #F3F7FD;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

h1, h2, h3 {
    color: #123B70 !important;
}

p, label {
    color: #405570;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #123B70, #287BC1);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 5px 18px rgba(18,59,112,0.15);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #EAF4FF;
    font-size: 16px;
    margin: 0;
}

/* Cards */
.info-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #DCE8F5;
    box-shadow: 0 4px 14px rgba(30,80,130,0.07);
    margin-bottom: 15px;
}

.info-card h3 {
    color: #1769AA !important;
    margin-top: 0;
}

.result-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #DCE8F5;
    margin-top: 20px;
    text-align: center;
}

.normal {
    color: #16834A !important;
}

.pneumonia {
    color: #D64545 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #4A90E2;
    border-radius: 15px;
    padding: 18px;
}

/* Buttons */
.stButton > button {
    background: #1769AA;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #0D477F;
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #DCE8F5;
}

.footer {
    text-align: center;
    color: #64748B;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown("""
<div class="hero">
    <h1>🩺 AI Medical Imaging</h1>
    <p>Intelligent Chest X-ray Analysis using
    Deep Learning and Convolutional Neural Networks</p>
</div>
""", unsafe_allow_html=True)


# INTRODUCTION
st.markdown("""
<div class="info-card">
    <h3>Welcome to Pneumonia Detection</h3>
    Upload a chest X-ray image to get a prediction
    from our trained CNN model.
</div>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/pneumonia_model.keras"
    )


try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please check the models folder.")
    st.stop()


# IMAGE UPLOAD SECTION
st.markdown("## 📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Uploaded Image")
        st.image(
            image,
            caption="Chest X-ray Image",
            width=350
        )

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Image Details</h3>
            <p>Image uploaded successfully.</p>
            <p><b>Format:</b> Chest X-ray</p>
            <p><b>Model:</b> CNN Deep Learning</p>
            <p><b>Input Size:</b> 150 × 150</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔍 Analyze X-ray Image"):

        with st.spinner("AI model is analyzing the image..."):

            resized_image = image.resize((150, 150))
            image_array = np.array(resized_image)
            image_array = np.expand_dims(
                image_array, axis=0
            )

            prediction = model.predict(
                image_array, verbose=0
            )[0][0]

            if prediction >= 0.5:
                result = "PNEUMONIA"
                confidence = prediction * 100
                result_class = "pneumonia"
                st.session_state["prediction_result"] = result
            else:
                result = "NORMAL"
                confidence = (1 - prediction) * 100
                result_class = "normal"
                st.session_state["prediction_result"] = result

            st.session_state["confidence"] = confidence

    # SHOW RESULT
    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]
        confidence = st.session_state["confidence"]

        result_class = (
            "normal" if result == "NORMAL"
            else "pneumonia"
        )

        st.markdown(f"""
        <div class="result-card">
            <h3>AI Prediction Result</h3>
            <h1 class="{result_class}">{result}</h1>
            <p>Model Score: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(
            min(100, max(0, int(confidence)))
        )

        st.caption(
            "Model score is not a clinically calibrated probability."
        )


# MODEL PERFORMANCE
st.markdown("## 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "80%")
col2.metric("Precision", "80%")
col3.metric("Recall", "80%")
col4.metric("F1-Score", "80%")

st.caption(
    "Preliminary evaluation on 20 test images: "
    "10 NORMAL and 10 PNEUMONIA."
)


# HOW IT WORKS
st.markdown("## ⚙️ How It Works")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>1. Upload</h3>
        Upload a chest X-ray image.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>2. Analyze</h3>
        CNN model processes the image.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>3. Result</h3>
        View the model prediction and score.
    </div>
    """, unsafe_allow_html=True)


# DISCLAIMER
st.warning(
    "⚠️ Educational prototype only. This application "
    "is not a medical diagnosis tool and must not be "
    "used for clinical decisions. Consult a qualified "
    "healthcare professional for medical advice."
)


# FOOTER
st.markdown("""
<div class="footer">
    AI Medical Imaging | Pneumonia Detection Project<br>
    Developed using Python, TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)
