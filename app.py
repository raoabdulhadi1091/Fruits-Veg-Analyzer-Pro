import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="CYBER-AGRI // Neural Vision Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Ultra-Futuristic CSS with Fixed Sidebar Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 0%, #090d16 0%, #020408 100%);
        font-family: 'Space Grotesk', sans-serif;
        color: #f3f4f6;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Force Sidebar Styling & Visibility */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b14 0%, #03050a 100%) !important;
        border-right: 1px solid rgba(52, 211, 153, 0.25) !important;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.8);
    }
    [data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }

    /* Cyber Hero Header */
    .cyber-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.12) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
    }
    .cyber-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #34d399, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .cyber-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Holographic Cards */
    .holo-card {
        background: rgba(13, 19, 33, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.7);
        margin-bottom: 20px;
    }

    /* Neon Cyber Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        border: 1px solid #34d399;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #065f46 100%);
        box-shadow: 0 0 35px rgba(52, 211, 153, 0.9);
        transform: translateY(-2px);
    }

    /* Glowing Badge */
    .cyber-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #34d399;
        color: #34d399 !important;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

DATA_LABELS = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chili pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
    'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'rhanish', 
    'soybeans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

IMG_HEIGHT = 180
IMG_WIDTH = 180

@st.cache_resource
def load_trained_model():
    return load_model("Fruits_Vegetables_model.keras")

# 3. Sidebar Navigation Panel (Wapas Sidebar mein set kar diya hai)
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #34d399; font-weight: 700; letter-spacing: 1px;'>⚡ NEURAL HUD</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin: 15px 0 25px 0;'><span class='cyber-badge'>ONLINE // SECURE</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎛️ SYSTEM CONTROLS")
    app_mode = st.radio("Select Matrix Mode", ["🔮 Live Neural Analyzer", "📊 Model Telemetry", "⚙️ Core Specs"])
    
    st.markdown("---")
    st.markdown("### 🔒 SECURITY PROTOCOL")
    st.info("Local TensorFlow weights loaded successfully with high-dimensional Tensor filters.")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.75rem; letter-spacing: 1px;'>DESIGNED FOR VIP // HADI</p>", unsafe_allow_html=True)

# 4. Main Cyber Banner Header
st.markdown("""
    <div class='cyber-banner'>
        <h1 class='cyber-title'>FRUIT & VEG NEURAL VISION PRO</h1>
        <p class='cyber-subtitle'>Ultra-Realistic Deep Learning Computer Vision Matrix</p>
    </div>
""", unsafe_allow_html=True)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"❌ Model load error! Ensure 'Fruits_Vegetables_model.keras' is in the root directory. Error: {e}")
    st.stop()

# 5. App Mode Logic
if app_mode == "🔮 Live Neural Analyzer":
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='holo-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 Input Specimen Stream")
        uploaded_file = st.file_uploader("Upload high-resolution sample image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Captured Target Specimen", use_container_width=True)
        else:
            st.markdown("<div style='text-align: center; padding: 60px; color: #64748b; border: 2px dashed rgba(255,255,255,0.1); border-radius: 14px;'>Awaiting visual feed...<br>Upload an image to activate neural scanner.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='holo-card'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Neural Matrix Diagnostics")
        
        if uploaded_file is not None:
            if st.button("🚀 INITIATE NEURAL SCAN"):
                with st.spinner("Processing convolutional tensor matrices..."):
                    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = tf.keras.utils.img_to_array(img_resized)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    
                    predicted_idx = np.argmax(score)
                    confidence = float(np.max(score) * 100)
                    
                    predicted_name = DATA_LABELS[predicted_idx] if predicted_idx < len(DATA_LABELS) else f"Class {predicted_idx}"
                
                if confidence > 75:
                    st.balloons()
                
                st.success("Target Identification Confirmed!")
                
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.12); border: 1px solid #34d399; padding: 22px; border-radius: 16px; margin-bottom: 20px; text-align: center; box-shadow: 0 0 20px rgba(16,185,129,0.2);'>
                        <span style='color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px;'>Classified Specimen Identity</span>
                        <h1 style='color: #34d399; margin: 8px 0 0 0; text-transform: uppercase; font-size: 2.6rem; text-shadow: 0 0 15px rgba(52,211,153,0.5);'>{predicted_name}</h1>
                        <h3 style='color: #38bdf8; margin-top: 10px;'>Confidence Index: {confidence:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🔬 Top 3 Matrix Probabilities")
                top_3_indices = np.argsort(score)[-3:][::-1]
                for idx in top_3_indices:
                    label = DATA_LABELS[idx] if idx < len(DATA_LABELS) else f"Class {idx}"
                    prob = float(score[idx] * 100)
                    st.markdown(f"**{label.upper()}** — `{prob:.2f}%`")
                    st.progress(int(prob))
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence HUD Meter", 'font': {'color': '#94a3b8', 'size': 14}},
                    number = {'font': {'color': '#ffffff', 'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#34d399"},
                        'bgcolor': "rgba(13, 19, 33, 0.9)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(52, 211, 153, 0.4)",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                            {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [80, 100], 'color': 'rgba(52, 211, 153, 0.2)'}
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=200,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'INITIATE NEURAL SCAN' to execute classification matrix.")
        else:
            st.markdown("<div style='text-align: center; padding: 70px; color: #64748b;'>Waiting for image feed...</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

elif app_mode == "📊 Model Telemetry":
    st.markdown("<div class='holo-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Architecture Telemetry & Specs")
    st.write("Ye deep learning model Convolutional Neural Networks (CNN) architecture par mabni hai jo fruit aur vegetable images ke intricate patterns ko extract karta hai.")
    st.markdown(f"- **Total Recognized Classes:** `{len(DATA_LABELS)}` categories")
    st.markdown("- **Tensor Resolution:** `180 x 180 pixels` (RGB)")
    st.markdown("- **Loss Optimizer:** `Adam` with Sparse Categorical Crossentropy")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='holo-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Core System Architecture")
    st.markdown("- **Backend Framework:** TensorFlow & Keras 3.x")
    st.markdown("- **Frontend Matrix:** Streamlit VIP Cyber-HUD")
    st.markdown("- **Data Visualization:** Plotly WebGL Interactive Graphics")
    st.markdown("- **Status:** Operational & Fully Optimized")
    st.markdown("</div>", unsafe_allow_html=True)