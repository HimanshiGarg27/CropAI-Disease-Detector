import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIG & DARK GLASS CSS
# ==========================================
st.set_page_config(page_title="CropAI | Smart Farming", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    /* Full-screen farm background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Dark semi-transparent box to make white text pop */
    .block-container, 
    [data-testid="stAppViewBlockContainer"], 
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.65) !important;
        padding: 3rem 2rem !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        max-width: 750px !important;
    }
    
    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force ALL text to be Bright White with a Black Shadow for readability */
    h1, h2, h3, p, label, span, div {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8) !important;
    }
    
    /* Beautiful Green Button Styling */
    .stButton>button {
        background-color: #2e7d32 !important; 
        color: white !important; 
        border-radius: 30px !important;
        border: 1px solid #4caf50 !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        text-shadow: none !important;
    }
    .stButton>button:hover {
        background-color: #4caf50 !important;
        box-shadow: 0 6px 12px rgba(76, 175, 80, 0.4) !important;
    }
    
    /* Tabs styling for dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT & TRANSLATIONS
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "language" not in st.session_state:
    st.session_state.language = "English"
if "messages" not in st.session_state:
    st.session_state.messages = []

text = {
    "English": {
        "login_title": "🌱 CropAI Edge",
        "login_sub": "Smart offline disease detection",
        "user": "Username",
        "pass": "Password",
        "login_btn": "Secure Login",
        "error": "Invalid credentials. Try admin / admin123",
        "logout": "Logout",
        "tab_scan": "📷 Scanner",
        "tab_details": "📋 Details",
        "tab_chat": "🤖 Chatbot",
        "scan_title": "Crop Disease Scanner",
        "scan_desc": "Take a clear picture of the leaf.",
        "camera": "Open Camera",
        "pred": "Diagnostic Result",
        "details_title": "Supported Diseases",
        "details_desc": "The AI is currently trained to detect the following conditions offline:",
        "chat_title": "CropCare AI Assistant",
        "chat_desc": "Ask me anything about farming, treatments, and crop health!",
        "chat_input": "Type your farming question here..."
    },
    "Hindi": {
        "login_title": "🌱 क्रॉप एआई (CropAI)",
        "login_sub": "स्मार्ट ऑफ़लाइन रोग पहचान",
        "user": "उपयोगकर्ता नाम (Username)",
        "pass": "पासवर्ड (Password)",
        "login_btn": "लॉग इन करें",
        "error": "अमान्य क्रेडेंशियल। admin / admin123 का प्रयास करें",
        "logout": "लॉग आउट",
        "tab_scan": "📷 स्कैनर",
        "tab_details": "📋 विवरण (Details)",
        "tab_chat": "🤖 चैटबॉट",
        "scan_title": "फसल रोग स्कैनर",
        "scan_desc": "पत्ती की एक स्पष्ट तस्वीर लें।",
        "camera": "कैमरा खोलें",
        "pred": "निदान परिणाम (Diagnostic Result)",
        "details_title": "समर्थित बीमारियां",
        "details_desc": "एआई वर्तमान में ऑफ़लाइन निम्नलिखित स्थितियों का पता लगाने के लिए प्रशिक्षित है:",
        "chat_title": "क्रॉपकेयर एआई सहायक",
        "chat_desc": "मुझसे खेती, उपचार और फसल के स्वास्थ्य के बारे में कुछ भी पूछें!",
        "chat_input": "अपना खेती से जुड़ा सवाल यहाँ लिखें..."
    }
}

t = text[st.session_state.language]

# ==========================================
# 3. LOAD MODELS (CACHED)
# ==========================================
@st.cache_data
def load_labels():
    with open("labels.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return ort.InferenceSession("mobilenetv3_cropcare.onnx")

try:
    labels = load_labels()
    session = load_model()
except Exception as e:
    st.error("⚠️ Error loading model assets. Make sure mobilenetv3_cropcare.onnx and labels.json exist.")
    labels = {}

# YOUR ACTUAL API KEY CONFIGURED HERE
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
chat_model = genai.GenerativeModel('gemini-3.6-flash')

# ==========================================
# 4. LOGIN PAGE WITH RELEVANT PICTURE
# ==========================================
if not st.session_state.logged_in:
    
    col_lang1, col_lang2, col_lang3 = st.columns([1, 1, 1])
    with col_lang2:
        st.selectbox("🌐 Language / भाषा", ["English", "Hindi"], key="lang_select", 
                     on_change=lambda: st.session_state.update(language=st.session_state.lang_select))
    
    st.markdown(f"<h1>{t['login_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{t['login_sub']}</p><br>", unsafe_allow_html=True)
    
    col_img1, col_img2, col_img3 = st.columns([1, 0.6, 1])
    with col_img2:
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892751.png", use_container_width=True)
    
    st.write("<br>", unsafe_allow_html=True)
    
    col_log1, col_log2, col_log3 = st.columns([1, 1.5, 1])
    with col_log2:
        username = st.text_input(t['user'], placeholder="admin")
        password = st.text_input(t['pass'], type="password", placeholder="••••••••")
        
        st.write("<br>", unsafe_allow_html=True)
        if st.button(t['login_btn']):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error(t['error'])

# ==========================================
# 5. MAIN APPLICATION
# ==========================================
else:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892751.png", width=120)
        st.markdown(f"### {t['login_title']}")
        st.selectbox("🌐 Language / भाषा", ["English", "Hindi"], key="lang_select_main", 
                     index=["English", "Hindi"].index(st.session_state.language),
                     on_change=lambda: st.session_state.update(language=st.session_state.lang_select_main))
        st.divider()
        if st.button(t['logout']):
            st.session_state.logged_in = False
            st.session_state.messages = []
            st.rerun()

    tab1, tab2, tab3 = st.tabs([t['tab_scan'], t['tab_details'], t['tab_chat']])

    with tab1:
        st.markdown(f"<h2>{t['scan_title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p>{t['scan_desc']}</p><br>", unsafe_allow_html=True)
        
        camera_image = st.camera_input(t['camera'])
        
        if camera_image is not None:
            with st.spinner("Analyzing..."):
                img = Image.open(camera_image).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                img_array = np.transpose(img_array, (2, 0, 1))
                input_tensor = np.expand_dims(img_array, axis=0)

                ort_inputs = {session.get_inputs()[0].name: input_tensor}
                output = session.run(None, ort_inputs)[0]
                pred_index = np.argmax(output)
                disease_name = labels.get(str(pred_index), "Unknown")
            
            st.success(f"**{t['pred']}:** {disease_name}", icon="✅")

    with tab2:
        st.markdown(f"<h2>{t['details_title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p>{t['details_desc']}</p><br>", unsafe_allow_html=True)
        
        if labels:
            disease_list = list(labels.values())
            for disease in disease_list:
                st.info(f"🌿 {disease}")

    with tab3:
        st.markdown(f"<h2>{t['chat_title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p>{t['chat_desc']}</p><br>", unsafe_allow_html=True)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(t['chat_input']):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                response = chat_model.generate_content(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"System Error: {str(e)}", icon="🚨")
