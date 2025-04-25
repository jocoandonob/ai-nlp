import streamlit as st
import base64

st.set_page_config(
    page_title="jocoNLP Portfolio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #7792E2, #5D7CE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        color: #7792E2;
    }
    
    .feature-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        border-left: 3px solid #7792E2;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 10px;
        background-color: #1E2130;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .stButton>button {
        background-color: #7792E2;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and display the logo from the SVG file
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Sidebar with logo
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
            <img src="data:image/svg+xml;base64,{get_image_as_base64('static/jocoNLP.svg')}" width="150">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("### Navigate")
    st.markdown("---")

# Main page content
st.markdown('<h1 class="main-header">Interactive NLP Portfolio</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="padding: 20px; border-radius: 10px; background-color: #1E2130; margin-bottom: 20px; border-left: 5px solid #7792E2;">
<h3 style="color: #7792E2;">Welcome to the jocoNLP Interactive Portfolio</h3>

<p>This portfolio showcases various NLP techniques and models, from basic concepts to advanced implementations.
You'll find interactive demos, explanations, and code examples for each technique.</p>

<h4 style="color: #7792E2; margin-top: 15px;">Featured NLP Techniques:</h4>
<ul>
  <li><strong>Tokenization</strong>: Breaking text into meaningful units</li>
  <li><strong>Embeddings</strong>: Word2Vec and GloVe for semantic word representations</li>
  <li><strong>Transformers</strong>: BERT and GPT models for advanced language understanding</li>
</ul>

<h4 style="color: #7792E2; margin-top: 15px;">Technologies Used:</h4>
<ul>
  <li><strong>spaCy</strong>: Industrial-strength NLP library</li>
  <li><strong>Hugging Face Transformers</strong>: State-of-the-art transformer models</li>
  <li><strong>NLTK</strong>: Natural Language Toolkit for linguistic processing</li>
</ul>

<h4 style="color: #7792E2; margin-top: 15px;">How to Use:</h4>
<p>Use the sidebar to navigate between different NLP concepts. Each page includes:</p>
<ul>
  <li>A brief explanation of the concept</li>
  <li>An interactive demo to try the technique</li>
  <li>Code examples showing how to implement it</li>
  <li>Visual representations of results where applicable</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Display feature cards with improved styling
st.markdown('<h2 class="sub-header">Featured Techniques</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Tokenization</h3>
        <p>Breaking text into words, phrases, symbols, or other meaningful elements.</p>
        <a href="Tokenization" style="color: #7792E2; text-decoration: none; font-weight: bold;">
            Explore Tokenization →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Word Embeddings</h3>
        <p>Representing words as numerical vectors that capture semantic meaning.</p>
        <a href="Embeddings" style="color: #7792E2; text-decoration: none; font-weight: bold;">
            Explore Embeddings →
        </a>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Transformer Models</h3>
        <p>State-of-the-art deep learning models for NLP tasks.</p>
        <a href="Transformers" style="color: #7792E2; text-decoration: none; font-weight: bold;">
            Explore Transformers →
        </a>
    </div>
    """, unsafe_allow_html=True)
