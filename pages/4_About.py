import streamlit as st
import base64

st.set_page_config(
    page_title="About - jocoNLP Portfolio",
    page_icon="ℹ️",
    layout="wide"
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
    
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #7792E2, #5D7CE2);
    }
    
    .about-section {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 30px;
        margin-bottom: 20px;
        border-left: 5px solid #7792E2;
    }
    
    .resource-card {
        background-color: #1E2130;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid #7792E2;
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

# Main page title
st.markdown('<h1 class="main-header">ℹ️ About jocoNLP</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="about-section">
    <h2 style="color: #7792E2;">Project Overview</h2>
    <p>This interactive Natural Language Processing (NLP) portfolio showcases various techniques from basic tokenization 
    to advanced transformer models. It's designed to provide both educational content and hands-on demonstrations of 
    NLP concepts.</p>
    
    <h3 style="color: #7792E2; margin-top: 20px;">Features</h3>
    <ul>
        <li><strong>Interactive Demos:</strong> Try out different NLP techniques with your own text inputs</li>
        <li><strong>Visual Representations:</strong> See the results of NLP processes through charts and visualizations</li>
        <li><strong>Code Examples:</strong> Learn how to implement these techniques in your own projects</li>
        <li><strong>Explanations:</strong> Understand the concepts behind each NLP approach</li>
    </ul>
    
    <h3 style="color: #7792E2; margin-top: 20px;">Technologies Used</h3>
    <ul>
        <li><strong>Streamlit:</strong> For the interactive web application</li>
        <li><strong>spaCy:</strong> For linguistic processing tasks</li>
        <li><strong>NLTK:</strong> For additional NLP functionality</li>
        <li><strong>Hugging Face Transformers:</strong> For state-of-the-art transformer models</li>
        <li><strong>Plotly:</strong> For interactive visualizations</li>
        <li><strong>Pandas:</strong> For data manipulation and display</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# NLP Techniques section with improved styling
st.markdown('<h2 class="sub-header">NLP Techniques Covered</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Tokenization</h3>
        <ul>
            <li>Word and sentence tokenization</li>
            <li>Part-of-speech tagging</li>
            <li>Named entity recognition</li>
            <li>Dependency parsing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Word Embeddings</h3>
        <ul>
            <li>Static embeddings (Word2Vec, GloVe)</li>
            <li>Contextual embeddings (BERT)</li>
            <li>Semantic similarity</li>
            <li>Dimensionality reduction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #7792E2;">Transformer Models</h3>
        <ul>
            <li>Text classification</li>
            <li>Named entity recognition</li>
            <li>Question answering</li>
            <li>Text summarization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Learning Resources section with improved styling
st.markdown('<h2 class="sub-header">Learning Resources</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="resource-card">
        <h3 style="color: #7792E2;">Books</h3>
        <ul>
            <li>"Speech and Language Processing" by Dan Jurafsky and James H. Martin</li>
            <li>"Natural Language Processing with Python" by Bird, Klein, & Loper</li>
            <li>"Transformers for NLP" by Denis Rothman</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="resource-card">
        <h3 style="color: #7792E2;">Online Courses</h3>
        <ul>
            <li><a href="https://web.stanford.edu/class/cs224n/" target="_blank">Stanford CS224N: NLP with Deep Learning</a></li>
            <li><a href="https://huggingface.co/course" target="_blank">Hugging Face Course</a></li>
            <li><a href="https://course.spacy.io" target="_blank">spaCy Course</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="resource-card">
        <h3 style="color: #7792E2;">Libraries Documentation</h3>
        <ul>
            <li><a href="https://spacy.io/usage" target="_blank">spaCy Documentation</a></li>
            <li><a href="https://huggingface.co/docs/transformers/index" target="_blank">Hugging Face Transformers</a></li>
            <li><a href="https://www.nltk.org" target="_blank">NLTK Documentation</a></li>
            <li><a href="https://streamlit.io/docs" target="_blank">Streamlit Documentation</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="resource-card">
        <h3 style="color: #7792E2;">Future Enhancements</h3>
        <ul>
            <li>Advanced NLP tasks like machine translation</li>
            <li>Fine-tuning demonstrations for custom models</li>
            <li>Detailed performance metrics and evaluations</li>
            <li>Additional transformer architectures and models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Contact information
st.markdown('<h2 class="sub-header">Contact Information</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="about-section">
    <p>If you have questions, suggestions, or feedback about this portfolio, feel free to reach out:</p>
    
    <ul>
        <li><strong>Email:</strong> contact@jocoNLP.ai</li>
        <li><strong>GitHub:</strong> <a href="https://github.com/jocoNLP/nlp-portfolio" target="_blank">github.com/jocoNLP/nlp-portfolio</a></li>
        <li><strong>Twitter:</strong> @jocoNLP</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Acknowledgements
st.markdown('<h2 class="sub-header">Acknowledgements</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="about-section">
    <p>This project was built using several open-source tools and pre-trained models. Special thanks to:</p>
    
    <ul>
        <li>The <strong>Hugging Face</strong> team for their Transformers library</li>
        <li>The <strong>spaCy</strong> team for their industrial-strength NLP library</li>
        <li>The <strong>Streamlit</strong> team for their app framework</li>
        <li>The creators and contributors of the various pre-trained models used in this portfolio</li>
    </ul>
    
    <p style="margin-top: 15px;">All models used in this portfolio are lightweight versions of larger models, optimized for demonstration purposes.</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown('<h2 class="sub-header">Disclaimer</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="about-section">
    <p>This portfolio is for educational and demonstration purposes only. The performance of the models used here may not reflect the capabilities of larger, more specialized models. For production applications, consider using more robust models and conducting thorough evaluations.</p>
</div>
""", unsafe_allow_html=True)

# Version info
st.markdown("<div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 50px;'>jocoNLP Portfolio v1.0.0 | Last Updated: April 2025</div>", unsafe_allow_html=True)
