import streamlit as st
import pandas as pd
import time
from utils import load_spacy_model, get_token_info, visualize_tokens
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import base64

st.set_page_config(
    page_title="Tokenization - jocoNLP Portfolio",
    page_icon="🔤",
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
st.markdown('<h1 class="main-header">🔤 Tokenization</h1>', unsafe_allow_html=True)

st.markdown("""
### What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, characters, or subwords. 
It's a fundamental step in natural language processing that enables computers to analyze text at a granular level.

### Types of Tokenization
- **Word Tokenization**: Splitting text into words
- **Sentence Tokenization**: Splitting text into sentences
- **Subword Tokenization**: Breaking words into meaningful subunits
""")

# Interactive demo
st.header("Interactive Demo")

# Example text
example_text = "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

# Input text
text_input = st.text_area("Enter text to tokenize:", value=example_text, height=150)

# Load NLP models
nlp = load_spacy_model()

# Process text when button is clicked
if st.button("Process Text"):
    # Create tabs for different tokenization approaches
    tab1, tab2, tab3 = st.tabs(["spaCy Tokenization", "NLTK Tokenization", "Advanced Features"])
    
    with tab1:
        st.subheader("spaCy Tokenization")
        
        # Process text with spaCy
        with st.spinner("Processing with spaCy..."):
            doc = nlp(text_input)
            
        # Display tokens
        token_data = get_token_info(doc)
        df = pd.DataFrame(token_data)
        st.dataframe(df)
        
        # Visualize token information
        st.subheader("Token Visualization")
        fig = visualize_tokens(doc)
        st.plotly_chart(fig, use_container_width=True)
        
        # Named Entity Recognition
        st.subheader("Named Entity Recognition")
        html = displacy.render(doc, style="ent", options={"colors": {"PERSON": "#AA0000", "ORG": "#0000AA"}})
        st.markdown(html, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("NLTK Tokenization")
        
        # Word tokenization with NLTK
        with st.spinner("Processing with NLTK..."):
            nltk_word_tokens = word_tokenize(text_input)
            nltk_sent_tokens = sent_tokenize(text_input)
        
        # Display NLTK word tokens
        st.markdown("#### Word Tokens")
        nltk_tokens_df = pd.DataFrame({
            "Index": range(len(nltk_word_tokens)),
            "Token": nltk_word_tokens
        })
        st.dataframe(nltk_tokens_df)
        
        # Display NLTK sentence tokens
        st.markdown("#### Sentence Tokens")
        for i, sent in enumerate(nltk_sent_tokens):
            st.markdown(f"**Sentence {i+1}:** {sent}")
    
    with tab3:
        st.subheader("Advanced Tokenization Features")
        
        # Part-of-Speech Tagging
        st.markdown("#### Part-of-Speech Tagging")
        pos_df = pd.DataFrame({
            "Token": [token.text for token in doc],
            "POS": [token.pos_ for token in doc],
            "POS Explanation": [spacy.explain(token.pos_) for token in doc]
        })
        st.dataframe(pos_df)
        
        # Dependency Parsing
        st.subheader("Dependency Parsing")
        html = displacy.render(doc, style="dep", options={"distance": 120})
        st.markdown(html, unsafe_allow_html=True)

# Code examples
st.header("Code Examples")

st.markdown("#### spaCy Tokenization")
st.code("""
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Process text
text = "Natural Language Processing (NLP) is amazing!"
doc = nlp(text)

# Print tokens
for token in doc:
    print(token.text, token.pos_, token.dep_)
""", language="python")

st.markdown("#### NLTK Tokenization")
st.code("""
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure you have the necessary data
nltk.download('punkt')

# Example text
text = "Natural Language Processing (NLP) is amazing! It has many applications."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word tokenization
words = word_tokenize(text)
print("Words:", words)
""", language="python")

# Additional information
st.header("Use Cases for Tokenization")

st.markdown("""
### Common Applications of Tokenization

1. **Text Classification**: Breaking text into tokens for feature extraction
2. **Machine Translation**: Processing source and target language text
3. **Information Retrieval**: Indexing documents for search
4. **Sentiment Analysis**: Analyzing token patterns to determine sentiment
5. **Text Summarization**: Identifying important tokens for summary generation

### Why Tokenization Matters

Tokenization is the foundation of most NLP tasks. The quality of tokenization directly impacts the performance of downstream tasks. Different languages and domains may require specialized tokenization approaches.
""")
