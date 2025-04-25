import streamlit as st
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import plotly.express as px
from utils import display_code_example

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def show():
    st.title("Tokenization in NLP")
    
    st.markdown("""
    ### What is Tokenization?
    
    Tokenization is the process of breaking down text into smaller units called tokens. 
    These tokens can be words, characters, subwords, or even sentences.
    
    Tokenization is usually the first step in any NLP pipeline and serves as the foundation for:
    - Text analysis
    - Language understanding
    - Feature extraction
    
    Let's explore different tokenization techniques and libraries.
    """)
    
    # Demo section
    st.header("Interactive Tokenization Demo")
    
    demo_text = st.text_area(
        "Enter text to tokenize",
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. It helps computers process and understand human language.",
        height=100
    )
    
    if demo_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Select Tokenization Method")
            tokenization_method = st.radio(
                "Choose a tokenization method:",
                ["Word Tokenization (NLTK)", "Sentence Tokenization (NLTK)", "spaCy Tokenization"]
            )
        
        with col2:
            st.subheader("Tokenization Result")
            
            if tokenization_method == "Word Tokenization (NLTK)":
                tokens = word_tokenize(demo_text)
                result = pd.DataFrame({"Token": tokens, "Index": list(range(len(tokens)))})
                st.table(result)
                
                # Visualization
                token_lengths = [len(token) for token in tokens]
                fig = px.bar(x=tokens, y=token_lengths, labels={'x': 'Token', 'y': 'Length'})
                fig.update_layout(title="Token Length Distribution")
                st.plotly_chart(fig)
                
            elif tokenization_method == "Sentence Tokenization (NLTK)":
                sentences = sent_tokenize(demo_text)
                for i, sent in enumerate(sentences):
                    st.markdown(f"**Sentence {i+1}:** {sent}")
                
                # Visualization
                sent_lengths = [len(sent.split()) for sent in sentences]
                fig = px.bar(
                    x=[f"Sentence {i+1}" for i in range(len(sentences))], 
                    y=sent_lengths,
                    labels={'x': 'Sentence', 'y': 'Word Count'}
                )
                fig.update_layout(title="Sentence Length (Word Count)")
                st.plotly_chart(fig)
                
            else:  # spaCy Tokenization
                nlp = load_spacy_model()
                doc = nlp(demo_text)
                
                # Create DataFrame for token information
                token_data = []
                for token in doc:
                    token_data.append({
                        "Token": token.text,
                        "Lemma": token.lemma_,
                        "POS": token.pos_,
                        "Is Stop Word": token.is_stop
                    })
                
                token_df = pd.DataFrame(token_data)
                st.dataframe(token_df)
                
                # Visualization of POS distribution
                pos_counts = token_df['POS'].value_counts().reset_index()
                pos_counts.columns = ['POS', 'Count']
                
                fig = px.pie(pos_counts, values='Count', names='POS', title='Distribution of Parts of Speech')
                st.plotly_chart(fig)
    
    # Code examples
    st.header("Code Examples")
    
    st.subheader("NLTK Word Tokenization")
    nltk_word_code = """
import nltk
from nltk.tokenize import word_tokenize

# Download punkt tokenizer if not already downloaded
nltk.download('punkt')

# Sample text
text = "Natural language processing (NLP) is a subfield of linguistics and AI."

# Tokenize into words
tokens = word_tokenize(text)
print(tokens)
# Output: ['Natural', 'language', 'processing', '(', 'NLP', ')', 'is', 'a', 'subfield', 'of', 'linguistics', 'and', 'AI', '.']
"""
    display_code_example(nltk_word_code)
    
    st.subheader("NLTK Sentence Tokenization")
    nltk_sent_code = """
from nltk.tokenize import sent_tokenize

# Sample text with multiple sentences
text = "Natural language processing is fascinating. It helps computers understand human language. This has many applications."

# Tokenize into sentences
sentences = sent_tokenize(text)
for i, sent in enumerate(sentences):
    print(f"Sentence {i+1}: {sent}")
"""
    display_code_example(nltk_sent_code)
    
    st.subheader("spaCy Tokenization")
    spacy_code = """
import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Natural language processing (NLP) is a subfield of linguistics and AI."

# Process text with spaCy
doc = nlp(text)

# Get tokens
tokens = [token.text for token in doc]
print(tokens)

# Get additional information about tokens
for token in doc:
    print(f"{token.text}\\t{token.lemma_}\\t{token.pos_}\\t{token.is_stop}")
"""
    display_code_example(spacy_code)

    # Additional information
    st.header("Tokenization Challenges")
    
    st.markdown("""
    ### Common Challenges in Tokenization
    
    1. **Handling Punctuation**: Deciding whether to keep or remove punctuation marks
    
    2. **Contractions**: Breaking down contractions like "don't" into "do" and "not"
    
    3. **Compound Words**: Deciding whether to split compound words like "ice cream"
    
    4. **Special Characters**: Handling emojis, hashtags, and special symbols
    
    5. **Multiple Languages**: Different languages have different tokenization rules
    
    6. **Domain-Specific Text**: Scientific, legal, or technical text may require specialized tokenization
    
    ### Advanced Tokenization Techniques
    
    - **Subword Tokenization**: Used by BERT, GPT, etc. (WordPiece, BPE, SentencePiece)
    - **Character-Level Tokenization**: Used in some deep learning models
    - **Regular Expression Based Tokenization**: For custom tokenization rules
    """)
