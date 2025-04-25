import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import base64
from utils import load_spacy_model, load_hf_model, get_sentence_embeddings, plot_similarity_heatmap, is_transformers_available
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Import torch if available
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

st.set_page_config(
    page_title="Word Embeddings - jocoNLP Portfolio",
    page_icon="🔠",
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
    
    .info-box {
        background-color: #1E2130;
        border-left: 5px solid #7792E2;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .warning-box {
        background-color: #332b17;
        border-left: 5px solid #f0ad4e;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
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
st.markdown('<h1 class="main-header">🔠 Word Embeddings</h1>', unsafe_allow_html=True)

# Check for required libraries
transformers_available = is_transformers_available()
if not transformers_available or not torch_available:
    st.markdown("""
    <div class="warning-box">
        <h3 style="color: #f0ad4e;">⚠️ Limited Functionality Notice</h3>
        <p>Some required libraries are not installed, which limits the functionality of this page.</p>
        <ul>
            <li>Transformers library available: {}</li>
            <li>PyTorch available: {}</li>
        </ul>
        <p>Some demos will show simplified versions or may not be available.</p>
    </div>
    """.format(transformers_available, torch_available), unsafe_allow_html=True)

st.markdown("""
### Understanding Word Embeddings

Word embeddings are dense vector representations of words that capture semantic meaning. Unlike simple one-hot encodings, 
these vectors position similar words closer together in a multi-dimensional space, allowing algorithms to understand relationships between words.

### Popular Embedding Techniques

1. **Word2Vec**: Learn word associations from a large corpus of text
   - Skip-gram: Predict context words given a target word
   - Continuous Bag of Words (CBOW): Predict target word from context words

2. **GloVe (Global Vectors)**: Combine global matrix factorization and local context window methods

3. **Contextual Embeddings**: Generate different vectors for the same word based on context (like BERT)
""")

# Interactive demo
st.header("Interactive Demo")

# Example sentences
example_sentences = """
Machine learning is fascinating.
Deep learning is a subset of machine learning.
AI and ML are transforming many industries.
"""

# Input text
sentences_input = st.text_area("Enter sentences (one per line):", value=example_sentences, height=150)

# Process sentences
sentences = [s for s in sentences_input.split('\n') if s.strip()]

embedding_model = st.selectbox(
    "Select embedding model:",
    ["spaCy (en_core_web_md)", "BERT (bert-base-uncased)"]
)

if st.button("Generate Embeddings"):
    if not sentences:
        st.error("Please enter at least one sentence.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Word Vectors", "Sentence Similarity", "Dimensionality Reduction"])
        
        with tab1:
            st.subheader("Word Vectors")
            
            # Update progress
            for i in range(33):
                progress_bar.progress(i)
                status_text.text(f"Loading models... {i}%")
                time.sleep(0.01)
            
            if embedding_model.startswith("spaCy"):
                # Use spaCy for word vectors
                nlp = load_spacy_model("en_core_web_md")
                
                # Update progress
                for i in range(33, 66):
                    progress_bar.progress(i)
                    status_text.text(f"Processing text... {i}%")
                    time.sleep(0.01)
                
                # Get sample words from the sentences
                all_words = []
                for sentence in sentences:
                    doc = nlp(sentence)
                    # Only include content words with vectors
                    for token in doc:
                        if token.has_vector and not token.is_stop and token.is_alpha:
                            all_words.append(token.text)
                
                # Select a few words to display
                sample_words = list(set(all_words))[:10]  # Limit to 10 unique words
                
                # Display word vectors
                word_vectors = {}
                for word in sample_words:
                    word_vectors[word] = nlp(word).vector
                
                # Display the first few dimensions of each vector
                vector_display = []
                for word, vector in word_vectors.items():
                    vector_display.append({
                        "Word": word,
                        "Vector Preview (first 5 dimensions)": str(vector[:5]),
                        "Vector Length": len(vector)
                    })
                
                st.dataframe(pd.DataFrame(vector_display))
                
                # Display similarity between words
                st.subheader("Word Similarity")
                
                if len(sample_words) >= 2:
                    word_similarity = []
                    for i, word1 in enumerate(sample_words):
                        for word2 in sample_words[i+1:]:
                            similarity = nlp(word1).similarity(nlp(word2))
                            word_similarity.append({
                                "Word 1": word1,
                                "Word 2": word2,
                                "Similarity": round(similarity, 3)
                            })
                    
                    st.dataframe(pd.DataFrame(word_similarity).sort_values(by="Similarity", ascending=False))
                else:
                    st.info("Need at least two content words to compare similarity.")
            
            else:  # BERT
                # Use BERT for word embeddings
                tokenizer, model = load_hf_model("bert-base-uncased")
                
                # Update progress
                for i in range(33, 66):
                    progress_bar.progress(i)
                    status_text.text(f"Processing with BERT... {i}%")
                    time.sleep(0.01)
                
                # Get sample words from the sentences
                all_words = []
                for sentence in sentences:
                    words = sentence.split()
                    all_words.extend([w for w in words if len(w) > 2])
                
                # Select a few words to display
                sample_words = list(set(all_words))[:10]  # Limit to 10 unique words
                
                # Get word embeddings
                word_vectors = {}
                for word in sample_words:
                    inputs = tokenizer(word, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Use the last hidden state of the first token (CLS) as the word embedding
                    word_vectors[word] = outputs.last_hidden_state[0, 1].numpy()  # Skip CLS token
                
                # Display the first few dimensions of each vector
                vector_display = []
                for word, vector in word_vectors.items():
                    vector_display.append({
                        "Word": word,
                        "Vector Preview (first 5 dimensions)": str(vector[:5]),
                        "Vector Length": len(vector)
                    })
                
                st.dataframe(pd.DataFrame(vector_display))
                
                # Calculate and display similarity between words
                st.subheader("Word Similarity")
                
                if len(sample_words) >= 2:
                    word_similarity = []
                    vectors = np.array(list(word_vectors.values()))
                    
                    # Compute cosine similarity
                    similarity_matrix = cosine_similarity(vectors)
                    
                    for i, word1 in enumerate(sample_words):
                        for j, word2 in enumerate(sample_words[i+1:], i+1):
                            similarity = similarity_matrix[i, j]
                            word_similarity.append({
                                "Word 1": word1,
                                "Word 2": word2,
                                "Similarity": round(similarity, 3)
                            })
                    
                    st.dataframe(pd.DataFrame(word_similarity).sort_values(by="Similarity", ascending=False))
                else:
                    st.info("Need at least two words to compare similarity.")
        
        with tab2:
            st.subheader("Sentence Similarity")
            
            # Update progress
            for i in range(66, 90):
                progress_bar.progress(i)
                status_text.text(f"Calculating sentence similarities... {i}%")
                time.sleep(0.01)
            
            if len(sentences) < 2:
                st.info("Need at least two sentences to compare similarity.")
            else:
                if embedding_model.startswith("spaCy"):
                    # Use spaCy for sentence embeddings
                    nlp = load_spacy_model("en_core_web_md")
                    
                    # Get sentence embeddings
                    sentence_vectors = [nlp(sentence).vector for sentence in sentences]
                    sentence_vectors = np.array(sentence_vectors)
                    
                    # Calculate similarity matrix
                    similarity_matrix = cosine_similarity(sentence_vectors)
                    
                elif embedding_model.startswith("BERT"):
                    # Use BERT for sentence embeddings
                    tokenizer, model = load_hf_model("bert-base-uncased")
                    
                    # Get sentence embeddings
                    embedded_sentences = get_sentence_embeddings(model, tokenizer, sentences)
                    sentence_vectors = embedded_sentences.numpy()
                    
                    # Calculate similarity matrix
                    similarity_matrix = cosine_similarity(sentence_vectors)
                
                # Display similarity heatmap
                fig = plot_similarity_heatmap(
                    [f"Sentence {i+1}" for i in range(len(sentences))], 
                    similarity_matrix
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the actual sentences for reference
                st.subheader("Sentences")
                for i, sentence in enumerate(sentences):
                    st.markdown(f"**Sentence {i+1}:** {sentence}")
        
        with tab3:
            st.subheader("Dimensionality Reduction (PCA)")
            
            # Update progress
            for i in range(90, 100):
                progress_bar.progress(i)
                status_text.text(f"Generating visualizations... {i}%")
                time.sleep(0.01)
            
            if embedding_model.startswith("spaCy"):
                nlp = load_spacy_model("en_core_web_md")
                
                # Get words from all sentences
                all_words = []
                for sentence in sentences:
                    doc = nlp(sentence)
                    for token in doc:
                        if token.has_vector and token.is_alpha and not token.is_stop:
                            all_words.append(token.text)
                
                # Get unique words
                unique_words = list(set(all_words))
                
                if len(unique_words) >= 5:
                    # Get word vectors
                    word_vectors = [nlp(word).vector for word in unique_words]
                    
                    # Reduce dimensions with PCA
                    pca = PCA(n_components=2)
                    reduced_vectors = pca.fit_transform(word_vectors)
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'word': unique_words,
                        'x': reduced_vectors[:, 0],
                        'y': reduced_vectors[:, 1]
                    })
                    
                    # Plot reduced vectors
                    fig = px.scatter(
                        plot_df, x='x', y='y', text='word',
                        title="2D PCA Projection of Word Embeddings"
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 5 unique content words for PCA visualization.")
            
            else:  # BERT
                tokenizer, model = load_hf_model("bert-base-uncased")
                
                # Get words from all sentences
                all_words = []
                for sentence in sentences:
                    words = sentence.split()
                    # Filter out very short words
                    all_words.extend([w for w in words if len(w) > 2])
                
                # Get unique words
                unique_words = list(set(all_words))[:20]  # Limit to 20 words for clarity
                
                if len(unique_words) >= 5:
                    # Get word embeddings
                    word_vectors = []
                    for word in unique_words:
                        inputs = tokenizer(word, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model(**inputs)
                        # Use the last hidden state of the first token (CLS) as the word embedding
                        word_vectors.append(outputs.last_hidden_state[0, 1].numpy())
                    
                    # Reduce dimensions with PCA
                    pca = PCA(n_components=2)
                    reduced_vectors = pca.fit_transform(word_vectors)
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'word': unique_words,
                        'x': reduced_vectors[:, 0],
                        'y': reduced_vectors[:, 1]
                    })
                    
                    # Plot reduced vectors
                    fig = px.scatter(
                        plot_df, x='x', y='y', text='word',
                        title="2D PCA Projection of Word Embeddings"
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 5 unique words for PCA visualization.")
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Processing complete!")

# Code examples
st.header("Code Examples")

st.markdown("#### Using Word2Vec with Gensim")
st.code("""
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Sample sentences
sentences = [
    ["machine", "learning", "is", "fascinating"],
    ["deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ["AI", "and", "ML", "are", "transforming", "industries"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar('learning', topn=5)
print("Words similar to 'learning':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

# Calculate word similarity
similarity = model.wv.similarity('machine', 'learning')
print(f"Similarity between 'machine' and 'learning': {similarity:.4f}")

# Get word vector
vector = model.wv['learning']
print(f"Vector for 'learning' (first 5 dimensions): {vector[:5]}")
""", language="python")

st.markdown("#### Using BERT for Contextual Embeddings")
st.code("""
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences showing different contexts of the word 'bank'
sentences = [
    "I need to go to the bank to deposit money.",
    "The river bank was eroding after the flood."
]

# Process both sentences
for sentence in sentences:
    # Tokenize and convert to tensor
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings for each token
    token_embeddings = outputs.last_hidden_state
    
    # Print the tokens and the shape of their embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Embedding shape: {token_embeddings.shape}")
    
    # Find the position of the word 'bank' in the tokenized input
    try:
        bank_index = tokens.index('bank')
        print(f"Embedding for 'bank' (first 5 dims): {token_embeddings[0, bank_index][:5]}")
    except ValueError:
        print("Word 'bank' not found as a single token")
""", language="python")

# Further explanation
st.header("Additional Information")

st.markdown("""
### How Word Embeddings Work

Word embeddings map words to points in a vector space such that semantically similar words are mapped to nearby points. This is achieved by training models on large corpora of text and leveraging the distributional hypothesis: words that occur in similar contexts tend to have similar meanings.

### Comparison of Embedding Techniques

| Technique | Contextual | Training Method | Key Features |
|-----------|------------|----------------|--------------|
| Word2Vec  | No | Predict context from word (skip-gram) or word from context (CBOW) | Fast, captures semantic relationships |
| GloVe     | No | Matrix factorization of co-occurrence statistics | Combines global and local context |
| BERT Embeddings | Yes | Masked language modeling and next sentence prediction | Captures word meaning based on surrounding context |

### Applications of Word Embeddings

- **Semantic Search**: Finding documents with similar meaning, not just keyword matching
- **Sentiment Analysis**: Understanding the emotional tone of text
- **Machine Translation**: Mapping words between languages
- **Document Classification**: Categorizing text based on meaning
- **Recommendation Systems**: Suggesting similar items based on text descriptions
""")
