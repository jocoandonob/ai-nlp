import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import time
import importlib

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy models
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        # If model not available, download it
        spacy.cli.download(model_name)
        return spacy.load(model_name)

# Check if transformers library is available
def is_transformers_available():
    try:
        importlib.import_module('transformers')
        return True
    except ImportError:
        return False

# Load Hugging Face models
@st.cache_resource
def load_hf_model(model_name, task=None):
    if not is_transformers_available():
        st.error("Transformers library is not installed. Some features are unavailable.")
        return None, None
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        if task:
            return pipeline(task, model=model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_token_info(doc):
    """Extract token information from spaCy doc"""
    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "shape": token.shape_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        })
    return tokens

def visualize_tokens(doc):
    """Create a visualization of tokens using Plotly"""
    tokens = [token.text for token in doc]
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Index', 'Token', 'POS Tag', 'Dependency'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[
            list(range(len(tokens))),
            [token.text for token in doc],
            [token.pos_ for token in doc],
            [token.dep_ for token in doc]
        ],
            fill_color='lavender',
            align='left')
    )])
    
    fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))
    return fig

def get_sentence_embeddings(model, tokenizer, sentences):
    """Get sentence embeddings using a Hugging Face model"""
    # Tokenize sentences and convert to tensor
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Get model output
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use the mean of the last hidden state as the sentence embedding
    embeddings = model_output.last_hidden_state.mean(dim=1)
    
    return embeddings

def plot_similarity_heatmap(sentences, similarity_matrix):
    """Create a heatmap visualization of sentence similarities"""
    fig = px.imshow(
        similarity_matrix,
        x=sentences,
        y=sentences,
        color_continuous_scale='Viridis',
        labels=dict(color="Cosine Similarity")
    )
    fig.update_layout(
        xaxis=dict(side="top"),
        margin=dict(l=5, r=5, b=10, t=10)
    )
    return fig

def display_transformer_attention(tokens, attention_weights, layer=0, head=0):
    """Create a heatmap visualization of transformer attention weights"""
    # Extract attention weights for the specified layer and head
    attention = attention_weights[layer][head].detach().numpy()
    
    # Create a heatmap
    fig = px.imshow(
        attention,
        x=tokens,
        y=tokens,
        color_continuous_scale='Viridis',
        labels=dict(color="Attention Weight")
    )
    fig.update_layout(
        xaxis=dict(side="top"),
        margin=dict(l=5, r=5, b=10, t=10)
    )
    return fig

def display_code(code):
    """Display code with syntax highlighting using st.code"""
    st.code(code, language="python")

def process_with_progress(process_func, *args, **kwargs):
    """Execute a function with a progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing with progress updates
    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f"Processing: {i}%")
        time.sleep(0.01)
    
    result = process_func(*args, **kwargs)
    status_text.text("Processing complete!")
    
    return result
