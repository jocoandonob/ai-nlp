import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import spacy
from utils import display_code_example

# Load spaCy model with word vectors
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_md')

def show():
    st.title("Word Embeddings in NLP")
    
    st.markdown("""
    ### What are Word Embeddings?
    
    Word embeddings are dense vector representations of words in a continuous vector space. Unlike one-hot encoding, 
    embeddings capture semantic relationships between words, where similar words have similar vectors.
    
    ### Key Embedding Techniques:
    
    - **Word2Vec**: Developed by Google, comes in two architectures:
      - Skip-gram: Predicts context words given a target word
      - CBOW (Continuous Bag of Words): Predicts a target word from context words
    
    - **GloVe (Global Vectors)**: Developed by Stanford, combines global matrix factorization and local context window methods
    
    - **FastText**: Developed by Facebook, extends Word2Vec by using subword information
    
    Let's explore these embedding techniques with some interactive demos.
    """)
    
    # Load spaCy model with word vectors
    nlp = load_spacy_model()
    
    # Demo section
    st.header("Interactive Word Embeddings Demo")
    
    demo_type = st.radio(
        "Choose a demo:",
        ["Word Similarity", "Word Analogies", "Visualize Word Embeddings"]
    )
    
    if demo_type == "Word Similarity":
        st.subheader("Word Similarity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            word1 = st.text_input("Enter first word:", "king")
        
        with col2:
            word2 = st.text_input("Enter second word:", "queen")
        
        if word1 and word2:
            if word1 in nlp.vocab and word2 in nlp.vocab:
                # Get word vectors
                vector1 = nlp(word1).vector
                vector2 = nlp(word2).vector
                
                # Calculate cosine similarity
                similarity = nlp(word1).similarity(nlp(word2))
                
                st.markdown(f"### Similarity between '{word1}' and '{word2}': {similarity:.4f}")
                
                # Visualize the vectors (reduced to 2D using PCA)
                if st.checkbox("Show vector visualization"):
                    # Get some related words for context
                    related_words = []
                    for word in [word1, word2, "man", "woman", "person", "royal", "leader"]:
                        if word in nlp.vocab:
                            related_words.append(word)
                    
                    # Get vectors for all words
                    vectors = np.array([nlp(word).vector for word in related_words])
                    
                    # Reduce to 2D
                    pca = PCA(n_components=2)
                    reduced_vectors = pca.fit_transform(vectors)
                    
                    # Create dataframe for plotting
                    df = pd.DataFrame({
                        'word': related_words,
                        'x': reduced_vectors[:, 0],
                        'y': reduced_vectors[:, 1],
                        'highlight': [word == word1 or word == word2 for word in related_words]
                    })
                    
                    # Plot
                    fig = px.scatter(
                        df, x='x', y='y', text='word',
                        color='highlight', color_discrete_map={True: 'red', False: 'blue'},
                        title="2D Projection of Word Vectors"
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig)
            else:
                st.error(f"One or both words are not in the vocabulary. Please try different words.")
    
    elif demo_type == "Word Analogies":
        st.subheader("Word Analogies")
        st.markdown("Find the fourth word in an analogy: A is to B as C is to ?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            word_a = st.text_input("A:", "king")
        
        with col2:
            word_b = st.text_input("B:", "queen")
        
        with col3:
            word_c = st.text_input("C:", "man")
        
        if word_a and word_b and word_c:
            if word_a in nlp.vocab and word_b in nlp.vocab and word_c in nlp.vocab:
                # Calculate analogy: vector_b - vector_a + vector_c
                vec_a = nlp(word_a).vector
                vec_b = nlp(word_b).vector
                vec_c = nlp(word_c).vector
                
                result_vector = vec_b - vec_a + vec_c
                
                # Find most similar words to the result vector
                words = []
                similarities = []
                
                # Get top 5 most similar words
                ms = nlp.vocab.vectors.most_similar(
                    result_vector.reshape(1, result_vector.shape[0]),
                    n=10
                )
                
                for word_id, score in zip(ms[0][0], ms[2][0]):
                    word = nlp.vocab.strings[word_id]
                    # Filter out the input words and lowercase/uppercase variants
                    if word.lower() not in [word_a.lower(), word_b.lower(), word_c.lower()]:
                        words.append(word)
                        similarities.append(score)
                        if len(words) >= 5:  # Limit to top 5
                            break
                
                st.markdown(f"### {word_a} is to {word_b} as {word_c} is to:")
                
                result_df = pd.DataFrame({
                    "Word": words,
                    "Similarity Score": similarities
                })
                
                st.table(result_df)
                
                # Create a bar chart of similarity scores
                fig = px.bar(
                    result_df, 
                    x='Word', 
                    y='Similarity Score',
                    title=f"Top matches for: {word_a} : {word_b} :: {word_c} : ?"
                )
                st.plotly_chart(fig)
            else:
                st.error("One or more words are not in the vocabulary. Please try different words.")
    
    elif demo_type == "Visualize Word Embeddings":
        st.subheader("Visualize Word Embeddings")
        
        # Input for custom words
        custom_words_input = st.text_input(
            "Enter words to visualize (comma-separated):",
            "king, queen, man, woman, child, person, computer, phone, car, tree"
        )
        
        if custom_words_input:
            # Parse input and filter words in vocabulary
            custom_words = [word.strip() for word in custom_words_input.split(",")]
            valid_words = [word for word in custom_words if word in nlp.vocab]
            
            if len(valid_words) < 2:
                st.error("Please enter at least 2 words that are in the vocabulary.")
            else:
                # Get vectors
                word_vectors = [nlp(word).vector for word in valid_words]
                
                # Reduce dimensions for visualization
                pca = PCA(n_components=3)
                reduced_vectors = pca.fit_transform(word_vectors)
                
                # Create dataframe for plotting
                df = pd.DataFrame({
                    'word': valid_words,
                    'x': reduced_vectors[:, 0],
                    'y': reduced_vectors[:, 1],
                    'z': reduced_vectors[:, 2] if reduced_vectors.shape[1] > 2 else np.zeros(len(valid_words))
                })
                
                # Create visualization
                view_type = st.radio("Select visualization type:", ["2D Plot", "3D Plot"])
                
                if view_type == "2D Plot":
                    fig = px.scatter(
                        df, x='x', y='y', text='word',
                        title="2D Projection of Word Embeddings"
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig)
                else:  # 3D Plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=df['x'],
                        y=df['y'],
                        z=df['z'],
                        mode='markers+text',
                        text=df['word'],
                        marker=dict(
                            size=10,
                            color=list(range(len(df))),
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        textposition='top center'
                    )])
                    fig.update_layout(
                        title="3D Projection of Word Embeddings",
                        scene=dict(
                            xaxis_title='PCA Component 1',
                            yaxis_title='PCA Component 2',
                            zaxis_title='PCA Component 3'
                        )
                    )
                    st.plotly_chart(fig)
                
                # Show explanation of missing words if any
                if len(valid_words) < len(custom_words):
                    missing_words = set(custom_words) - set(valid_words)
                    st.warning(f"The following words were not found in the vocabulary: {', '.join(missing_words)}")
    
    # Code examples
    st.header("Code Examples")
    
    st.subheader("Word2Vec with Gensim")
    gensim_code = """
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Train a Word2Vec model on your corpus
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]  # Normally this would be your tokenized corpus
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar("dog", topn=5)
print(similar_words)

# Perform analogy tasks
result = model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
print(f"king - man + woman = {result[0][0]}")

# Save and load model
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")
"""
    display_code_example(gensim_code)
    
    st.subheader("Using Pre-trained GloVe Vectors")
    glove_code = """
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Convert GloVe format to Word2Vec format (only need to do this once)
glove_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec"
glove2word2vec(glove_file, word2vec_output_file)

# Load the pre-trained vectors
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Find similar words
similar_words = glove_model.most_similar("computer", topn=5)
print(similar_words)

# Get vector for a word
word_vector = glove_model["apple"]
print(f"Vector shape: {word_vector.shape}")

# Calculate similarity between words
similarity = glove_model.similarity("king", "queen")
print(f"Similarity between 'king' and 'queen': {similarity:.4f}")
"""
    display_code_example(glove_code)
    
    st.subheader("Using Word Vectors in spaCy")
    spacy_embeddings_code = """
import spacy

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")  # Medium model with word vectors

# Get vector for a word
word = nlp("computer")
word_vector = word.vector
print(f"Vector dimension: {word_vector.shape}")

# Calculate similarity between words
word1 = nlp("king")
word2 = nlp("queen")
similarity = word1.similarity(word2)
print(f"Similarity between 'king' and 'queen': {similarity:.4f}")

# Find most similar words (requires a custom function with spaCy)
def most_similar(word, topn=5):
    word_id = nlp.vocab.strings[word]
    word_vector = nlp.vocab.vectors[word_id]
    
    # Find most similar words by vector
    most_similar = nlp.vocab.vectors.most_similar(
        word_vector.reshape(1, word_vector.shape[0]),
        n=topn+1  # +1 because the word itself will be included
    )
    
    results = []
    for word_id, score in zip(most_similar[0][0], most_similar[2][0]):
        word = nlp.vocab.strings[word_id]
        if word != word:  # Skip the input word
            results.append((word, score))
    
    return results[:topn]

similar_words = most_similar("computer")
print(similar_words)
"""
    display_code_example(spacy_embeddings_code)
    
    # Additional information
    st.header("Word Embeddings: Technical Details")
    
    st.markdown("""
    ### How Word Embeddings Work
    
    Word embeddings capture semantic meaning by placing words with similar contexts close to each other in vector space.
    This allows for fascinating properties:
    
    - **Semantic Similarity**: Words with similar meanings have similar vectors
    - **Semantic Arithmetic**: Vector operations reveal relationships (e.g., king - man + woman ≈ queen)
    - **Cross-lingual Alignment**: Embeddings can be aligned across languages
    
    ### Common Embedding Dimensions
    
    - **Low Dimensional** (50-100): Good for small datasets, faster but less expressive
    - **Medium Dimensional** (200-300): Good balance for most applications
    - **High Dimensional** (500+): Better for capturing nuanced relationships in large corpora
    
    ### Contextual vs. Static Embeddings
    
    - **Static Embeddings** (like Word2Vec, GloVe): One vector per word regardless of context
    - **Contextual Embeddings** (like BERT, ELMo): Different vectors for the same word in different contexts
    
    ### Evaluation Methods
    
    - **Word Similarity**: Testing on human-annotated word pair similarity datasets
    - **Word Analogies**: Testing on analogy tasks (a:b::c:?)
    - **Extrinsic Evaluation**: Measuring performance on downstream tasks
    """)
