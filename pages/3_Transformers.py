import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import base64
from utils import load_hf_model, is_transformers_available

# Import torch if available
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

st.set_page_config(
    page_title="Transformers - jocoNLP Portfolio",
    page_icon="🤖",
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
    
    .error-box {
        background-color: #2c1618;
        border-left: 5px solid #d9534f;
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
st.markdown('<h1 class="main-header">🤖 Transformer Models</h1>', unsafe_allow_html=True)

st.markdown("""
### The Revolution of Transformer Models

Transformer models have revolutionized NLP since their introduction in the paper "Attention is All You Need" (2017). 
They rely on a mechanism called self-attention, which allows the model to weigh the importance of different words in a sentence
when processing each word, regardless of their position.

### Key Transformer Models

1. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Bidirectional context understanding
   - Pre-trained on masked language modeling and next sentence prediction
   - Excellent for understanding tasks like classification and NER

2. **GPT (Generative Pre-trained Transformer)**
   - Autoregressive language model
   - Pre-trained on next token prediction
   - Strong at text generation and completion tasks

### Hugging Face Transformers

The Hugging Face Transformers library provides easy access to state-of-the-art transformer models.
In this demo, we'll explore different applications of transformer models using small, efficient models
from the Hugging Face Hub.
""")

# Interactive demo
st.header("Interactive Demo")

# Create tabs for different transformer applications
tab1, tab2, tab3, tab4 = st.tabs([
    "Text Classification", 
    "Named Entity Recognition", 
    "Question Answering",
    "Text Summarization"
])

with tab1:
    st.subheader("Sentiment Analysis with BERT")
    
    st.markdown("""
    This demo uses a fine-tuned BERT model to classify text sentiment.
    The model categorizes text as positive, negative, or neutral.
    """)
    
    # Example text
    example_reviews = [
        "I absolutely loved this product! It exceeded all my expectations.",
        "The service was terrible and the staff was rude.",
        "The movie was okay, not great but not terrible either."
    ]
    
    selected_example = st.selectbox(
        "Choose an example or write your own:", 
        ["Custom input"] + example_reviews
    )
    
    if selected_example == "Custom input":
        text_input = st.text_area("Enter text for sentiment analysis:", height=100)
    else:
        text_input = selected_example
    
    if st.button("Analyze Sentiment") and text_input:
        with st.spinner("Analyzing sentiment..."):
            # Load sentiment analysis pipeline
            try:
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                
                # Get sentiment scores
                results = sentiment_analyzer(text_input)[0]
                
                # Display results
                sentiment_df = pd.DataFrame(results)
                
                # Create bar chart
                fig = px.bar(
                    sentiment_df, 
                    x='label', 
                    y='score', 
                    color='score',
                    color_continuous_scale='RdYlGn',
                    title="Sentiment Analysis Results"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show explanation
                max_sentiment = max(results, key=lambda x: x['score'])
                st.markdown(f"**Analysis:** The text is primarily **{max_sentiment['label'].lower()}** with a confidence score of {max_sentiment['score']:.2f}")
                
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
    
    st.markdown("#### How It Works")
    st.markdown("""
    The model processes text in the following steps:
    1. **Tokenization**: Text is split into tokens and converted to IDs
    2. **Embedding**: Tokens are converted to vector representations
    3. **Transformer Layers**: Self-attention mechanisms process the embeddings
    4. **Classification Head**: A final layer predicts sentiment probabilities
    """)
    
    # Code example
    st.subheader("Code Example")
    st.code("""
from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Analyze text
text = "I absolutely loved this product! It exceeded all my expectations."
result = sentiment_analyzer(text)

print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")
    """, language="python")

with tab2:
    st.subheader("Named Entity Recognition (NER)")
    
    st.markdown("""
    This demo uses a transformer model to identify entities like people, organizations, 
    locations, and more in text.
    """)
    
    # Example text
    example_texts = [
        "Apple Inc. is planning to open a new store in New York City, according to CEO Tim Cook.",
        "The European Union and United Nations are working together on climate change initiatives in Paris, France.",
        "Amazon and Microsoft are competing for a $10 billion contract from the Pentagon."
    ]
    
    selected_example = st.selectbox(
        "Choose an example or write your own:", 
        ["Custom input"] + example_texts,
        key="ner_example"
    )
    
    if selected_example == "Custom input":
        text_input = st.text_area("Enter text for entity recognition:", height=100, key="ner_input")
    else:
        text_input = selected_example
    
    if st.button("Identify Entities") and text_input:
        with st.spinner("Identifying entities..."):
            try:
                # Load NER pipeline
                ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                
                # Get entities
                entities = ner_pipeline(text_input)
                
                # Display entities
                if entities:
                    # Create a DataFrame for better visualization
                    entity_df = pd.DataFrame(entities)
                    
                    # Add the actual text for each entity
                    entity_df["entity_text"] = entity_df.apply(
                        lambda row: text_input[row["start"]:row["end"]], axis=1
                    )
                    
                    # Reorder and select columns
                    display_df = entity_df[["entity_group", "entity_text", "score", "start", "end"]]
                    display_df = display_df.rename(columns={
                        "entity_group": "Entity Type",
                        "entity_text": "Text",
                        "score": "Confidence",
                        "start": "Start Position",
                        "end": "End Position"
                    })
                    
                    st.dataframe(display_df)
                    
                    # Visualize entities in text
                    st.subheader("Highlighted Entities")
                    highlighted_text = text_input
                    
                    # Sort entities by start position in reverse order to avoid messing up indices
                    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
                        entity_type = entity["entity_group"]
                        start = entity["start"]
                        end = entity["end"]
                        confidence = entity["score"]
                        
                        # Define color based on entity type
                        colors = {
                            "PER": "lightblue",
                            "ORG": "lightgreen",
                            "LOC": "lightyellow",
                            "MISC": "lightpink"
                        }
                        color = colors.get(entity_type, "lightgrey")
                        
                        # Insert HTML tags for highlighting
                        highlighted_text = (
                            highlighted_text[:start] + 
                            f'<span style="background-color: {color}; padding: 0px 2px; border-radius: 3px;" title="{entity_type} ({confidence:.2f})">' + 
                            highlighted_text[start:end] + 
                            '</span>' + 
                            highlighted_text[end:]
                        )
                    
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                    
                    # Show entity type legend
                    st.markdown("**Entity Types:**")
                    legend_html = ""
                    for entity_type, color in colors.items():
                        legend_html += f'<span style="background-color: {color}; padding: 2px 8px; margin-right: 10px; border-radius: 3px;">{entity_type}</span>'
                    st.markdown(legend_html, unsafe_allow_html=True)
                else:
                    st.warning("No entities detected in the text.")
            except Exception as e:
                st.error(f"Error in entity recognition: {str(e)}")
    
    # Code example
    st.subheader("Code Example")
    st.code("""
from transformers import pipeline

# Load pre-trained NER model
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

# Analyze text
text = "Apple Inc. is planning to open a new store in New York City, according to CEO Tim Cook."
entities = ner_pipeline(text)

# Print detected entities
for entity in entities:
    print(f"{text[entity['start']:entity['end']]} - {entity['entity_group']} ({entity['score']:.2f})")
    """, language="python")

with tab3:
    st.subheader("Question Answering")
    
    st.markdown("""
    This demo uses a transformer model to answer questions based on a given context.
    The model finds the relevant span of text in the context that answers the question.
    """)
    
    # Example contexts and questions
    example_contexts = [
        {
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the entrance to the 1889 World's Fair.",
            "questions": [
                "Who designed the Eiffel Tower?",
                "When was the Eiffel Tower built?",
                "Where is the Eiffel Tower located?"
            ]
        },
        {
            "context": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
            "questions": [
                "Who created Python?",
                "When was Python first released?",
                "What does Python's design philosophy emphasize?"
            ]
        }
    ]
    
    # Choose context
    context_option = st.selectbox(
        "Choose an example context or write your own:",
        ["Custom input"] + [f"Example {i+1}: {ctx['context'][:50]}..." for i, ctx in enumerate(example_contexts)]
    )
    
    if context_option == "Custom input":
        context = st.text_area("Enter context paragraph:", height=150)
        question = st.text_input("Enter your question:")
    else:
        # Extract example number
        example_index = int(context_option.split(":")[0].replace("Example ", "")) - 1
        context = example_contexts[example_index]["context"]
        st.text_area("Context:", value=context, height=150, disabled=True)
        
        # Show predefined questions as options
        question_option = st.selectbox(
            "Choose a question or write your own:",
            ["Custom question"] + example_contexts[example_index]["questions"]
        )
        
        if question_option == "Custom question":
            question = st.text_input("Enter your question:")
        else:
            question = question_option
    
    if st.button("Answer Question") and context and question:
        with st.spinner("Finding answer..."):
            try:
                # Load question answering pipeline
                qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )
                
                # Get answer
                result = qa_pipeline(question=question, context=context)
                
                # Display answer
                st.subheader("Answer")
                
                # Calculate confidence as percentage
                confidence = result["score"] * 100
                
                # Create answer card
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: #f0f2f6;">
                    <p style="font-weight: bold; font-size: 18px;">{result["answer"]}</p>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Highlight answer in context
                start = result["start"]
                end = result["end"]
                
                highlighted_context = (
                    context[:start] + 
                    f'<span style="background-color: #FFFF00;">{context[start:end]}</span>' + 
                    context[end:]
                )
                
                st.markdown("**Context with highlighted answer:**")
                st.markdown(highlighted_context, unsafe_allow_html=True)
                
                # Show additional information
                st.markdown("**Model explanation:**")
                st.markdown("""
                The model processed the question and context, computing attention scores to find
                the most relevant span of text. It then predicted the start and end positions
                of the answer in the context.
                """)
            except Exception as e:
                st.error(f"Error in question answering: {str(e)}")
    
    # Code example
    st.subheader("Code Example")
    st.code("""
from transformers import pipeline

# Load pre-trained question answering model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Define context and question
context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889."
question = "Who designed the Eiffel Tower?"

# Get answer
result = qa_pipeline(question=question, context=context)

# Print result
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.4f}")
print(f"Start position: {result['start']}")
print(f"End position: {result['end']}")
    """, language="python")

with tab4:
    st.subheader("Text Summarization")
    
    st.markdown("""
    This demo uses a transformer model to create concise summaries of longer texts.
    The model is trained to identify and extract the most important information.
    """)
    
    # Example texts for summarization
    example_texts = [
        """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.""",
        
        """Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system and caused change on a global scale. The largest driver of warming is the emission of greenhouse gases, of which more than 90% are carbon dioxide and methane. Fossil fuel burning for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation, and manufacturing."""
    ]
    
    selected_example = st.selectbox(
        "Choose an example or write your own:",
        ["Custom input"] + [f"Example {i+1}" for i in range(len(example_texts))],
        key="summarization_example"
    )
    
    if selected_example == "Custom input":
        text_input = st.text_area("Enter text to summarize:", height=200, key="summarization_input")
    else:
        # Extract example number
        example_index = int(selected_example.split(" ")[1]) - 1
        text_input = example_texts[example_index]
        st.text_area("Text to summarize:", value=text_input, height=200, disabled=False, key="summarization_display")
    
    # Summarization parameters
    st.subheader("Summarization Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length (words)", 30, 150, 75)
    with col2:
        min_length = st.slider("Minimum summary length (words)", 10, 50, 30)
    
    if st.button("Generate Summary") and text_input:
        with st.spinner("Generating summary..."):
            try:
                # Load summarization pipeline
                summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
                
                # Calculate token counts (approximate)
                max_tokens = max_length * 1.5  # Rough conversion from words to tokens
                min_tokens = min_length * 1.5
                
                # Generate summary
                summary = summarizer(
                    text_input,
                    max_length=int(max_tokens),
                    min_length=int(min_tokens),
                    do_sample=False
                )
                
                # Display summary
                st.subheader("Summary")
                
                # Create summary card
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 20px;">
                    <p>{summary[0]['summary_text']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate and display statistics
                original_word_count = len(text_input.split())
                summary_word_count = len(summary[0]['summary_text'].split())
                reduction_percentage = ((original_word_count - summary_word_count) / original_word_count) * 100
                
                st.markdown("**Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{original_word_count} words")
                with col2:
                    st.metric("Summary Length", f"{summary_word_count} words")
                with col3:
                    st.metric("Reduction", f"{reduction_percentage:.1f}%")
                
                # Show explanation
                st.markdown("**How it works:**")
                st.markdown("""
                The summarization model uses an encoder-decoder architecture:
                1. The encoder processes the input text to understand its meaning
                2. The decoder generates a concise summary while preserving key information
                3. The model uses attention mechanisms to focus on the most important parts of the text
                """)
            except Exception as e:
                st.error(f"Error in summarization: {str(e)}")
    
    # Code example
    st.subheader("Code Example")
    st.code("""
from transformers import pipeline

# Load pre-trained summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Text to summarize
text = \"\"\"
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
\"\"\"

# Generate summary
summary = summarizer(
    text,
    max_length=100,
    min_length=30,
    do_sample=False
)

# Print summary
print(summary[0]['summary_text'])
    """, language="python")

# Further information
st.header("Comparing BERT and GPT")

st.markdown("""
BERT and GPT represent two different approaches to transformer architecture:

| Feature | BERT | GPT |
|---------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Directionality | Bidirectional | Unidirectional (left-to-right) |
| Pre-training tasks | Masked language modeling, Next sentence prediction | Next token prediction |
| Best suited for | Understanding tasks (classification, NER, QA) | Generation tasks (text completion, summarization) |
| Context handling | Processes entire context at once | Processes context sequentially |

### The Power of Attention Mechanisms

What makes transformers revolutionary is their attention mechanism:

1. **Self-attention**: Allows each token to attend to all other tokens
2. **Multi-head attention**: Enables the model to focus on different aspects of the input
3. **Parallel processing**: Unlike RNNs, transformers process all tokens simultaneously

This allows transformer models to capture long-range dependencies and relationships between words far more effectively than previous architectures.

### Practical Applications

- **Document classification**: Categorizing texts by topic or sentiment
- **Named entity recognition**: Identifying people, organizations, locations, etc.
- **Question answering**: Finding answers to questions in a given text
- **Text generation**: Creating coherent, contextually relevant text
- **Text summarization**: Condensing long documents while preserving key information
- **Translation**: Converting text between languages
""")
