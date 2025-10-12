# AI Interview Coach - NLP Demo App

This is a simple Streamlit application that demonstrates the NLP processing capabilities from the `notebooks/nlp_test.ipynb` notebook.

## Features

- **Text Preprocessing**: Removes filler words, punctuation, stopwords, and applies lemmatization
- **Feature Extraction**: Extracts keywords, named entities, and performs sentiment analysis
- **Evaluation Rubric**: Scores responses on relevance, clarity, and tone
- **Interactive UI**: Clean, user-friendly interface with visualizations
- **Real-time Analysis**: Process candidate responses instantly

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r demo_requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**: The app will automatically open at `http://localhost:8501`

## How to Use

1. **Select a Question**: Choose from sample questions or enter your own
2. **Enter Response**: Type or paste a candidate's response
3. **Analyze**: Click "Analyze Response" to see the NLP processing results
4. **Review Results**: Explore the different tabs to see:
   - Text preprocessing steps
   - Keywords and named entities
   - Sentiment analysis
   - Visualizations
   - Raw data

## Sample Questions

- "Tell me about teamwork"
- "Describe a challenging project you worked on"
- "What are your technical skills?"
- "How do you handle conflicts in a team?"
- "Tell me about a time you failed and learned from it"

## Sample Responses

The app includes several sample responses to test the NLP processing capabilities.

## Technical Details

The app uses the exact same NLP processing logic from the notebook:
- NLTK for text preprocessing
- spaCy for named entity recognition
- Transformers for sentiment analysis
- Custom evaluation rubric for scoring

## Output

The app provides:
- Overall score (0-3)
- Sentiment analysis with confidence scores
- Keyword extraction
- Named entity recognition
- Detailed preprocessing steps
- Interactive visualizations
- Downloadable results in JSON format
