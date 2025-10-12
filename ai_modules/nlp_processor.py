"""
NLP Processor Module
Replicates the functionality from notebooks/nlp_test.ipynb
"""

import nltk
import spacy
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.pipeline import EntityRuler
from transformers import pipeline
from pprint import pprint

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

# Initialize models
def initialize_models():
    """Initialize spaCy and sentiment analysis models"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy English model: python -m spacy download en_core_web_sm")
        return None, None
    
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
    except Exception as e:
        print(f"Error loading sentiment analysis model: {e}")
        return nlp, None
    
    return nlp, sentiment_pipeline

class NLPProcessor:
    def __init__(self):
        download_nltk_data()
        self.nlp, self.sentiment_pipeline = initialize_models()
        
        # Filler words to remove
        self.filler_words = [
            "umm", "uh", "erm", "hmm",
            "like", "you know", "i mean", "actually",
            "basically", "literally", "seriously", 
            "okay", "ok", "so", "well",
            "right", "yeah", "yep", "y'know",
            "sort of", "kind of", "kinda",
            "just", "really", "anyway",
            "alright", "mm", "huh", "ah",
            "oh", "huh", "hmmm",
            "gotcha", "look", "see",
            "stuff", "things", "whatever"
        ]
        
        # English stopwords
        self.en_stopwords = set(stopwords.words("english"))
        
        # Setup entity ruler for programming languages
        if self.nlp:
            self.ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "LANGUAGE", "pattern": "Python"},
                {"label": "LANGUAGE", "pattern": "Java"},
                {"label": "LANGUAGE", "pattern": "C++"},
                {"label": "LANGUAGE", "pattern": "JavaScript"},
                {"label": "LANGUAGE", "pattern": "C#"},
                {"label": "LANGUAGE", "pattern": "Go"},
                {"label": "LANGUAGE", "pattern": "Rust"},
                {"label": "LANGUAGE", "pattern": "Swift"},
                {"label": "LANGUAGE", "pattern": "Kotlin"},
                {"label": "LANGUAGE", "pattern": "Scala"}
            ]
            self.ruler.add_patterns(patterns)

    def preprocess_text(self, user_response):
        """Step 2: Preprocessing and cleaning"""
        # 1. Convert to lowercase
        user_response_lower = user_response.lower()
        
        # 2. Remove filler words
        pattern = r"\b(" + "|".join(map(re.escape, self.filler_words)) + r")\b"
        user_response_no_fillers = re.sub(pattern, "", user_response_lower)
        user_response_no_fillers = re.sub(r"\s+", " ", user_response_no_fillers).strip()
        
        # 3. Remove punctuation/special chars
        user_response_no_punc = re.sub(r'[^A-Za-z0-9\s]', '', user_response_no_fillers)
        
        # 4. Tokenize
        user_response_tokenize = word_tokenize(user_response_no_punc)
        
        # 5. Lemmatize using POS
        if self.nlp:
            doc = self.nlp(" ".join(user_response_tokenize))
            user_response_lemmatized = [token.lemma_ for token in doc]
        else:
            user_response_lemmatized = user_response_tokenize
        
        # 6. Remove stopwords
        user_response_noStopwords = [
            word.lower() for word in user_response_lemmatized 
            if word.lower() not in self.en_stopwords
        ]
        
        return {
            'lowercase': user_response_lower,
            'no_fillers': user_response_no_fillers,
            'no_punctuation': user_response_no_punc,
            'tokenized': user_response_tokenize,
            'lemmatized': user_response_lemmatized,
            'no_stopwords': user_response_noStopwords
        }

    def extract_features(self, user_response, cleaned_data):
        """Step 3: Feature extraction"""
        features = {}
        
        if self.nlp:
            # 1. Extract keywords
            doc_clean = self.nlp(" ".join(cleaned_data['no_stopwords']))
            features['keywords'] = [token.text for token in doc_clean if token.pos_ in ["NOUN", "PROPN", "VERB"]]
            
            # 2. Extract named entities
            doc_original = self.nlp(user_response)
            features['named_entities'] = [(ent.text, ent.label_) for ent in doc_original.ents]
        else:
            features['keywords'] = []
            features['named_entities'] = []
        
        # 3. Detect sentiment
        if self.sentiment_pipeline:
            try:
                sentiment_result = self.sentiment_pipeline(user_response)
                features['sentiment_label'] = sentiment_result[0]['label']
                features['sentiment_score'] = sentiment_result[0]['score']
            except:
                features['sentiment_label'] = 'UNKNOWN'
                features['sentiment_score'] = 0.0
        else:
            features['sentiment_label'] = 'UNKNOWN'
            features['sentiment_score'] = 0.0
        
        return features

    def evaluate_response(self, system_question, features, cleaned_data):
        """Step 4: Define evaluation rubric"""
        # 1. Relevance check
        question_keywords = ["teamwork", "collaboration", "team", "group", "work together"]
        answer_keywords = features['keywords']
        relevance = int(any(word.lower() in question_keywords for word in answer_keywords))
        
        # 2. Clarity
        word_count = len(cleaned_data['no_stopwords'])
        has_entities = len(features['named_entities']) > 0
        clarity = int(word_count >= 5 and has_entities)
        
        # 3. Tone/Sentiment
        tone = 1 if features['sentiment_label'] in ["POSITIVE", "NEUTRAL"] else 0
        
        rubric = {
            "relevance": relevance,
            "clarity": clarity,
            "tone": tone
        }
        
        overall_score = sum(rubric.values())
        
        return rubric, overall_score

    def process_response(self, user_response, system_question="Tell me about teamwork"):
        """Main processing function that combines all steps"""
        if not self.nlp:
            return {"error": "spaCy model not loaded. Please install: python -m spacy download en_core_web_sm"}
        
        # Step 1: Input data
        original_response = user_response
        
        # Step 2: Preprocessing
        cleaned_data = self.preprocess_text(user_response)
        
        # Step 3: Feature extraction
        features = self.extract_features(user_response, cleaned_data)
        
        # Step 4: Evaluation
        rubric, overall_score = self.evaluate_response(system_question, features, cleaned_data)
        
        # Step 5: Organize outputs
        user_output = {
            "original_response": original_response,
            "cleaned_response": cleaned_data['no_stopwords'],
            "tokenized_words": cleaned_data['tokenized'],
            "lemmatized_words": cleaned_data['lemmatized'],
            "keywords": features['keywords'],
            "named_entities": features['named_entities'],
            "sentiment_label": features['sentiment_label'],
            "sentiment_score": features.get('sentiment_score', 0.0),
            "rubric": rubric,
            "overall_score": overall_score,
            "preprocessing_steps": {
                "lowercase": cleaned_data['lowercase'],
                "no_fillers": cleaned_data['no_fillers'],
                "no_punctuation": cleaned_data['no_punctuation']
            }
        }
        
        return user_output

# Convenience function for easy usage
def process_interview_response(user_response, system_question="Tell me about teamwork"):
    """Convenience function to process a single response"""
    processor = NLPProcessor()
    return processor.process_response(user_response, system_question)
