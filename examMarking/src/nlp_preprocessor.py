"""
NLP Preprocessing Module
Handles tokenization, lemmatization, and text normalization
"""

import nltk
import spacy
from typing import List, Dict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPPreprocessor:
    """Preprocess text using NLP techniques"""
    
    def __init__(self, spacy_model="en_core_web_sm"):
        """
        Initialize NLP preprocessor
        
        Args:
            spacy_model: Name of spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Download it using:")
            logger.warning(f"python -m spacy download {spacy_model}")
            self.nlp = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Fallback: simple split
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into words
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        try:
            words = nltk.word_tokenize(text)
            return words
        except:
            # Fallback: simple split
            return text.split()
    
    def lemmatize(self, text: str, remove_stopwords=True, lowercase=True) -> List[str]:
        """
        Lemmatize text using spaCy
        
        Args:
            text: Input text
            remove_stopwords: Remove stopwords if True
            lowercase: Convert to lowercase if True
            
        Returns:
            List of lemmatized tokens
        """
        if self.nlp is None:
            logger.error("spaCy model not loaded. Cannot lemmatize.")
            return self.tokenize_words(text.lower() if lowercase else text)
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Skip stopwords if requested
            if remove_stopwords and token.text.lower() in self.stop_words:
                continue
            
            # Get lemma
            lemma = token.lemma_.lower() if lowercase else token.lemma_
            tokens.append(lemma)
        
        return tokens
    
    def extract_keywords(self, text: str, top_n=10) -> List[tuple]:
        """
        Extract important keywords using simple frequency
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of (word, frequency) tuples
        """
        tokens = self.lemmatize(text, remove_stopwords=True)
        
        # Count frequency
        freq_dist = {}
        for token in tokens:
            if len(token) > 2:  # Ignore very short words
                freq_dist[token] = freq_dist.get(token, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
    
    def preprocess(self, text: str, pipeline=['clean', 'lemmatize']) -> Dict:
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw input text
            pipeline: List of steps to apply
            
        Returns:
            Dictionary with processed results
        """
        result = {
            "original": text,
            "cleaned": text,
            "sentences": [],
            "tokens": [],
            "lemmas": [],
            "keywords": []
        }
        
        if 'clean' in pipeline:
            result["cleaned"] = self.clean_text(text)
        
        if 'sentences' in pipeline:
            result["sentences"] = self.tokenize_sentences(result["cleaned"])
        
        if 'tokenize' in pipeline:
            result["tokens"] = self.tokenize_words(result["cleaned"])
        
        if 'lemmatize' in pipeline:
            result["lemmas"] = self.lemmatize(result["cleaned"])
        
        if 'keywords' in pipeline:
            result["keywords"] = self.extract_keywords(result["cleaned"])
        
        return result


# Utility functions
def quick_preprocess(text: str) -> List[str]:
    """Quick preprocessing for simple use cases"""
    preprocessor = NLPPreprocessor()
    return preprocessor.lemmatize(text)
