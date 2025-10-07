# data/preprocessor.py

import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from textblob import TextBlob
from loguru import logger

class TextPreprocessor:
    """Preprocessor for cleaning and standardizing text data"""
    
    def __init__(self):
        self.stock_symbol_pattern = re.compile(r'\$[A-Z]{1,5}')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs but keep the context
        text = self.url_pattern.sub('[URL]', text)
        
        # Handle stock symbols - keep them but standardize format
        text = re.sub(r'\$([A-Z]{1,5})', r'$\1', text)
        
        # Remove mentions and hashtags (optional - might want to keep for sentiment)
        # text = self.mention_pattern.sub('', text)
        # text = self.hashtag_pattern.sub('', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\$\.\,\!\?\;\:\(\)\-\']', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = self.stock_symbol_pattern.findall(text.upper())
        return [symbol[1:] for symbol in symbols]  # Remove $ prefix
    
    def get_sentiment_score(self, text: str) -> Dict[str, float]:
        """Get comprehensive sentiment analysis"""
        try:
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                'compound_score': self._calculate_compound_score(text)
            }
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'compound_score': 0.0}
    
    def _calculate_compound_score(self, text: str) -> float:
        """Calculate a compound sentiment score using custom keywords"""
        positive_keywords = [
            'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up', 
            'positive', 'good', 'great', 'excellent', 'outperform', 'beat', 'exceed',
            'moon', 'rocket', 'surge', 'rally', 'breakout'
        ]
        
        negative_keywords = [
            'bearish', 'sell', 'weak', 'loss', 'decline', 'fall', 'down', 'drop',
            'negative', 'bad', 'terrible', 'underperform', 'miss', 'disappoint',
            'crash', 'dump', 'tank', 'plummet', 'collapse'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length and return score between -1 and 1
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        return max(-1.0, min(1.0, (positive_score - negative_score) * 10))
    
    def preprocess_news_article(self, article: Dict) -> Dict:
        """Preprocess a news article"""
        processed = article.copy()
        
        # Clean text fields
        for field in ['title', 'description', 'content']:
            if field in processed and processed[field]:
                processed[f'cleaned_{field}'] = self.clean_text(processed[field])
        
        # Extract sentiment
        full_text = ' '.join([
            processed.get('title', ''),
            processed.get('description', ''),
            processed.get('content', '')
        ])
        
        processed['sentiment'] = self.get_sentiment_score(full_text)
        processed['mentioned_symbols'] = self.extract_stock_symbols(full_text)
        processed['processed_at'] = datetime.now().isoformat()
        
        return processed
    
    def preprocess_social_post(self, post: Dict) -> Dict:
        """Preprocess a social media post"""
        processed = post.copy()
        
        # Clean text content
        text_content = ' '.join([
            processed.get('title', ''),
            processed.get('text', ''),
            processed.get('selftext', '')  # Reddit specific
        ])
        
        processed['cleaned_content'] = self.clean_text(text_content)
        processed['sentiment'] = self.get_sentiment_score(text_content)
        processed['mentioned_symbols'] = self.extract_stock_symbols(text_content)
        processed['processed_at'] = datetime.now().isoformat()
        
        return processed