# agents/sentiment_agent.py

import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini

from .base_agent import BaseAgent, AgentType, AgentResult, AgentTask
from config import settings
from loguru import logger

class SentimentAnalysisAgent(BaseAgent):
    """Advanced sentiment analysis agent with financial market focus"""
    
    def __init__(self, llm_provider: str = "gemini"):
        super().__init__(
            agent_type=AgentType.SENTIMENT_ANALYZER,
            name="SentimentAnalysisAgent",
            llm_provider=llm_provider,
            max_concurrent_tasks=10
        )
        
        self.llm = self._initialize_llm()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced financial sentiment lexicon
        self.bullish_terms = {
            'moon': 4, 'rocket': 4, 'diamond hands': 3, 'hodl': 2, 'buy the dip': 3,
            'bullish': 3, 'bull run': 4, 'to the moon': 4, 'strong buy': 4,
            'outperform': 3, 'beat expectations': 3, 'record high': 3,
            'breakout': 2, 'rally': 2, 'surge': 2, 'pump': 2,
            'long': 1, 'call': 1, 'bullish af': 4, 'stonks': 2
        }
        
        self.bearish_terms = {
            'paper hands': -3, 'dump': -3, 'crash': -4, 'tank': -3,
            'bearish': -3, 'bear market': -4, 'drill': -3, 'red': -1,
            'underperform': -3, 'missed expectations': -3, 'sell off': -3,
            'breakdown': -2, 'decline': -2, 'fall': -1, 'drop': -1,
            'short': -1, 'put': -1, 'rip': -2, 'bagholders': -2
        }
        
        # Emotion indicators
        self.fear_indicators = [
            'scared', 'worried', 'panic', 'anxiety', 'concerned',
            'fearful', 'terrified', 'nervous', 'unsure', 'doubt'
        ]
        
        self.greed_indicators = [
            'fomo', 'greedy', 'all in', 'yolo', 'lambo',
            'rich', 'wealthy', 'millionaire', 'gains', 'profit'
        ]
        
        self.confidence_indicators = [
            'confident', 'sure', 'certain', 'convinced', 'believe',
            'trust', 'solid', 'strong conviction', 'dd', 'research'
        ]
    
    def _initialize_llm(self):
        """Initialize LLM for advanced sentiment analysis"""
        try:
            if self.llm_provider == "gemini":
                return Gemini(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    model=settings.GEMINI_MODEL,
                    temperature=0.1
                )
            elif self.llm_provider == "mistral":
                return MistralAI(
                    api_key=os.environ.get("MISTRAL_API_KEY"),
                    model=settings.MISTRAL_MODEL,
                    temperature=0.1
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input for sentiment analysis"""
        required_fields = ['content', 'symbol']
        
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        content = input_data['content']
        if not isinstance(content, (list, str)) or len(str(content).strip()) == 0:
            logger.error("Content must be non-empty string or list")
            return False
        
        return True
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process sentiment analysis task"""
        start_time = datetime.now()
        
        try:
            input_data = task.input_data
            content = input_data['content']
            symbol = input_data['symbol']
            content_type = input_data.get('content_type', 'mixed')
            include_emotions = input_data.get('include_emotions', True)
            
            # Handle different content types
            if isinstance(content, list):
                # Multiple pieces of content (e.g., multiple posts/articles)
                analysis_results = []
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        text = self._extract_text_from_item(item)
                        metadata = item
                    else:
                        text = str(item)
                        metadata = {'index': i}
                    
                    if text.strip():
                        item_analysis = await self._analyze_single_content(text, symbol, metadata)
                        analysis_results.append(item_analysis)
                
                # Aggregate results
                aggregated_analysis = self._aggregate_sentiment_results(analysis_results)
                
            else:
                # Single content piece
                text = str(content)
                analysis_results = [await self._analyze_single_content(text, symbol, {})]
                aggregated_analysis = analysis_results[0]
            
            # Add emotion analysis if requested
            if include_emotions:
                emotion_analysis = self._analyze_emotions(
                    [r['text'] for r in analysis_results]
                )
                aggregated_analysis['emotions'] = emotion_analysis
            
            # Generate LLM insights
            llm_insights = await self._generate_llm_sentiment_insights(
                analysis_results, symbol, content_type
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result_data = {
                'symbol': symbol,
                'content_type': content_type,
                'sentiment_summary': aggregated_analysis,
                'individual_analyses': analysis_results if len(analysis_results) <= 50 else analysis_results[:50],  # Limit for large datasets
                'llm_insights': llm_insights,
                'processing_stats': {
                    'items_analyzed': len(analysis_results),
                    'avg_confidence': sum(r.get('confidence', 0) for r in analysis_results) / len(analysis_results) if analysis_results else 0,
                    'processing_time_seconds': processing_time
                }
            }
            
            return AgentResult(
                agent_type=self.agent_type,
                task_id=task.task_id,
                success=True,
                data=result_data,
                metadata={
                    'content_items': len(analysis_results),
                    'llm_provider': self.llm_provider,
                    'content_type': content_type
                },
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence_score=aggregated_analysis.get('confidence', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise
    
    def _extract_text_from_item(self, item: Dict) -> str:
        """Extract text content from various item formats"""
        # Handle different content structures
        text_fields = ['text', 'content', 'title', 'description', 'selftext', 'cleaned_content']
        
        text_parts = []
        for field in text_fields:
            if field in item and item[field]:
                text_parts.append(str(item[field]))
        
        return ' '.join(text_parts) if text_parts else ''
    
    async def _analyze_single_content(self, text: str, symbol: str, metadata: Dict) -> Dict:
        """Analyze sentiment for a single piece of content"""
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Multiple sentiment analysis methods
        textblob_sentiment = self._textblob_analysis(cleaned_text)
        vader_sentiment = self._vader_analysis(cleaned_text)
        financial_sentiment = self._financial_lexicon_analysis(cleaned_text)
        social_sentiment = self._social_media_analysis(cleaned_text)
        
        # Combine sentiments with weights
        combined_score = (
            textblob_sentiment['polarity'] * 0.25 +
            vader_sentiment['compound'] * 0.25 +
            financial_sentiment['score'] * 0.35 +
            social_sentiment['score'] * 0.15
        )
        
        # Calculate confidence based on agreement between methods
        confidence = self._calculate_confidence([
            textblob_sentiment['polarity'],
            vader_sentiment['compound'],
            financial_sentiment['score'],
            social_sentiment['score']
        ])
        
        # Determine sentiment label
        if combined_score >= 0.2:
            label = "Bullish"
        elif combined_score <= -0.2:
            label = "Bearish"
        else:
            label = "Neutral"
        
        return {
            'text': text[:200] + '...' if len(text) > 200 else text,  # Truncate for storage
            'symbol': symbol,
            'sentiment_score': round(combined_score, 3),
            'sentiment_label': label,
            'confidence': round(confidence, 3),
            'detailed_scores': {
                'textblob': textblob_sentiment,
                'vader': vader_sentiment,
                'financial_lexicon': financial_sentiment,
                'social_media': social_sentiment
            },
            'metadata': metadata
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle stock symbols (preserve them)
        text = re.sub(r'\$([a-z]+)', r'STOCK_\1', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _textblob_analysis(self, text: str) -> Dict[str, float]:
        """TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def _vader_analysis(self, text: str) -> Dict[str, float]:
        """VADER sentiment analysis"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def _financial_lexicon_analysis(self, text: str) -> Dict[str, Any]:
        """Custom financial sentiment lexicon analysis"""
        score = 0.0
        matched_terms = []
        
        # Check bullish terms
        for term, weight in self.bullish_terms.items():
            if term in text:
                score += weight * 0.1  # Scale down the weights
                matched_terms.append(f"+{term}")
        
        # Check bearish terms
        for term, weight in self.bearish_terms.items():
            if term in text:
                score += weight * 0.1  # weight is already negative
                matched_terms.append(f"-{term}")
        
        # Normalize score
        score = max(-1.0, min(1.0, score))
        
        return {
            'score': score,
            'matched_terms': matched_terms,
            'term_count': len(matched_terms)
        }
    
    def _social_media_analysis(self, text: str) -> Dict[str, Any]:
        """Social media specific sentiment indicators"""
        
        # Count emojis and their sentiment
        emoji_sentiment = 0
        positive_emojis = ['🚀', '🌙', '💎', '👍', '🔥', '💪', '🎉', '📈']
        negative_emojis = ['💀', '😭', '😰', '👎', '🔻', '📉', '💸', '😱']
        
        for emoji in positive_emojis:
            emoji_sentiment += text.count(emoji) * 0.2
        
        for emoji in negative_emojis:
            emoji_sentiment -= text.count(emoji) * 0.2
        
        # Count caps (indicates strong emotion)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        caps_intensity = min(caps_ratio * 2, 0.3)  # Cap at 0.3
        
        # Exclamation marks
        exclamation_count = text.count('!')
        exclamation_intensity = min(exclamation_count * 0.1, 0.2)
        
        total_score = emoji_sentiment + exclamation_intensity
        if caps_ratio > 0.3:  # If lots of caps, amplify sentiment
            total_score *= 1.5
        
        return {
            'score': max(-1.0, min(1.0, total_score)),
            'emoji_sentiment': emoji_sentiment,
            'caps_ratio': caps_ratio,
            'exclamation_count': exclamation_count
        }
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on agreement between different methods"""
        if not scores or len(scores) < 2:
            return 0.5
        
        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Convert to confidence (lower std_dev = higher confidence)
        confidence = max(0.1, 1.0 - std_dev)
        
        # Boost confidence if all methods agree on direction
        positive_methods = sum(1 for score in scores if score > 0.1)
        negative_methods = sum(1 for score in scores if score < -0.1)
        
        if positive_methods == len(scores) or negative_methods == len(scores):
            confidence = min(1.0, confidence * 1.2)
        
        return confidence
    
    def _aggregate_sentiment_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment results from multiple content pieces"""
        if not results:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0}
            }
        
        # Calculate weighted average (weight by confidence)
        total_weighted_score = 0
        total_weight = 0
        
        sentiment_counts = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
        
        for result in results:
            score = result.get('sentiment_score', 0)
            confidence = result.get('confidence', 0.5)
            label = result.get('sentiment_label', 'Neutral')
            
            total_weighted_score += score * confidence
            total_weight += confidence
            sentiment_counts[label] += 1
        
        overall_sentiment = total_weighted_score / total_weight if total_weight > 0 else 0
        overall_confidence = total_weight / len(results) if results else 0
        
        # Determine overall label
        if overall_sentiment >= 0.2:
            overall_label = "Bullish"
        elif overall_sentiment <= -0.2:
            overall_label = "Bearish"
        else:
            overall_label = "Neutral"
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_label': overall_label,
            'confidence': round(overall_confidence, 3),
            'distribution': {
                'bullish': sentiment_counts['Bullish'],
                'bearish': sentiment_counts['Bearish'],
                'neutral': sentiment_counts['Neutral']
            },
            'total_analyzed': len(results),
            'score_range': {
                'min': min(r.get('sentiment_score', 0) for r in results),
                'max': max(r.get('sentiment_score', 0) for r in results),
                'std': self._calculate_std([r.get('sentiment_score', 0) for r in results])
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _analyze_emotions(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze emotional indicators in the content"""
        
        combined_text = ' '.join(texts).lower()
        
        fear_count = sum(1 for indicator in self.fear_indicators if indicator in combined_text)
        greed_count = sum(1 for indicator in self.greed_indicators if indicator in combined_text)
        confidence_count = sum(1 for indicator in self.confidence_indicators if indicator in combined_text)
        
        total_emotional_indicators = fear_count + greed_count + confidence_count
        
        if total_emotional_indicators == 0:
            return {
                'dominant_emotion': 'Neutral',
                'fear_level': 0.0,
                'greed_level': 0.0,
                'confidence_level': 0.0,
                'emotional_intensity': 0.0
            }
        
        fear_level = fear_count / len(texts)
        greed_level = greed_count / len(texts)
        confidence_level = confidence_count / len(texts)
        
        # Determine dominant emotion
        emotion_scores = {
            'Fear': fear_level,
            'Greed': greed_level,
            'Confidence': confidence_level
        }
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        emotional_intensity = max(emotion_scores.values())
        
        return {
            'dominant_emotion': dominant_emotion,
            'fear_level': round(fear_level, 3),
            'greed_level': round(greed_level, 3),
            'confidence_level': round(confidence_level, 3),
            'emotional_intensity': round(emotional_intensity, 3)
        }
    
    async def _generate_llm_sentiment_insights(
        self, 
        results: List[Dict], 
        symbol: str, 
        content_type: str
    ) -> Dict[str, str]:
        """Generate LLM-powered sentiment insights"""
        
        try:
            # Prepare summary of sentiment analysis
            if len(results) > 10:
                # Summarize for large datasets
                sample_results = results[:5] + results[-5:]  # First 5 and last 5
                summary_note = f" (Sample of {len(sample_results)} from {len(results)} total analyses)"
            else:
                sample_results = results
                summary_note = ""
            
            sentiment_summary = []
            for result in sample_results:
                sentiment_summary.append(
                    f"Text: {result['text'][:100]}... | "
                    f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.2f}) | "
                    f"Confidence: {result['confidence']:.2f}"
                )
            
            # Calculate overall sentiment statistics to provide more context
            sentiment_scores = [r['sentiment_score'] for r in results]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Enhanced prompt with clearer structure requirements
            prompt = f"""
            As a financial sentiment analyst, analyze the following sentiment analysis results for {symbol} from {content_type} content{summary_note}:

            {chr(10).join(sentiment_summary)}

            Overall average sentiment score: {avg_sentiment:.2f}

            Based on these sentiment scores and patterns, provide a structured analysis with these EXACT section headers:

            1. Market Psychology: What does the overall sentiment pattern reveal about investor psychology?
            2. Sentiment Trend: Is the sentiment trending positive, negative, or sideways?
            3. Key Concerns: What specific concerns or excitement are investors expressing?
            4. Trading Implications: How might this sentiment impact short-term trading?

            Format your response with numbered sections and clear headers exactly as shown above.
            Keep your analysis concise and focused on actionable insights for traders and investors.
            """
            
            response = await self.llm.acomplete(prompt)
            response_text = str(response)
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response for sentiment analysis: {response_text[:500]}...")
            
            # Extract sections with improved extraction
            insights = {
                'market_psychology': self._extract_llm_section(response_text, 'Market Psychology'),
                'sentiment_trend': self._extract_llm_section(response_text, 'Sentiment Trend'),
                'key_concerns': self._extract_llm_section(response_text, 'Key Concerns'),
                'trading_implications': self._extract_llm_section(response_text, 'Trading Implications'),
                'full_analysis': response_text
            }
            
            # Verify we got meaningful content
            for key, value in insights.items():
                if key != 'full_analysis' and (not value or 'not clearly identified' in value or 'Unable to extract' in value):
                    # Try a fallback approach - look for numbered sections
                    section_num = {'market_psychology': '1', 'sentiment_trend': '2', 'key_concerns': '3', 'trading_implications': '4'}
                    if key in section_num:
                        pattern = rf"{section_num[key]}\.\s*(?:[A-Za-z\s]+:)?\s*(.+?)(?=\n\d+\.|\n\n|\Z)"
                        match = re.search(pattern, response_text, re.DOTALL)
                        if match:
                            insights[key] = match.group(1).strip()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating LLM sentiment insights: {e}")
            return {
                'market_psychology': 'Unable to analyze market psychology at this time.',
                'sentiment_trend': 'Sentiment trend analysis unavailable.',
                'key_concerns': 'Key concerns analysis unavailable.',
                'trading_implications': 'Trading implications analysis unavailable.',
                'full_analysis': f'Error generating insights: {str(e)}'
            }
    
    def _extract_llm_section(self, text: str, section_name: str) -> str:
        """Extract specific section from LLM response"""
        try:
            # More robust patterns to handle various LLM response formats
            patterns = [
                # Standard numbered format (1. Market Psychology: text)
                rf"\d+\.\s*{section_name}[:\-\s]+(.+?)(?=\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)",
                # Section name as header (Market Psychology: text)
                rf"{section_name}[:\-\s]+(.+?)(?=\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)",
                # Section name in bold/markdown (**Market Psychology**: text)
                rf"\*\*{section_name}\*\*[:\-\s]+(.+?)(?=\n\*\*|\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)",
                # Just the section name followed by text (Market Psychology text)
                rf"{section_name}\s+(.+?)(?=\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)",
                # Fallback for any mention of the section name
                rf"(?:^|\n)(?:[^.]*?){section_name}(?:[^.]*?)(?::|\.)\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    # Clean up any trailing section markers
                    extracted = re.sub(r'\n\d+\..*$', '', extracted)
                    return extracted
            
            # If no match found, try to find any paragraph containing the section name
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if section_name.lower() in para.lower():
                    # Remove the section name itself
                    cleaned = re.sub(rf'{section_name}[:\-\s]*', '', para, flags=re.IGNORECASE)
                    if cleaned.strip():
                        return cleaned.strip()
            
            return f"{section_name} analysis not clearly identified."
            
        except Exception as e:
            logger.error(f"Error extracting {section_name} section: {e}")
            return f"Unable to extract {section_name} section."