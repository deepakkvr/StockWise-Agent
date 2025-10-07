# agents/news_agent.py

import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from textblob import TextBlob
import re

from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini

from .base_agent import BaseAgent, AgentType, AgentResult, AgentTask
from config import settings
from loguru import logger

class NewsAnalysisAgent(BaseAgent):
    """Specialized agent for analyzing news articles and extracting financial insights"""
    
    def __init__(self, llm_provider: str = "mistral"):
        super().__init__(
            agent_type=AgentType.NEWS_ANALYZER,
            name="NewsAnalysisAgent",
            llm_provider=llm_provider,
            max_concurrent_tasks=5
        )
        
        self.llm = self._initialize_llm()
        
        # Financial keywords for relevance scoring
        self.financial_keywords = {
            'earnings': 3,
            'revenue': 3,
            'profit': 2,
            'loss': 2,
            'acquisition': 3,
            'merger': 3,
            'ipo': 3,
            'dividend': 2,
            'guidance': 3,
            'forecast': 2,
            'upgrade': 2,
            'downgrade': 2,
            'partnership': 2,
            'lawsuit': 2,
            'regulation': 2,
            'competition': 1,
            'innovation': 1,
            'product': 1,
            'market': 1
        }
        
        # Sentiment indicators
        self.positive_indicators = [
            'beat expectations', 'strong growth', 'record high', 'outperform',
            'bullish', 'positive outlook', 'exceeded', 'surge', 'rally',
            'breakthrough', 'success', 'profitable', 'expansion'
        ]
        
        self.negative_indicators = [
            'missed expectations', 'decline', 'plummet', 'bearish',
            'negative outlook', 'disappointed', 'crash', 'fall',
            'concern', 'risk', 'loss', 'debt', 'crisis'
        ]
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        try:
            if self.llm_provider == "mistral":
                return MistralAI(
                    api_key=os.environ.get("MISTRAL_API_KEY"),
                    model=settings.MISTRAL_MODEL,
                    temperature=0.1,
                    max_tokens=1000
                )
            elif self.llm_provider == "gemini":
                return Gemini(
                    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                    model=settings.GEMINI_MODEL,
                    temperature=0.1
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for news analysis"""
        required_fields = ['articles', 'symbol']
        
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        if not isinstance(input_data['articles'], list) or len(input_data['articles']) == 0:
            logger.error("Articles must be a non-empty list")
            return False
        
        return True
    
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process news analysis task"""
        start_time = datetime.now()
        
        try:
            input_data = task.input_data
            articles = input_data['articles']
            symbol = input_data['symbol']
            analysis_type = input_data.get('analysis_type', 'comprehensive')
            
            logger.info(f"Analyzing {len(articles)} articles for {symbol}")
            
            # Step 1: Basic article processing
            processed_articles = self._preprocess_articles(articles, symbol)
            
            # Step 2: Extract key themes and topics
            themes = self._extract_themes(processed_articles)
            
            # Step 3: Calculate relevance scores
            relevance_scores = self._calculate_relevance_scores(processed_articles, symbol)
            
            # Step 4: Generate LLM analysis
            llm_analysis = await self._generate_llm_analysis(processed_articles, symbol, themes)
            
            # Step 5: Create summary insights
            summary = self._create_summary(processed_articles, themes, llm_analysis, relevance_scores)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result_data = {
                'symbol': symbol,
                'analysis_type': analysis_type,
                'articles_analyzed': len(articles),
                'relevant_articles': len([a for a in processed_articles if a['relevance_score'] > 0.3]),
                'themes': themes,
                'summary': summary,
                'llm_analysis': llm_analysis,
                'top_articles': self._get_top_articles(processed_articles, limit=5),
                'sentiment_breakdown': self._get_sentiment_breakdown(processed_articles),
                'time_distribution': self._get_time_distribution(processed_articles)
            }
            
            return AgentResult(
                agent_type=self.agent_type,
                task_id=task.task_id,
                success=True,
                data=result_data,
                metadata={
                    'articles_processed': len(articles),
                    'themes_extracted': len(themes),
                    'llm_provider': self.llm_provider
                },
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence_score=self._calculate_confidence_score(processed_articles, llm_analysis),
                sources=[article.get('source', 'unknown') for article in articles]
            )
            
        except Exception as e:
            logger.error(f"Error in news analysis: {e}")
            raise
    
    def _preprocess_articles(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """Preprocess articles with enhanced analysis"""
        processed = []
        
        for article in articles:
            try:
                # Combine text content
                text_content = ' '.join([
                    article.get('title', ''),
                    article.get('description', ''),
                    article.get('content', '')
                ]).strip()
                
                if not text_content:
                    continue
                
                # Clean text
                cleaned_text = self._clean_text(text_content)
                
                # Calculate sentiment
                sentiment = self._calculate_enhanced_sentiment(cleaned_text)
                
                # Calculate relevance to symbol
                relevance_score = self._calculate_relevance_score(cleaned_text, symbol)
                
                # Extract key phrases
                key_phrases = self._extract_key_phrases(cleaned_text)
                
                processed_article = {
                    **article,
                    'cleaned_text': cleaned_text,
                    'sentiment': sentiment,
                    'relevance_score': relevance_score,
                    'key_phrases': key_phrases,
                    'word_count': len(cleaned_text.split()),
                    'symbol_mentions': len(re.findall(rf'\b{symbol}\b', text_content, re.IGNORECASE))
                }
                
                processed.append(processed_article)
                
            except Exception as e:
                logger.warning(f"Error processing article: {e}")
                continue
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
        
        return text
    
    def _calculate_enhanced_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate enhanced sentiment with financial context"""
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Financial keyword sentiment
            text_lower = text.lower()
            positive_score = sum(1 for phrase in self.positive_indicators if phrase in text_lower)
            negative_score = sum(1 for phrase in self.negative_indicators if phrase in text_lower)
            
            # Combine scores
            keyword_sentiment = (positive_score - negative_score) / max(1, positive_score + negative_score)
            
            # Weighted final sentiment
            final_sentiment = (polarity * 0.6) + (keyword_sentiment * 0.4)
            
            return {
                'polarity': float(polarity),
                'subjectivity': float(subjectivity),
                'financial_sentiment': float(keyword_sentiment),
                'combined_sentiment': float(final_sentiment),
                'positive_indicators': positive_score,
                'negative_indicators': negative_score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'financial_sentiment': 0.0,
                'combined_sentiment': 0.0,
                'positive_indicators': 0,
                'negative_indicators': 0
            }
    
    def _calculate_relevance_score(self, text: str, symbol: str) -> float:
        """Calculate how relevant the article is to the stock symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Base score for symbol mentions
        symbol_mentions = len(re.findall(rf'\b{symbol_lower}\b', text_lower))
        base_score = min(symbol_mentions * 0.2, 0.4)  # Cap at 0.4
        
        # Keyword relevance
        keyword_score = 0
        for keyword, weight in self.financial_keywords.items():
            if keyword in text_lower:
                keyword_score += weight * 0.05
        
        # Length penalty for very short articles
        words = len(text.split())
        length_factor = min(words / 50, 1.0)  # Penalty for articles < 50 words
        
        total_score = (base_score + keyword_score) * length_factor
        return min(total_score, 1.0)  # Cap at 1.0
    
    def _calculate_relevance_scores(self, articles: List[Dict], symbol: str) -> Dict[str, float]:
        """Calculate relevance scores for all articles"""
        scores = {}
        for article in articles:
            title = article.get('title', 'Untitled')
            scores[title] = article.get('relevance_score', 0.0)
        return scores
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - can be enhanced with NER or advanced NLP
        words = text.lower().split()
        
        # Find financial terms
        key_phrases = []
        for phrase in self.positive_indicators + self.negative_indicators:
            if phrase in text.lower():
                key_phrases.append(phrase)
        
        # Add financial keywords found
        for keyword in self.financial_keywords.keys():
            if keyword in text.lower():
                key_phrases.append(keyword)
        
        return list(set(key_phrases))  # Remove duplicates
    
    def _extract_themes(self, articles: List[Dict]) -> Dict[str, Any]:
        """Extract major themes from articles"""
        themes = {}
        
        # Count theme occurrences
        theme_counts = {}
        for article in articles:
            for phrase in article.get('key_phrases', []):
                theme_counts[phrase] = theme_counts.get(phrase, 0) + 1
        
        # Get top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        themes['top_themes'] = sorted_themes[:10]
        themes['earnings_related'] = sum(1 for a in articles if 'earnings' in a.get('cleaned_text', '').lower())
        themes['regulatory_related'] = sum(1 for a in articles if any(word in a.get('cleaned_text', '').lower() 
                                          for word in ['regulation', 'lawsuit', 'sec', 'fda']))
        themes['product_related'] = sum(1 for a in articles if any(word in a.get('cleaned_text', '').lower() 
                                       for word in ['product', 'launch', 'innovation', 'development']))
        
        return themes
    
    async def _generate_llm_analysis(self, articles: List[Dict], symbol: str, themes: Dict) -> Dict[str, str]:
        """Generate LLM-powered analysis"""
        try:
            # Prepare context from top articles
            top_articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)[:5]
            
            context = f"Analysis for {symbol}:\n\n"
            for i, article in enumerate(top_articles):
                context += f"Article {i+1}:\n"
                context += f"Title: {article.get('title', 'N/A')}\n"
                context += f"Content: {article.get('cleaned_text', '')[:300]}...\n"
                context += f"Sentiment: {article.get('sentiment', {}).get('combined_sentiment', 0):.2f}\n\n"
            
            # Create analysis prompt
            prompt = f"""
            As a financial analyst, analyze the following news articles about {symbol} and provide insights.
            
            {context}
            
            Key themes identified: {', '.join([theme[0] for theme in themes.get('top_themes', [])[:5]])}
            
            Please provide:
            1. Market Impact Assessment: How might these developments affect the stock price?
            2. Key Insights: What are the most important takeaways for investors?
            3. Risk Factors: What potential risks or concerns should investors be aware of?
            4. Investment Outlook: Based on this news, what's the short-term outlook?
            
            Keep your analysis concise but comprehensive, focusing on actionable insights.
            """
            
            response = await self.llm.acomplete(prompt)
            
            # Parse response into sections
            response_text = str(response)
            sections = {
                'market_impact': self._extract_section(response_text, 'Market Impact'),
                'key_insights': self._extract_section(response_text, 'Key Insights'),
                'risk_factors': self._extract_section(response_text, 'Risk Factors'),
                'investment_outlook': self._extract_section(response_text, 'Investment Outlook'),
                'full_analysis': response_text
            }
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generating LLM analysis: {e}")
            return {
                'market_impact': 'Analysis unavailable due to technical error.',
                'key_insights': 'Unable to generate insights at this time.',
                'risk_factors': 'Risk analysis unavailable.',
                'investment_outlook': 'Outlook analysis unavailable.',
                'full_analysis': f'Error: {str(e)}'
            }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        try:
            # Look for section headers
            patterns = [
                rf"{section_name}[:\-\s]+(.+?)(?=\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)",
                rf"\d+\.\s*{section_name}[:\-\s]+(.+?)(?=\n\d+\.|\n[A-Z][a-z]+ [A-Z]|$)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # If no specific section found, return generic message
            return f"{section_name} information not clearly identified in analysis."
            
        except Exception as e:
            logger.warning(f"Error extracting section {section_name}: {e}")
            return f"Unable to extract {section_name} section."
    
    def _create_summary(
        self, 
        articles: List[Dict], 
        themes: Dict, 
        llm_analysis: Dict,
        relevance_scores: Dict
    ) -> Dict[str, Any]:
        """Create comprehensive summary"""
        
        relevant_articles = [a for a in articles if a.get('relevance_score', 0) > 0.3]
        
        # Calculate overall sentiment
        if relevant_articles:
            avg_sentiment = sum(a.get('sentiment', {}).get('combined_sentiment', 0) 
                              for a in relevant_articles) / len(relevant_articles)
        else:
            avg_sentiment = 0.0
        
        # Determine sentiment label
        if avg_sentiment > 0.2:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.2:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Create summary
        summary = {
            'overall_sentiment': {
                'score': float(avg_sentiment),
                'label': sentiment_label
            },
            'article_count': {
                'total': len(articles),
                'relevant': len(relevant_articles)
            },
            'top_themes': themes.get('top_themes', [])[:5],
            'key_insights': llm_analysis.get('key_insights', 'No insights available'),
            'market_impact': llm_analysis.get('market_impact', 'Impact assessment unavailable'),
            'recommendation': self._generate_recommendation(avg_sentiment, themes, relevant_articles),
            'confidence_level': self._assess_confidence_level(relevant_articles, themes)
        }
        
        return summary
    
    def _generate_recommendation(self, sentiment: float, themes: Dict, articles: List[Dict]) -> str:
        """Generate investment recommendation based on analysis"""
        
        article_count = len(articles)
        earnings_focus = themes.get('earnings_related', 0) > 0
        
        if article_count < 3:
            return "Insufficient data - Monitor for more news"
        
        if sentiment > 0.3 and earnings_focus:
            return "Positive outlook - Consider buying on earnings strength"
        elif sentiment > 0.2:
            return "Cautiously optimistic - Watch for confirmation"
        elif sentiment < -0.3:
            return "Negative outlook - Consider risk management"
        elif sentiment < -0.2:
            return "Cautiously pessimistic - Monitor closely"
        else:
            return "Neutral outlook - Wait for clearer signals"
    
    def _assess_confidence_level(self, articles: List[Dict], themes: Dict) -> str:
        """Assess confidence level of the analysis"""
        
        if len(articles) >= 10 and themes.get('top_themes'):
            return "High"
        elif len(articles) >= 5:
            return "Medium"
        else:
            return "Low"
    
    def _get_top_articles(self, articles: List[Dict], limit: int = 5) -> List[Dict]:
        """Get top articles by relevance score"""
        sorted_articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        top_articles = []
        for article in sorted_articles[:limit]:
            top_articles.append({
                'title': article.get('title', 'N/A'),
                'source': article.get('source', 'Unknown'),
                'publishedAt': article.get('publishedAt', ''),
                'relevance_score': article.get('relevance_score', 0),
                'sentiment': article.get('sentiment', {}).get('combined_sentiment', 0),
                'url': article.get('url', ''),
                'key_phrases': article.get('key_phrases', [])[:3]
            })
        
        return top_articles
    
    def _get_sentiment_breakdown(self, articles: List[Dict]) -> Dict[str, Any]:
        """Get detailed sentiment breakdown"""
        positive = sum(1 for a in articles if a.get('sentiment', {}).get('combined_sentiment', 0) > 0.1)
        negative = sum(1 for a in articles if a.get('sentiment', {}).get('combined_sentiment', 0) < -0.1)
        neutral = len(articles) - positive - negative
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_percentage': (positive / len(articles) * 100) if articles else 0,
            'negative_percentage': (negative / len(articles) * 100) if articles else 0,
            'neutral_percentage': (neutral / len(articles) * 100) if articles else 0
        }
    
    def _get_time_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Get time distribution of articles"""
        # Use timezone-aware UTC to avoid naive/aware subtraction errors
        now = datetime.now(timezone.utc)
        
        time_buckets = {
            'last_24h': 0,
            'last_week': 0,
            'last_month': 0,
            'older': 0
        }
        
        for article in articles:
            pub_date = article.get('publishedAt')
            if not pub_date:
                time_buckets['older'] += 1
                continue
            
            if isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except:
                    time_buckets['older'] += 1
                    continue
            # Ensure timezone-aware datetime (UTC)
            if isinstance(pub_date, datetime):
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
            
            time_diff = now - pub_date
            
            if time_diff.days == 0:
                time_buckets['last_24h'] += 1
            elif time_diff.days <= 7:
                time_buckets['last_week'] += 1
            elif time_diff.days <= 30:
                time_buckets['last_month'] += 1
            else:
                time_buckets['older'] += 1
        
        return time_buckets
    
    def _calculate_confidence_score(self, articles: List[Dict], llm_analysis: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        
        # Base confidence on number of relevant articles
        relevant_articles = len([a for a in articles if a.get('relevance_score', 0) > 0.3])
        article_confidence = min(relevant_articles / 10, 1.0)  # Max at 10 articles
        
        # Confidence based on sentiment consistency
        sentiments = [a.get('sentiment', {}).get('combined_sentiment', 0) for a in articles]
        if sentiments:
            import statistics
            sentiment_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
        else:
            sentiment_std = 1.0
        sentiment_confidence = max(0, 1 - sentiment_std)  # Lower std = higher confidence
        
        # LLM analysis confidence (simple check for meaningful content)
        llm_confidence = 0.8 if len(llm_analysis.get('full_analysis', '')) > 100 else 0.3
        
        # Weighted average
        overall_confidence = (
            article_confidence * 0.4 +
            sentiment_confidence * 0.3 +
            llm_confidence * 0.3
        )
        
        return round(float(overall_confidence), 3)