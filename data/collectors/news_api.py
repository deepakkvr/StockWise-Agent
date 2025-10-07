# data/collectors/news_api.py

import asyncio
import aiohttp
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
from config import settings

class NewsAPICollector:
    """Collects news articles from NewsAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = NewsApiClient(api_key=api_key)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_stock_news(
        self,
        symbol: str,
        company_name: str = "",
        days_back: int = 7,
        max_articles: int = 50
    ) -> List[Dict]:
        """Get news articles related to a stock"""
        try:
            # Build search query
            query_terms = [symbol]
            if company_name:
                # Add company name variations
                query_terms.extend([
                    company_name,
                    company_name.replace(" Inc", "").replace(" Corp", "").replace(" Corporation", "")
                ])
            
            query = f"({' OR '.join(query_terms)}) AND (stock OR shares OR trading OR earnings OR revenue)"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for articles
            response = self.client.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=min(max_articles, 100)  # API limit is 100
            )
            
            articles = []
            for article in response.get('articles', []):
                # Filter out articles with [Removed] content
                if (article.get('description') and 
                    '[Removed]' not in article.get('description', '') and
                    article.get('title') and
                    '[Removed]' not in article.get('title', '')):
                    
                    articles.append({
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': datetime.fromisoformat(
                            article.get('publishedAt', '').replace('Z', '+00:00')
                        ),
                        'symbol': symbol,
                        'type': 'newsapi'
                    })
            
            logger.info(f"Retrieved {len(articles)} articles for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_market_news(self, max_articles: int = 30) -> List[Dict]:
        """Get general market news"""
        try:
            response = self.client.get_everything(
                q="stock market OR trading OR Wall Street OR NYSE OR NASDAQ",
                language='en',
                sort_by='publishedAt',
                page_size=max_articles,
                from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            )
            
            articles = []
            for article in response.get('articles', []):
                if (article.get('description') and 
                    '[Removed]' not in article.get('description', '')):
                    
                    articles.append({
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': datetime.fromisoformat(
                            article.get('publishedAt', '').replace('Z', '+00:00')
                        ),
                        'symbol': 'MARKET',
                        'type': 'newsapi'
                    })
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []