import praw
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
from textblob import TextBlob

class RedditCollector:
    """Collects relevant posts from Reddit financial subreddits"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str = "StockWise:1.0"):
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                read_only=True,
                check_for_async=False  # suppress async environment warnings per PRAW docs
            )
            self.enabled = True
        except Exception as e:
            logger.warning(f"Reddit collector not available: {e}")
            self.enabled = False
    
    def get_stock_mentions(
        self,
        symbol: str,
        subreddits: List[str] = None,
        limit: int = 100,
        time_filter: str = "week"
    ) -> List[Dict]:
        """Get Reddit posts mentioning a stock symbol"""
        if not self.enabled:
            return []
        
        if subreddits is None:
            subreddits = ["stocks", "investing", "SecurityAnalysis", "ValueInvesting", "wallstreetbets"]
        
        try:
            posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts containing the symbol
                    search_results = subreddit.search(
                        f"${symbol} OR {symbol}",
                        sort="new",
                        time_filter=time_filter,
                        limit=limit // len(subreddits)
                    )
                    
                    for post in search_results:
                        if self._is_relevant_post(post, symbol):
                            sentiment_score = self._get_sentiment_score(post.title + " " + post.selftext)
                            
                            posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'url': post.url,
                                'subreddit': subreddit_name,
                                'author': str(post.author) if post.author else '[deleted]',
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'created_utc': datetime.fromtimestamp(post.created_utc),
                                'symbol': symbol,
                                'sentiment_score': sentiment_score,
                                'type': 'reddit'
                            })
                            
                except Exception as e:
                    logger.warning(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            # Sort by score and recency
            posts.sort(key=lambda x: (x['score'], x['created_utc']), reverse=True)
            logger.info(f"Retrieved {len(posts)} Reddit posts for {symbol}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data for {symbol}: {e}")
            return []
    
    def _is_relevant_post(self, post, symbol: str) -> bool:
        """Check if post is relevant to the stock symbol"""
        text_content = (post.title + " " + post.selftext).lower()
        symbol_lower = symbol.lower()
        
        # Check for symbol mentions
        if f"${symbol_lower}" in text_content or f" {symbol_lower} " in text_content:
            return True
        
        # Avoid false positives with common words
        common_false_positives = ["a", "i", "it", "is", "on", "or", "to", "be", "go"]
        if symbol_lower in common_false_positives:
            return f"${symbol_lower}" in text_content
        
        return True
    
    def _get_sentiment_score(self, text: str) -> float:
        """Get sentiment score using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0