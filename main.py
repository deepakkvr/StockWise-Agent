# main.py (Enhanced for Phase 2)

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger

# Configure logging
logger.add("logs/stockwise_{time}.log", rotation="1 day", retention="7 days")

from config import settings, load_api_keys
from data.collectors import YahooFinanceCollector, NewsAPICollector, RedditCollector
from data.preprocessor import TextPreprocessor
from rag import StockVectorStore, StockQueryEngine
from agents import StockAnalysisOrchestrator, AnalysisRequest

class StockWiseSystem:
    """Enhanced StockWise system with intelligent agents"""
    
    def __init__(self):
        # Load API keys
        try:
            load_api_keys()
        except ValueError as e:
            logger.error(f"Missing API keys: {e}")
            raise
        
        # Initialize core components
        self.preprocessor = TextPreprocessor()
        self.vector_store = StockVectorStore()
        
        # Initialize new agent orchestrator
        self.orchestrator = StockAnalysisOrchestrator(self.vector_store)
        
        # Initialize data collectors
        self.yahoo_collector = YahooFinanceCollector()
        self.news_collector = NewsAPICollector(os.environ["NEWSAPI_KEY"])
        
        # Initialize Reddit collector if credentials available
        reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
        reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        
        if reddit_client_id and reddit_client_secret:
            self.reddit_collector = RedditCollector(reddit_client_id, reddit_client_secret)
        else:
            self.reddit_collector = None
            logger.info("Reddit collector not available - missing credentials")
        
        logger.info("StockWise System initialized with intelligent agents")
    
    async def update_stock_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Collect and store fresh data for stocks"""
        logger.info(f"Updating data for {len(symbols)} stocks")
        
        all_data = {}
        
        async with self.yahoo_collector as yahoo:
            # Collect stock data
            stock_data = await yahoo.get_multiple_stocks(symbols)
            
            for symbol in symbols:
                logger.info(f"Updating data for {symbol}")
                
                symbol_data = {
                    'symbol': symbol,
                    'stock_info': stock_data.get(symbol),
                    'news_articles': [],
                    'reddit_posts': [],
                    'collection_timestamp': datetime.now()
                }
                
                # Collect news articles
                try:
                    news_articles = self.news_collector.get_stock_news(
                        symbol=symbol,
                        max_articles=settings.MAX_NEWS_ARTICLES_PER_STOCK
                    )
                    
                    # Preprocess news articles
                    processed_articles = []
                    for article in news_articles:
                        processed_article = self.preprocessor.preprocess_news_article(article)
                        processed_articles.append(processed_article)
                    
                    symbol_data['news_articles'] = processed_articles
                    logger.info(f"Collected {len(processed_articles)} news articles for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error collecting news for {symbol}: {e}")

                # Collect Yahoo Finance news as fallback/augmentation
                try:
                    yahoo_news = self.yahoo_collector.get_stock_news(
                        symbol=symbol,
                        max_articles=settings.MAX_NEWS_ARTICLES_PER_STOCK
                    )

                    normalized_yahoo_articles = []
                    for yn in yahoo_news:
                        normalized_yahoo_articles.append({
                            'title': yn.get('title', ''),
                            'description': yn.get('summary', ''),
                            'content': '',
                            'url': yn.get('link', ''),
                            'source': yn.get('publisher', ''),
                            'publishedAt': yn.get('providerPublishTime', datetime.now()),
                            'symbol': symbol,
                            'type': yn.get('type', 'yahoo_finance')
                        })

                    processed_yahoo_articles = []
                    for article in normalized_yahoo_articles:
                        processed_article = self.preprocessor.preprocess_news_article(article)
                        processed_yahoo_articles.append(processed_article)

                    # Merge with NewsAPI articles
                    symbol_data['news_articles'].extend(processed_yahoo_articles)
                    logger.info(
                        f"Augmented with {len(processed_yahoo_articles)} Yahoo Finance articles for {symbol}"
                    )

                except Exception as e:
                    logger.error(f"Error collecting Yahoo Finance news for {symbol}: {e}")
                
                # Collect Reddit posts if available
                if self.reddit_collector:
                    try:
                        reddit_posts = self.reddit_collector.get_stock_mentions(
                            symbol=symbol,
                            limit=settings.MAX_SOCIAL_POSTS_PER_STOCK
                        )
                        
                        # Preprocess Reddit posts
                        processed_posts = []
                        for post in reddit_posts:
                            processed_post = self.preprocessor.preprocess_social_post(post)
                            processed_posts.append(processed_post)
                        
                        symbol_data['reddit_posts'] = processed_posts
                        logger.info(f"Collected {len(processed_posts)} Reddit posts for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error collecting Reddit data for {symbol}: {e}")
                
                # Store data
                self._store_symbol_data(symbol_data)
                all_data[symbol] = symbol_data
        
        # Clean up old data
        self.vector_store.clear_old_data(days_to_keep=settings.DATA_RETENTION_DAYS)
        
        logger.info("Data update completed")
        return all_data
    
    def _store_symbol_data(self, symbol_data: Dict):
        """Store data for a single symbol"""
        symbol = symbol_data['symbol']
        logger.debug(f"Storing data for {symbol} in vector store")
        
        try:
            # Store news articles
            if symbol_data.get('news_articles'):
                self.vector_store.add_news_articles(
                    symbol_data['news_articles'], 
                    symbol
                )
            
            # Store Reddit posts
            if symbol_data.get('reddit_posts'):
                self.vector_store.add_reddit_posts(
                    symbol_data['reddit_posts'], 
                    symbol
                )
            
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
    
    async def analyze_stock(
        self,
        symbol: str,
        analysis_type: str = "comprehensive",
        time_horizon: str = "short_term",
        focus: str = "investment",
        include_social: bool = True
    ) -> Dict[str, Any]:
        """Perform intelligent analysis using agents"""
        
        logger.info(f"Starting intelligent analysis for {symbol}")
        
        # Create analysis request
        request = AnalysisRequest(
            symbol=symbol,
            analysis_type=analysis_type,
            time_horizon=time_horizon,
            focus=focus,
            include_social=include_social,
            max_articles=50
        )
        
        # Execute analysis
        result = await self.orchestrator.analyze_stock(request)
        
        if result.success:
            logger.info(f"Analysis completed for {symbol} with {result.confidence_score:.1%} confidence")
        else:
            logger.error(f"Analysis failed for {symbol}: {result.errors}")
        
        return self._format_analysis_result(result)
    
    async def analyze_multiple_stocks(
        self,
        symbols: List[str],
        **analysis_options
    ) -> Dict[str, Dict]:
        """Analyze multiple stocks with intelligent agents"""
        
        logger.info(f"Starting batch analysis for {len(symbols)} stocks")
        
        # Execute batch analysis
        results = await self.orchestrator.analyze_multiple_stocks(symbols, **analysis_options)
        
        # Format results
        formatted_results = {}
        for result in results:
            formatted_results[result.symbol] = self._format_analysis_result(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch analysis completed: {successful}/{len(symbols)} successful")
        
        return formatted_results
    
    def _format_analysis_result(self, result) -> Dict[str, Any]:
        """Format analysis result for output"""
        
        formatted = {
            'symbol': result.symbol,
            'success': result.success,
            'confidence_score': result.confidence_score,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp.isoformat(),
            'analysis_type': result.analysis_type
        }
        
        if result.success:
            # Extract key insights from different agents
            results_data = result.results
            
            # News analysis insights
            if 'news' in results_data:
                news_data = results_data['news'].get('data', {})
                formatted['news_analysis'] = {
                    'articles_analyzed': news_data.get('articles_analyzed', 0),
                    'overall_sentiment': news_data.get('summary', {}).get('overall_sentiment', {}),
                    'key_insights': news_data.get('llm_analysis', {}).get('key_insights', 'N/A'),
                    'market_impact': news_data.get('llm_analysis', {}).get('market_impact', 'N/A')
                }
            
            # Sentiment analysis insights
            if 'sentiment' in results_data:
                sentiment_data = results_data['sentiment'].get('data', {})
                formatted['sentiment_analysis'] = {
                    'overall_sentiment': sentiment_data.get('sentiment_summary', {}).get('overall_sentiment', 0),
                    'sentiment_label': sentiment_data.get('sentiment_summary', {}).get('sentiment_label', 'Neutral'),
                    'confidence': sentiment_data.get('sentiment_summary', {}).get('confidence', 0),
                    'dominant_emotion': sentiment_data.get('sentiment_summary', {}).get('emotions', {}).get('dominant_emotion', 'Neutral'),
                    'items_analyzed': sentiment_data.get('processing_stats', {}).get('items_analyzed', 0),
                    'market_psychology': sentiment_data.get('llm_insights', {}).get('market_psychology', 'N/A'),
                    'sentiment_trend': sentiment_data.get('llm_insights', {}).get('sentiment_trend', 'N/A'),
                    'key_concerns': sentiment_data.get('llm_insights', {}).get('key_concerns', 'N/A'),
                    'trading_implications': sentiment_data.get('llm_insights', {}).get('trading_implications', 'N/A')
                }
            
            # Synthesis insights
            if 'synthesis' in results_data:
                synthesis_data = results_data['synthesis'].get('data', {})
                formatted['synthesis'] = {
                    'executive_summary': synthesis_data.get('executive_summary', {}),
                    'recommendations': synthesis_data.get('recommendations', {}),
                    'risk_assessment': synthesis_data.get('risk_assessment', {}),
                    'investment_thesis': synthesis_data.get('investment_thesis', {})
                }
        else:
            formatted['error'] = result.errors[0] if result.errors else 'Unknown error'
        
        return formatted
    
    async def get_quick_insights(self, symbol: str, question: str = None) -> Dict[str, Any]:
        """Get quick insights using the legacy query engine (faster)"""
        
        if question is None:
            question = f"What are the recent key developments and sentiment for {symbol}?"
        
        logger.info(f"Getting quick insights for {symbol}")
        
        # Use the existing query engine for faster results
        query_engine = StockQueryEngine(self.vector_store, llm_provider="mistral")
        insights = query_engine.query_stock_insights(symbol, question)
        
        return {
            'symbol': symbol,
            'question': question,
            'insights': insights['answer'],
            'sources_count': len(insights.get('sources', [])),
            'timestamp': datetime.now().isoformat(),
            'method': 'quick_query'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get agent status
        agent_status = self.orchestrator.get_agent_status()
        
        # Get vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        
        # Get recent analyses
        recent_analyses = self.orchestrator.get_recent_analyses(limit=5)
        
        return {
            'system_status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'vector_store': vector_stats,
            'agents': agent_status,
            'recent_analyses': recent_analyses,
            'tracked_stocks': settings.DEFAULT_STOCKS,
            'capabilities': {
                'intelligent_analysis': True,
                'multi_agent_synthesis': True,
                'social_sentiment': self.reddit_collector is not None,
                'real_time_data': True
            }
        }

# CLI Interface for Phase 2 Testing
async def test_phase2():
    """Test Phase 2 functionality"""
    print("🧠 Testing StockWise Phase 2 - Intelligent Agents")
    print("=" * 60)
    
    try:
        # Initialize system
        system = StockWiseSystem()
        
        # Test stocks
        test_symbols = ["AAPL", "TSLA"]
        
        print(f"🔄 Updating data for {', '.join(test_symbols)}...")
        await system.update_stock_data(test_symbols)
        
        print("\n📊 Running intelligent analysis...")
        
        # Test comprehensive analysis
        for symbol in test_symbols:
            print(f"\n📈 Analyzing {symbol} with intelligent agents...")
            
            analysis = await system.analyze_stock(
                symbol=symbol,
                analysis_type="comprehensive",
                focus="investment",
                time_horizon="short_term"
            )
            
            if analysis['success']:
                print(f"✅ Analysis completed with {analysis['confidence_score']:.1%} confidence")
                
                # Print key insights
                if 'news_analysis' in analysis:
                    news = analysis['news_analysis']
                    print(f"   📰 News: {news['articles_analyzed']} articles, sentiment: {news['overall_sentiment'].get('label', 'N/A')}")
                
                if 'sentiment_analysis' in analysis:
                    sentiment = analysis['sentiment_analysis']
                    print(f"   💭 Sentiment: {sentiment['sentiment_label']} ({sentiment['overall_sentiment']:.2f})")
                
                if 'synthesis' in analysis:
                    synthesis = analysis['synthesis']
                    exec_summary = synthesis.get('executive_summary', {})
                    print(f"   🎯 Recommendation: {exec_summary.get('headline', 'N/A')}")
            else:
                print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
        
        # Test quick insights
        print(f"\n⚡ Testing quick insights for AAPL...")
        quick_insights = await system.get_quick_insights("AAPL", "What's the market sentiment?")
        print(f"✅ Quick insights: {quick_insights['insights'][:100]}...")
        
        # Show system status
        status = system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Agents: {', '.join(status['agents']['orchestrator']['performance_metrics']['agent_utilization'].keys())}")
        print(f"   Vector Store: {status['vector_store']['total_documents']} documents")
        print(f"   Recent Analyses: {len(status['recent_analyses'])}")
        
        print("\n🎉 Phase 2 testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Phase 2 test failed: {e}")
        print(f"❌ Phase 2 test failed: {e}")

async def interactive_mode():
    """Interactive mode for testing different analysis types"""
    system = StockWiseSystem()
    
    print("🤖 StockWise Interactive Mode - Phase 2")
    print("Commands: analyze <symbol>, batch <symbol1,symbol2>, quick <symbol>, status, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            
            elif command.startswith("analyze "):
                symbol = command[8:].upper()
                print(f"🔍 Analyzing {symbol}...")
                result = await system.analyze_stock(symbol)
                
                if result['success']:
                    print(f"✅ Success! Confidence: {result['confidence_score']:.1%}")
                    if 'synthesis' in result:
                        headline = result['synthesis']['executive_summary'].get('headline', 'N/A')
                        print(f"📈 {headline}")
                else:
                    print(f"❌ Failed: {result.get('error')}")
            
            elif command.startswith("batch "):
                symbols = [s.strip().upper() for s in command[6:].split(",")]
                print(f"📊 Batch analyzing: {', '.join(symbols)}")
                results = await system.analyze_multiple_stocks(symbols)
                
                for symbol, result in results.items():
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} {symbol}: {result.get('confidence_score', 0):.1%}")
            
            elif command.startswith("quick "):
                symbol = command[6:].upper()
                print(f"⚡ Quick insights for {symbol}...")
                result = await system.get_quick_insights(symbol)
                print(f"💡 {result['insights'][:200]}...")
            
            elif command == "status":
                status = system.get_system_status()
                print(f"📊 Status: {status['vector_store']['total_documents']} documents")
                print(f"🤖 Agents: {len(status['recent_analyses'])} recent analyses")
            
            else:
                print("❓ Unknown command. Try: analyze AAPL, batch AAPL,MSFT, quick TSLA, status, quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(test_phase2())