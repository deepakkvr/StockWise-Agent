# agents/orchestrator.py

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import json

from .base_agent import BaseAgent, AgentType, AgentTask, AgentResult, AgentStatus
from .news_agent import NewsAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .synthesizer_agent import InsightSynthesizerAgent

from rag import StockVectorStore, StockQueryEngine
from config import settings
from loguru import logger

@dataclass
class AnalysisRequest:
    """Request for comprehensive stock analysis"""
    symbol: str
    analysis_type: str = "comprehensive"  # comprehensive, news_only, sentiment_only, synthesis_only
    time_horizon: str = "short_term"  # short_term, medium_term, long_term
    focus: str = "investment"  # investment, trading, risk_assessment
    include_social: bool = True
    max_articles: int = 50
    priority: int = 1

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    symbol: str
    request_id: str
    success: bool
    analysis_type: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    confidence_score: float
    metadata: Dict[str, Any]
    errors: List[str] = None

class StockAnalysisOrchestrator:
    """Orchestrates multiple agents to provide comprehensive stock analysis"""
    
    def __init__(self, vector_store: StockVectorStore):
        self.vector_store = vector_store
        
        # Initialize agents
        self.news_agent = NewsAnalysisAgent(llm_provider="mistral")
        self.sentiment_agent = SentimentAnalysisAgent(llm_provider="gemini")
        self.synthesizer_agent = InsightSynthesizerAgent(llm_provider="mistral")
        
        # Initialize query engine
        self.query_engine = StockQueryEngine(vector_store, llm_provider="mistral")
        
        # Active analyses tracking
        self.active_analyses: Dict[str, AnalysisRequest] = {}
        self.completed_analyses: List[AnalysisResult] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'agent_utilization': {
                'news': 0,
                'sentiment': 0,
                'synthesizer': 0
            }
        }
        
        logger.info("StockAnalysisOrchestrator initialized with all agents")
    
    async def analyze_stock(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform comprehensive stock analysis"""
        
        start_time = datetime.now()
        request_id = f"analysis_{request.symbol}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comprehensive analysis for {request.symbol} (ID: {request_id})")
        
        try:
            # Track active analysis
            self.active_analyses[request_id] = request
            
            # Step 1: Gather data from vector store
            data_sources = await self._gather_data_sources(request.symbol, request.max_articles)
            
            # Step 2: Run agents based on analysis type
            agent_results = await self._run_agents(request, data_sources)
            
            # Step 3: Synthesize results if comprehensive analysis
            if request.analysis_type == "comprehensive" and len(agent_results) > 1:
                synthesis_result = await self._synthesize_results(request, agent_results)
                agent_results['synthesis'] = synthesis_result
            
            # Step 4: Create final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate overall confidence
            confidence_scores = [result.confidence_score for result in agent_results.values() 
                               if hasattr(result, 'confidence_score') and result.confidence_score]
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            analysis_result = AnalysisResult(
                symbol=request.symbol,
                request_id=request_id,
                success=True,
                analysis_type=request.analysis_type,
                results=self._format_results(agent_results),
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence_score=overall_confidence,
                metadata={
                    'agents_used': list(agent_results.keys()),
                    'data_sources_count': len(data_sources),
                    'request_details': request.__dict__
                },
                errors=[]
            )
            
            # Update metrics and cleanup
            self._update_performance_metrics(processing_time, True)
            self.completed_analyses.append(analysis_result)
            del self.active_analyses[request_id]
            
            logger.info(f"Analysis {request_id} completed successfully in {processing_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Analysis {request_id} failed: {e}")
            
            error_result = AnalysisResult(
                symbol=request.symbol,
                request_id=request_id,
                success=False,
                analysis_type=request.analysis_type,
                results={'error': str(e)},
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence_score=0.0,
                metadata={'error_type': type(e).__name__},
                errors=[str(e)]
            )
            
            self._update_performance_metrics(processing_time, False)
            self.completed_analyses.append(error_result)
            if request_id in self.active_analyses:
                del self.active_analyses[request_id]
            
            return error_result
    
    async def _gather_data_sources(self, symbol: str, max_articles: int) -> Dict[str, Any]:
        """Gather data from vector store for analysis"""
        
        logger.info(f"Gathering data sources for {symbol}")
        
        try:
            # Get recent articles
            articles = self.vector_store.search_by_symbol(symbol, "recent news analysis", top_k=max_articles)
            
            # Get social media posts if available
            social_posts = self.vector_store.search_by_symbol(symbol, "social media sentiment", top_k=20)
            
            # Get historical data (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            historical_data = self.vector_store.search_by_timeframe(start_date, end_date, symbol, top_k=30)
            
            data_sources = {
                'news_articles': self._extract_articles_from_search(articles),
                'social_posts': self._extract_posts_from_search(social_posts),
                'historical_data': historical_data,
                'metadata': {
                    'total_articles': len(articles),
                    'total_social_posts': len(social_posts),
                    'historical_entries': len(historical_data),
                    'data_freshness': self._assess_data_freshness(articles + social_posts)
                }
            }
            
            logger.info(f"Gathered {len(articles)} articles and {len(social_posts)} social posts for {symbol}")
            
            return data_sources
            
        except Exception as e:
            logger.error(f"Error gathering data sources for {symbol}: {e}")
            return {
                'news_articles': [],
                'social_posts': [],
                'historical_data': [],
                'metadata': {'error': str(e)}
            }
    
    def _extract_articles_from_search(self, search_results: List[Dict]) -> List[Dict]:
        """Extract article data from vector search results"""
        articles = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            
            if metadata.get('type') == 'news':
                # Try to reconstruct article structure
                article = {
                    'title': metadata.get('title', ''),
                    'content': result.get('content', ''),
                    'source': metadata.get('source', ''),
                    'url': metadata.get('url', ''),
                    'publishedAt': metadata.get('timestamp', ''),
                    'symbol': metadata.get('symbol', ''),
                    'relevance_score': result.get('score', 0.0)
                }
                articles.append(article)
        
        return articles
    
    def _extract_posts_from_search(self, search_results: List[Dict]) -> List[Dict]:
        """Extract social media posts from vector search results"""
        posts = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            
            if metadata.get('type') in ['reddit', 'social']:
                post = {
                    'text': result.get('content', ''),
                    'source': metadata.get('source', ''),
                    'url': metadata.get('url', ''),
                    'created_utc': metadata.get('timestamp', ''),
                    'score': metadata.get('score', 0),
                    'sentiment_score': metadata.get('sentiment_score', 0),
                    'symbol': metadata.get('symbol', ''),
                    'relevance_score': result.get('score', 0.0)
                }
                posts.append(post)
        
        return posts
    
    def _assess_data_freshness(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Assess how fresh the available data is"""
        now = datetime.now(timezone.utc)
        
        timestamps = []
        for result in search_results:
            timestamp_str = result.get('metadata', {}).get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Ensure timezone-aware datetime
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    timestamps.append(timestamp)
                except:
                    continue
        
        if not timestamps:
            return {'freshness': 'unknown', 'latest_data': None, 'data_points': 0}
        
        latest_data = max(timestamps)
        hours_since_latest = (now - latest_data).total_seconds() / 3600
        
        if hours_since_latest < 2:
            freshness = 'very_fresh'
        elif hours_since_latest < 12:
            freshness = 'fresh'
        elif hours_since_latest < 48:
            freshness = 'moderate'
        else:
            freshness = 'stale'
        
        return {
            'freshness': freshness,
            'latest_data': latest_data.isoformat(),
            'hours_since_latest': round(hours_since_latest, 1),
            'data_points': len(timestamps)
        }
    
    async def _run_agents(self, request: AnalysisRequest, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Run appropriate agents based on analysis type"""
        
        agent_results = {}
        tasks = []
        
        # Determine which agents to run
        run_news = request.analysis_type in ['comprehensive', 'news_only']
        run_sentiment = request.analysis_type in ['comprehensive', 'sentiment_only'] 
        
        # Prepare tasks
        if run_news and data_sources['news_articles']:
            news_task_data = {
                'articles': data_sources['news_articles'],
                'symbol': request.symbol,
                'analysis_type': 'comprehensive'
            }
            tasks.append(('news', self.news_agent.submit_task(news_task_data)))
        
        if run_sentiment and (data_sources['news_articles'] or data_sources['social_posts']):
            # Combine content for sentiment analysis
            content_items = []
            
            # Add news articles
            for article in data_sources['news_articles']:
                content_items.append({
                    'title': article.get('title', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', ''),
                    'type': 'news'
                })
            
            # Add social posts if requested
            if request.include_social:
                for post in data_sources['social_posts']:
                    content_items.append({
                        'text': post.get('text', ''),
                        'source': post.get('source', ''),
                        'type': 'social'
                    })
            
            if content_items:
                sentiment_task_data = {
                    'content': content_items,
                    'symbol': request.symbol,
                    'content_type': 'mixed',
                    'include_emotions': True
                }
                tasks.append(('sentiment', self.sentiment_agent.submit_task(sentiment_task_data)))
        
        # Submit all tasks
        submitted_tasks = {}
        for agent_name, task_coro in tasks:
            task_id = await task_coro
            submitted_tasks[agent_name] = task_id
        
        # Execute tasks concurrently
        execution_tasks = []
        for agent_name, task_id in submitted_tasks.items():
            if agent_name == 'news':
                execution_tasks.append(('news', self.news_agent.execute_task(task_id)))
            elif agent_name == 'sentiment':
                execution_tasks.append(('sentiment', self.sentiment_agent.execute_task(task_id)))
        
        # Wait for all tasks to complete
        if execution_tasks:
            results = await asyncio.gather(
                *[task_coro for _, task_coro in execution_tasks],
                return_exceptions=True
            )
            
            # Collect results
            for (agent_name, _), result in zip(execution_tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"{agent_name} agent failed: {result}")
                    agent_results[agent_name] = {
                        'success': False,
                        'error': str(result),
                        'confidence_score': 0.0
                    }
                else:
                    agent_results[agent_name] = result
                    self.performance_metrics['agent_utilization'][agent_name] += 1
        
        return agent_results
    
    async def _synthesize_results(self, request: AnalysisRequest, agent_results: Dict[str, Any]) -> Any:
        """Synthesize results from multiple agents"""
        
        logger.info(f"Synthesizing results for {request.symbol}")
        
        try:
            # Prepare sources for synthesis
            sources = {}
            
            if 'news' in agent_results and agent_results['news'].success:
                sources['news'] = agent_results['news'].data
            
            if 'sentiment' in agent_results and agent_results['sentiment'].success:
                sources['sentiment'] = agent_results['sentiment'].data
            
            # Add market data if available (placeholder for future enhancement)
            # sources['market_data'] = await self._get_market_data(request.symbol)
            
            if not sources:
                logger.warning(f"No valid sources for synthesis of {request.symbol}")
                return None
            
            # Submit synthesis task with expected input schema
            synthesis_task_data = {
                'symbol': request.symbol,
                'news_analysis': sources.get('news', {}),
                'sentiment_analysis': sources.get('sentiment', {}),
                'focus': request.focus,
                'time_horizon': request.time_horizon
            }
            
            task_id = await self.synthesizer_agent.submit_task(synthesis_task_data)
            result = await self.synthesizer_agent.execute_task(task_id)
            
            self.performance_metrics['agent_utilization']['synthesizer'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _format_results(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for final output"""
        
        formatted = {
            'timestamp': datetime.now().isoformat(),
            'agents_executed': list(agent_results.keys())
        }
        
        for agent_name, result in agent_results.items():
            if hasattr(result, 'data'):  # AgentResult object
                formatted[agent_name] = {
                    'success': result.success,
                    'data': result.data,
                    'confidence': getattr(result, 'confidence_score', 0.0),
                    'processing_time': result.processing_time,
                    'metadata': result.metadata
                }
            else:  # Dictionary result
                formatted[agent_name] = result
        
        return formatted
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update orchestrator performance metrics"""
        self.performance_metrics['total_analyses'] += 1
        
        if success:
            self.performance_metrics['successful_analyses'] += 1
        
        # Update average processing time
        total = self.performance_metrics['total_analyses']
        current_avg = self.performance_metrics['average_processing_time']
        
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.performance_metrics['average_processing_time'] = new_avg
    
    async def analyze_multiple_stocks(self, symbols: List[str], **kwargs) -> List[AnalysisResult]:
        """Analyze multiple stocks concurrently"""
        
        logger.info(f"Starting batch analysis for {len(symbols)} stocks")
        
        # Create analysis requests
        requests = []
        for symbol in symbols:
            request = AnalysisRequest(symbol=symbol, **kwargs)
            requests.append(request)
        
        # Execute analyses concurrently (with some concurrency limit)
        max_concurrent = min(5, len(requests))  # Limit concurrent analyses
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(request):
            async with semaphore:
                return await self.analyze_stock(request)
        
        results = await asyncio.gather(
            *[analyze_with_semaphore(req) for req in requests],
            return_exceptions=True
        )
        
        # Handle exceptions in results
        final_results = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis for {symbol} failed: {result}")
                error_result = AnalysisResult(
                    symbol=symbol,
                    request_id=f"batch_error_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    success=False,
                    analysis_type="comprehensive",
                    results={'error': str(result)},
                    processing_time=0.0,
                    timestamp=datetime.now(),
                    confidence_score=0.0,
                    metadata={'batch_analysis': True},
                    errors=[str(result)]
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        successful = sum(1 for r in final_results if r.success)
        logger.info(f"Batch analysis completed: {successful}/{len(symbols)} successful")
        
        return final_results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'orchestrator': {
                'active_analyses': len(self.active_analyses),
                'completed_analyses': len(self.completed_analyses),
                'performance_metrics': self.performance_metrics
            },
            'news_agent': self.news_agent.get_status(),
            'sentiment_agent': self.sentiment_agent.get_status(),
            'synthesizer_agent': self.synthesizer_agent.get_status()
        }
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis results"""
        recent = self.completed_analyses[-limit:] if self.completed_analyses else []
        
        return [{
            'symbol': analysis.symbol,
            'request_id': analysis.request_id,
            'success': analysis.success,
            'analysis_type': analysis.analysis_type,
            'processing_time': analysis.processing_time,
            'timestamp': analysis.timestamp.isoformat(),
            'confidence_score': analysis.confidence_score,
            'agents_used': analysis.metadata.get('agents_used', [])
        } for analysis in recent]
    
    def clear_completed_analyses(self, keep_last: int = 100):
        """Clear old completed analyses to free memory"""
        if len(self.completed_analyses) > keep_last:
            removed = len(self.completed_analyses) - keep_last
            self.completed_analyses = self.completed_analyses[-keep_last:]
            logger.info(f"Cleared {removed} old analysis results")
        
        # Also clear agent task histories
        self.news_agent.clear_completed_tasks(50)
        self.sentiment_agent.clear_completed_tasks(50)
        self.synthesizer_agent.clear_completed_tasks(50)