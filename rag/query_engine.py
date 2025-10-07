from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.mistralai import MistralAI
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from config import settings
import os

class StockQueryEngine:
    """Advanced query engine for stock-related queries"""
    
    def __init__(self, vector_store, llm_provider: str = "mistral"):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm = self._initialize_llm()
        self.query_engine = None
        self._setup_query_engine()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        try:
            if self.llm_provider == "mistral":
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
    
    def _setup_query_engine(self):
        """Setup the query engine with custom configurations"""
        try:
            if not self.vector_store.index:
                self.vector_store._create_index()
            
            # Configure retriever
            retriever = VectorIndexRetriever(
                index=self.vector_store.index,
                similarity_top_k=20
            )
            
            # Configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode="tree_summarize"
            )
            
            # Configure postprocessor to filter by similarity
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[postprocessor]
            )
            
        except Exception as e:
            logger.error(f"Error setting up query engine: {e}")
            raise
    
    def query_stock_insights(self, symbol: str, question: str) -> Dict[str, Any]:
        """Query insights for a specific stock"""
        try:
            # Enhance the question with context
            enhanced_question = f"""
            Based on recent news, social media sentiment, and market data for {symbol}, 
            please answer the following question: {question}
            
            Provide a comprehensive analysis including:
            1. Key findings from recent news
            2. Market sentiment analysis
            3. Potential impact on stock price
            4. Risk factors to consider
            
            Question: {question}
            """

            # Build a symbol-scoped retriever for better relevance
            if not self.vector_store.index:
                self.vector_store._create_index()

            scoped_retriever = self.vector_store.index.as_retriever(
                similarity_top_k=20,
                filters=(
                    MetadataFilters(filters=[ExactMatchFilter(key="symbol", value=symbol)])
                    if symbol and symbol != "ALL" else None
                )
            )

            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode="tree_summarize"
            )

            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)

            scoped_engine = RetrieverQueryEngine(
                retriever=scoped_retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[postprocessor]
            )

            response = scoped_engine.query(enhanced_question)
            
            # Extract source information
            sources = []
            for node in response.source_nodes:
                sources.append({
                    'content_preview': node.node.text[:200] + "...",
                    'metadata': node.node.metadata,
                    'relevance_score': node.score
                })
            
            answer_text = str(response).strip() if response else ""
            if not answer_text:
                answer_text = (
                    f"No relevant insights were found for {symbol} at the moment. "
                    "Try 'Update Data Now' in the sidebar, or adjust your question."
                )

            return {
                'answer': answer_text,
                'sources': sources,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'llm_provider': self.llm_provider
            }
            
        except Exception as e:
            logger.error(f"Error querying stock insights for {symbol}: {e}")
            return {
                'answer': f"Error generating insights: {str(e)}",
                'sources': [],
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_stock_summary(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive summary for a stock"""
        try:
            summary_prompt = f"""
            Generate a comprehensive summary for stock {symbol} based on the last {days_back} days of data.
            
            Please structure your response as follows:
            1. **Current Market Position**: Brief overview of recent price movement and trading volume
            2. **News Highlights**: Key news stories and their potential impact
            3. **Sentiment Analysis**: Overall market sentiment from news and social media
            4. **Key Developments**: Important corporate announcements, earnings, partnerships, etc.
            5. **Risk Factors**: Current risks and challenges facing the company
            6. **Outlook**: Short-term outlook based on available information
            
            Focus on actionable insights and avoid speculation. Base your analysis only on the provided data.
            """
            
            response = self.query_engine.query(summary_prompt)
            
            return {
                'summary': str(response),
                'symbol': symbol,
                'period': f"{days_back} days",
                'generated_at': datetime.now().isoformat(),
                'sources_count': len(response.source_nodes)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary for {symbol}: {e}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'symbol': symbol,
                'error': str(e)
            }
    
    def compare_stocks(self, symbols: List[str], comparison_criteria: str = "overall performance") -> Dict[str, Any]:
        """Compare multiple stocks based on specified criteria"""
        try:
            symbols_str = ", ".join(symbols)
            comparison_prompt = f"""
            Compare the following stocks: {symbols_str} based on {comparison_criteria}.
            
            For each stock, analyze:
            1. Recent performance and market sentiment
            2. News coverage and key developments
            3. Social media sentiment and investor discussions
            4. Risk factors and opportunities
            
            Provide a comparative analysis highlighting:
            - Which stock shows the most positive sentiment
            - Key differentiators between the stocks
            - Relative strengths and weaknesses
            - Investment considerations for each
            
            Base your analysis on recent data and avoid speculation.
            """
            
            response = self.query_engine.query(comparison_prompt)
            
            return {
                'comparison': str(response),
                'symbols': symbols,
                'criteria': comparison_criteria,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing stocks {symbols}: {e}")
            return {
                'comparison': f"Error generating comparison: {str(e)}",
                'symbols': symbols,
                'error': str(e)
            }