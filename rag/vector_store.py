# rag/vector_store.py

import chromadb
from chromadb.config import Settings
from llama_index.core import Document, VectorStoreIndex
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from config import settings
import json
import os

class StockVectorStore:
    """Enhanced vector store for stock-related data"""
    
    def __init__(self, collection_name: str = None):
        # Configure a real LLM globally for LlamaIndex to avoid MockLLM
        try:
            llm = None
            mistral_key = os.environ.get("MISTRAL_API_KEY")
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

            if mistral_key:
                llm = MistralAI(
                    api_key=mistral_key,
                    model=settings.MISTRAL_MODEL,
                    temperature=0.0,
                )
            elif gemini_key:
                llm = Gemini(
                    api_key=gemini_key,
                    model=settings.GEMINI_MODEL,
                    temperature=0.0,
                )

            if llm:
                LlamaSettings.llm = llm
        except Exception as e:
            logger.warning(f"Could not initialize global LLM for LlamaIndex: {e}")
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embed_model = HuggingFaceEmbedding(model_name=settings.DEFAULT_EMBEDDING_MODEL)
        self.client = None
        self.collection = None
        self.vector_store = None
        self.index = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Stock insights and analysis data"}
            )
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_news_articles(self, articles: List[Dict], symbol: str):
        """Add news articles to the vector store"""
        if not articles:
            return
        
        try:
            documents = []
            
            for article in articles:
                # Create combined text content
                content_parts = []
                if article.get('title'):
                    content_parts.append(f"Title: {article['title']}")
                if article.get('description'):
                    content_parts.append(f"Description: {article['description']}")
                if article.get('content'):
                    content_parts.append(f"Content: {article['content']}")
                
                if not content_parts:
                    continue
                
                content = "\n\n".join(content_parts)
                
                # Prepare metadata
                metadata = {
                    'symbol': symbol,
                    'type': 'news',
                    'source': article.get('source', article.get('type', 'unknown')),
                    'url': article.get('url', ''),
                    'timestamp': article.get('publishedAt', datetime.now()).isoformat(),
                    'title': article.get('title', '')[:100]  # Truncate for metadata
                }
                
                # Create document
                doc = Document(
                    text=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            if documents:
                self._ingest_documents(documents)
                logger.info(f"Added {len(documents)} news articles for {symbol}")
                
        except Exception as e:
            logger.error(f"Error adding news articles: {e}")
    
    def add_reddit_posts(self, posts: List[Dict], symbol: str):
        """Add Reddit posts to the vector store"""
        if not posts:
            return
        
        try:
            documents = []
            
            for post in posts:
                # Create combined text content
                content_parts = []
                if post.get('title'):
                    content_parts.append(f"Title: {post['title']}")
                if post.get('text'):
                    content_parts.append(f"Post: {post['text']}")
                
                if not content_parts:
                    continue
                
                content = "\n\n".join(content_parts)
                
                # Prepare metadata
                metadata = {
                    'symbol': symbol,
                    'type': 'reddit',
                    'source': f"r/{post.get('subreddit', 'unknown')}",
                    'url': post.get('url', ''),
                    'timestamp': post.get('created_utc', datetime.now()).isoformat(),
                    'title': post.get('title', '')[:100],
                    'score': post.get('score', 0),
                    'sentiment_score': post.get('sentiment_score', 0)
                }
                
                # Create document
                doc = Document(
                    text=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            if documents:
                self._ingest_documents(documents)
                logger.info(f"Added {len(documents)} Reddit posts for {symbol}")
                
        except Exception as e:
            logger.error(f"Error adding Reddit posts: {e}")
    
    def add_stock_analysis(self, analysis: Dict, symbol: str):
        """Add stock analysis data to the vector store"""
        try:
            content_parts = []
            
            if analysis.get('summary'):
                content_parts.append(f"Analysis Summary: {analysis['summary']}")
            
            if analysis.get('key_points'):
                content_parts.append(f"Key Points: {' '.join(analysis['key_points'])}")
            
            if analysis.get('sentiment_analysis'):
                content_parts.append(f"Sentiment Analysis: {analysis['sentiment_analysis']}")
            
            if analysis.get('recommendation'):
                content_parts.append(f"Recommendation: {analysis['recommendation']}")
            
            if not content_parts:
                return
            
            content = "\n\n".join(content_parts)
            
            metadata = {
                'symbol': symbol,
                'type': 'analysis',
                'source': 'internal_analysis',
                'timestamp': datetime.now().isoformat(),
                'analyst': analysis.get('analyst', 'AI_Agent')
            }
            
            doc = Document(text=content, metadata=metadata)
            self._ingest_documents([doc])
            logger.info(f"Added analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"Error adding stock analysis: {e}")
    
    def _ingest_documents(self, documents: List[Document]):
        """Ingest documents into the vector store"""
        try:
            # Create ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(
                        chunk_size=settings.CHUNK_SIZE,
                        chunk_overlap=settings.CHUNK_OVERLAP
                    ),
                    self.embed_model
                ],
                vector_store=self.vector_store
            )
            
            # Run the pipeline
            pipeline.run(documents=documents)
            
            # Recreate index to include new documents
            self._create_index()
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def _create_index(self):
        """Create or recreate the vector store index"""
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def search_by_symbol(self, symbol: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents related to a specific symbol"""
        if not self.index:
            self._create_index()
        
        try:
            # Create query engine with metadata filters
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=(
                    MetadataFilters(filters=[ExactMatchFilter(key="symbol", value=symbol)])
                    if symbol != "ALL" else None
                )
            )

            nodes = retriever.retrieve(query)

            # Extract source nodes with metadata
            results = []
            for node in nodes:
                results.append({
                    'content': node.node.get_content(metadata_mode=MetadataMode.ALL),
                    'score': node.score,
                    'metadata': node.node.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by symbol {symbol}: {e}")
            return []
    
    def search_by_timeframe(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        symbol: str = None, 
        top_k: int = 10
    ) -> List[Dict]:
        """Search for documents within a specific timeframe"""
        if not self.index:
            self._create_index()
        
        try:
            # Build metadata filters
            filters = (
                MetadataFilters(filters=[ExactMatchFilter(key="symbol", value=symbol)])
                if symbol else None
            )
            
            # Note: ChromaDB doesn't support date range filters directly
            # We'll filter after retrieval
            retriever = self.index.as_retriever(
                similarity_top_k=top_k * 2,  # Get more to filter by date
                filters=filters
            )

            # Use a general query to get recent documents
            nodes = retriever.retrieve("recent stock analysis news sentiment")
            
            # Filter by date range
            filtered_results = []
            for node in nodes:
                doc_timestamp = node.node.metadata.get('timestamp')
                if doc_timestamp:
                    try:
                        doc_date = datetime.fromisoformat(doc_timestamp.replace('Z', '+00:00'))
                        # Normalize to naive datetime for consistent comparison
                        doc_date_naive = doc_date.replace(tzinfo=None)
                        if start_date <= doc_date_naive <= end_date:
                            filtered_results.append({
                                'content': node.node.get_content(metadata_mode=MetadataMode.ALL),
                                'score': node.score,
                                'metadata': node.node.metadata,
                                'timestamp': doc_date_naive
                            })
                    except ValueError:
                        continue
            
            # Sort by timestamp (most recent first)
            filtered_results.sort(key=lambda x: x['timestamp'], reverse=True)
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching by timeframe: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand content distribution
            if count > 0:
                sample_results = self.collection.get(limit=min(100, count))
                
                # Analyze metadata
                symbols = set()
                types = set()
                sources = set()
                by_symbol: Dict[str, Dict[str, int]] = {}
                
                for metadata in sample_results.get('metadatas', []):
                    if metadata:
                        symbols.add(metadata.get('symbol', 'unknown'))
                        types.add(metadata.get('type', 'unknown'))
                        sources.add(metadata.get('source', 'unknown'))
                        sym = metadata.get('symbol', 'unknown')
                        typ = metadata.get('type', 'unknown')
                        by_symbol.setdefault(sym, {})
                        by_symbol[sym][typ] = by_symbol[sym].get(typ, 0) + 1
                
                # Best-effort full breakdown when feasible
                try:
                    full = self.collection.get(limit=count)
                    full_by_symbol: Dict[str, Dict[str, int]] = {}
                    for meta in full.get('metadatas', []):
                        if meta:
                            sym = meta.get('symbol', 'unknown')
                            typ = meta.get('type', 'unknown')
                            full_by_symbol.setdefault(sym, {})
                            full_by_symbol[sym][typ] = full_by_symbol[sym].get(typ, 0) + 1
                    # Prefer full counts if available
                    if full_by_symbol:
                        by_symbol = full_by_symbol
                except Exception:
                    # If full retrieval fails, keep sample-based counts
                    pass

                return {
                    'total_documents': count,
                    'unique_symbols': len(symbols),
                    'document_types': list(types),
                    'sources': list(sources),
                    'symbols': list(symbols),
                    'by_symbol': by_symbol
                }
            
            return {'total_documents': 0}
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Get all documents
            all_docs = self.collection.get()
            
            ids_to_delete = []
            for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
                if metadata and metadata.get('timestamp'):
                    try:
                        doc_date = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                        # Normalize to naive datetime for consistent comparison
                        doc_date_naive = doc_date.replace(tzinfo=None)
                        if doc_date_naive < cutoff_date:
                            ids_to_delete.append(doc_id)
                    except ValueError:
                        continue
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old documents")
                
                # Recreate index after deletion
                self._create_index()
            
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")