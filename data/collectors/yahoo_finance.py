# data/collectors/yahoo_finance.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import asyncio
import aiohttp
from dataclasses import dataclass

@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    timestamp: datetime
    news: List[Dict]

class YahooFinanceCollector:
    """Collects stock data and news from Yahoo Finance"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_stock_info(self, symbol: str) -> Optional[StockData]:
        """Get current stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="2d")
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
            
            # Get recent news
            news_data = self.get_stock_news(symbol)
            
            return StockData(
                symbol=symbol,
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                timestamp=datetime.now(),
                news=news_data
            )
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """Get recent news for a stock symbol"""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            formatted_news = []
            for article in news[:max_articles]:
                formatted_news.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'publisher': article.get('publisher', ''),
                    'providerPublishTime': datetime.fromtimestamp(
                        article.get('providerPublishTime', 0)
                    ),
                    'link': article.get('link', ''),
                    'type': 'yahoo_finance'
                })
            
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            
            if not hist.empty:
                hist.reset_index(inplace=True)
                hist['Symbol'] = symbol
                return hist
            return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, StockData]:
        """Get data for multiple stocks concurrently"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._async_get_stock_info(symbol))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
            elif result:
                stock_data[symbol] = result
        
        return stock_data
    
    async def _async_get_stock_info(self, symbol: str) -> Optional[StockData]:
        """Async wrapper for get_stock_info"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_info, symbol)