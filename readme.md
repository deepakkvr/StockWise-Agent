
# StockWise Agent

StockWise Agent is an intelligent stock analysis system that leverages AI to provide comprehensive insights on stocks by analyzing financial data, news articles, and social media sentiment.

## Features

- **Intelligent Stock Analysis**: Comprehensive analysis using multiple data sources
- **News Analysis**: Processes and analyzes news articles related to stocks
- **Sentiment Analysis**: Evaluates market sentiment from various sources
- **Interactive Dashboard**: Streamlit-based UI for easy interaction
- **Real-time Data**: Fetches up-to-date stock information
- **Multi-Agent System**: Uses specialized agents for different analysis tasks

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/deepakkvr/StockWise-Agent.git
   cd StockWiseAgent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Create a `.env` file in the root directory
   - Add the following API keys:
     ```
     NEWSAPI_KEY=your_news_api_key
     OPENAI_API_KEY=your_openai_api_key
     REDDIT_CLIENT_ID=your_reddit_client_id
     REDDIT_CLIENT_SECRET=your_reddit_client_secret
     ```

## Usage

### Running the Streamlit Dashboard

```
python -m streamlit run web/streamlit_app.py
```

The dashboard provides:
- Stock selection
- Analysis mode selection (Quick Insights or Full Analysis)
- Real-time data updates
- Visualization of analysis results

### Using the Python API

```python
from main import StockWiseSystem

# Initialize the system
system = StockWiseSystem()

# Analyze a stock
result = await system.analyze_stock(
    symbol="AAPL",
    analysis_type="comprehensive",
    focus="investment",
    time_horizon="short_term"
)

# Get quick insights
insights = await system.get_quick_insights("TSLA", "What are the latest trends?")
```

## Project Structure

- `agents/`: Specialized AI agents for different analysis tasks
- `config/`: Configuration settings and API key management
- `data/`: Data collection and preprocessing modules
- `rag/`: Retrieval-Augmented Generation system for insights
- `visualization/`: Charts and reporting tools
- `web/`: Streamlit web application
- `tests/`: Test suite

## Dependencies

- Python 3.9+
- pandas, numpy, matplotlib
- streamlit
- langchain, openai, chromadb
- yfinance, newsapi-python, praw

## License

[Your License]

## Contributing

[Your contribution guidelines]
