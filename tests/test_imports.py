def test_module_imports():
    import agents
    import rag
    import visualization
    import web
    import config.settings as settings

    assert hasattr(rag, "StockVectorStore")
    assert hasattr(visualization, "stock_price_chart")
    assert isinstance(settings.DEFAULT_STOCKS, list)