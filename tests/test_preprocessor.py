import pytest
from data.preprocessor import TextPreprocessor


def test_clean_text_basic():
    tp = TextPreprocessor()
    text = "TSLA!!! surges 10%!!! Visit https://example.com for more."
    cleaned = tp.clean_text(text)
    assert "example.com" not in cleaned
    assert "TSLA" in cleaned


def test_sentiment_scoring():
    tp = TextPreprocessor()
    pos = tp.calculate_sentiment("Profits rise and outlook improves")
    neg = tp.calculate_sentiment("Losses widen and outlook deteriorates")
    assert pos > neg