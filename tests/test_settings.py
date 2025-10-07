from config import settings


def test_defaults_present():
    assert settings.UPDATE_INTERVAL_HOURS >= 1
    assert len(settings.DEFAULT_STOCKS) >= 1