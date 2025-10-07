import streamlit as st
from typing import List


def stock_selector(default_symbols: List[str]) -> List[str]:
    """Render a symbol multiselect and return current selection."""
    return st.multiselect("Select Symbols", default_symbols, default=default_symbols)