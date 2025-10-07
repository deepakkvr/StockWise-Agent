import os
import sys
import asyncio
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from main import StockWiseSystem
from config import settings
from visualization import (
    stock_price_chart,
    sentiment_trend_chart,
    combined_insights_figure,
    vector_store_overview_figure,
)
from rag import RealTimeRetrievalSystem
from components.insight_display import (
    display_quick_insights,
    display_full_analysis,
)


@st.cache_resource
def get_system() -> StockWiseSystem:
    return StockWiseSystem()


def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def run_app():
    st.set_page_config(page_title="StockWise Dashboard", layout="wide")
    st.title("StockWise Dashboard")

    system = get_system()
    default_symbols = settings.DEFAULT_STOCKS

    st.sidebar.header("Controls")
    symbols = st.sidebar.multiselect("Select Symbols", default_symbols, default=default_symbols)
    mode = st.sidebar.selectbox("Mode", ["Quick Insights", "Full Analysis"]) 
    update_data = st.sidebar.button("Update Data Now")
    if st.sidebar.button("Clear Cache and Reload"):
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()

    # Scheduler controls
    st.sidebar.subheader("Real-Time Updates")
    interval = st.sidebar.slider("Interval (hours)", 1, 24, settings.UPDATE_INTERVAL_HOURS)

    if "retrieval" not in st.session_state:
        st.session_state.retrieval = RealTimeRetrievalSystem(
            symbols=symbols,
            update_fn=system.update_stock_data,
            interval_hours=interval,
        )

    retrieval: RealTimeRetrievalSystem = st.session_state.retrieval
    retrieval.set_symbols(symbols)
    retrieval.interval_hours = interval

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Scheduler"):
            ensure_event_loop()
            asyncio.run(retrieval.start())
            st.success("Scheduler started")
        if st.button("Stop Scheduler"):
            ensure_event_loop()
            asyncio.run(retrieval.stop())
            st.info("Scheduler stopped")
    with col2:
        eta = retrieval.next_run_eta()
        st.caption(f"Next run in: {eta}" if eta else "Scheduler not yet run")

    # Data update
    if update_data:
        ensure_event_loop()
        res = asyncio.run(system.update_stock_data(symbols))
        st.success("Data updated")

    tabs = st.tabs(["Insights", "Charts", "Vector Store", "Status"]) 

    # Insights tab
    with tabs[0]:
        st.subheader("Insights")
        ensure_event_loop()
        if mode == "Quick Insights":
            query = st.text_input("Ask a question about selected stocks", "What are the latest insights?")
            selected_symbol = st.selectbox("Select symbol", symbols or default_symbols, index=0)
            if st.button("Get Quick Insights"):
                ensure_event_loop()
                result = asyncio.run(system.get_quick_insights(selected_symbol, query))
                display_quick_insights(result)
        else:
            if st.button("Run Full Analysis"):
                result = asyncio.run(system.analyze_multiple_stocks(symbols))
                for sym in symbols:
                    if sym in result:
                        display_full_analysis(result[sym])

    # Charts tab
    with tabs[1]:
        st.subheader("Charts")
        if symbols:
            for sym in symbols:
                st.markdown(f"### {sym}")
                fig = combined_insights_figure(system.vector_store, sym)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select one or more symbols in the sidebar to view charts.")

    # Vector Store tab
    with tabs[2]:
        st.subheader("Vector Store Overview")
        stats = system.vector_store.get_collection_stats()
        st.plotly_chart(vector_store_overview_figure(stats), use_container_width=True)
        st.write(stats)

    # Status tab
    with tabs[3]:
        st.subheader("System Status")
        st.json(system.get_system_status())


if __name__ == "__main__":
    run_app()