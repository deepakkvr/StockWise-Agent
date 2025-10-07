import streamlit as st
from typing import Dict, Any, List
try:
    import pandas as pd
except Exception:
    pd = None


def _metric(label: str, value: Any, help_text: str = ""):
    try:
        st.metric(label, value)
        if help_text:
            st.caption(help_text)
    except Exception:
        st.write(f"{label}: {value}")


def _sentiment_badge(label: str):
    label_norm = (label or "").lower()
    color = {
        "positive": "#22c55e",
        "neutral": "#6b7280",
        "negative": "#ef4444",
    }.get(label_norm, "#6b7280")
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 8px;border-radius:12px;font-size:0.85rem;'>"
        f"{label or 'Unknown'}</span>",
        unsafe_allow_html=True,
    )


def _confidence_chart(conf):
    try:
        val = float(conf) if conf is not None else 0.0
    except Exception:
        val = 0.0
    pct = max(0, min(100, int(val * 100 if val <= 1 else val)))
    st.progress(pct)
    st.caption(f"Confidence: {pct}%")


def _sentiment_chart(sentiment: Dict[str, Any]):
    """Display a mini sentiment distribution chart."""
    try:
        if sentiment is None:
            st.info("No sentiment distribution available.")
            return
        pos = neu = neg = None
        if isinstance(sentiment, dict):
            # Accept various keys (positive/neutral/negative or pos/neu/neg)
            pos = sentiment.get("positive", sentiment.get("pos", 0)) or 0
            neu = sentiment.get("neutral", sentiment.get("neu", 0)) or 0
            neg = sentiment.get("negative", sentiment.get("neg", 0)) or 0
        elif isinstance(sentiment, (int, float)):
            # Convert compound score [-1, 1] to distribution
            v = float(sentiment)
            pos = max(v, 0.0)
            neg = max(-v, 0.0)
            neu = max(0.0, 1.0 - (pos + neg))
        else:
            st.info("No sentiment distribution available.")
            return

        # Normalize values if they don't sum to 1
        total = (pos or 0) + (neu or 0) + (neg or 0)
        if total <= 0:
            pos, neu, neg = 0.0, 1.0, 0.0
            total = 1.0
        pos_n = (pos or 0) / total
        neu_n = (neu or 0) / total
        neg_n = (neg or 0) / total

        data = {
            "label": ["Positive", "Neutral", "Negative"],
            "value": [pos_n, neu_n, neg_n],
        }
        if pd is not None:
            df = pd.DataFrame(data)
            st.bar_chart(df, x="label", y="value", use_container_width=True)
        else:
            st.write(data)
    except Exception as e:
        st.warning(f"Error displaying sentiment chart: {str(e)}")
        st.caption("Unable to display sentiment distribution")


def _recent_headlines(articles: List[Dict[str, Any]]):
    """Display recent headlines with count."""
    try:
        if not articles or not isinstance(articles, list):
            st.info("No recent headlines available.")
            return
        
        valid_articles = [a for a in articles if isinstance(a, dict)]
        count = len(valid_articles)
        _metric("Articles", count)
        st.markdown("**Recent Headlines**")
        
        for a in valid_articles[:5]:
            title = a.get("title") or a.get("headline") or "Untitled"
            publisher = a.get("publisher") or a.get("source") or "Unknown"
            link = a.get("link") or a.get("url") or ""
            if link:
                st.markdown(f"- [{title}]({link}) — {publisher}")
            else:
                st.markdown(f"- {title} — {publisher}")
    except Exception as e:
        st.warning(f"Could not display headlines: {str(e)}")


def display_quick_insights(result: Dict[str, Any]):
    """Render quick insights response in a human-friendly layout."""
    symbol = result.get("symbol", "")
    st.markdown(f"### {symbol} · Quick Insights")

    # Summary card
    with st.container():
        st.markdown("**Summary**")
        insights_text = result.get("insights", "")
        if insights_text:
            st.markdown(insights_text)
        else:
            st.info("No insights available yet. Try updating data or refining the question.")

    # Meta metrics and mini visuals
    cols = st.columns(3)
    with cols[0]:
        _metric("Sources", result.get("sources_count", 0))
    with cols[1]:
        _metric("Method", result.get("method", "quick_query"))
    with cols[2]:
        st.caption(f"Generated: {result.get('timestamp', '')}")


def display_full_analysis(result: Dict[str, Any]):
    """Render full analysis with badges, mini charts, headlines, and summary card."""
    try:
        symbol = result.get("symbol", "")
        st.markdown(f"### {symbol} · Full Analysis")

        # Status and high-level metrics
        cols = st.columns(3)
        with cols[0]:
            status = "Success" if result.get("success") else "Failed"
            _metric("Status", status)
        with cols[1]:
            conf = result.get("confidence_score")
            _confidence_chart(conf)
        with cols[2]:
            _metric("Processing Time", f"{result.get('processing_time', 0):.1f}s")
            st.caption(f"Generated: {result.get('timestamp', '')}")

        # Summary card with key takeaways
        syn = result.get("synthesis") or {}
        exec_sum = syn.get("executive_summary", {}) if isinstance(syn, dict) else {}
        headline = exec_sum.get("headline") or result.get("headline")
        key_points = exec_sum.get("key_points") or syn.get("highlights") or []
        with st.container():
            st.markdown("**Summary**")
            if headline:
                st.markdown(f"> {headline}")
            if isinstance(key_points, list):
                for kp in key_points[:5]:
                    st.markdown(f"- {kp}")

        # News Analysis with mini sentiment chart and headlines
        if result.get("news_analysis"):
            with st.expander("News Analysis", expanded=True):
                news = result["news_analysis"]
                overall_sent = news.get("overall_sentiment", {})
                cols2 = st.columns(2)
                with cols2[0]:
                    st.caption("Sentiment Distribution")
                    _sentiment_chart(overall_sent)
                with cols2[1]:
                    label = news.get("sentiment_label") or result.get("sentiment_label")
                    st.caption("Sentiment")
                    _sentiment_badge(label or "Unknown")

                st.markdown("**Key Insights**")
                st.write(news.get("key_insights", "N/A"))
                st.markdown("**Market Impact**")
                st.write(news.get("market_impact", "N/A"))

                # Headlines and counts
                articles = (
                    news.get("articles")
                    or news.get("recent_articles")
                    or news.get("top_articles")
                    or []
                )
                _recent_headlines(articles)

        # Sentiment Analysis with badge and mini chart
        if result.get("sentiment_analysis"):
            with st.expander("Sentiment Analysis", expanded=True):
                s = result["sentiment_analysis"]
                cols3 = st.columns(2)
                with cols3[0]:
                    st.caption("Sentiment")
                    _sentiment_badge(s.get("sentiment_label", "Unknown"))
                with cols3[1]:
                    st.caption("Confidence")
                    _confidence_chart(s.get("confidence"))
                st.caption("Distribution")
                _sentiment_chart(s.get("overall_sentiment", {}))
                
                # Add LLM insights similar to news analysis
                st.markdown("**Market Psychology**")
                st.write(s.get("market_psychology", "N/A"))
                st.markdown("**Sentiment Trend**")
                st.write(s.get("sentiment_trend", "N/A"))
                st.markdown("**Key Concerns**")
                st.write(s.get("key_concerns", "N/A"))
                st.markdown("**Trading Implications**")
                st.write(s.get("trading_implications", "N/A"))
                
                # Show metadata at the bottom
                with st.expander("Metadata", expanded=False):
                    st.write({
                        "dominant_emotion": s.get("dominant_emotion"),
                        "items_analyzed": s.get("items_analyzed"),
                    })

        # Synthesis
        if result.get("synthesis"):
            with st.expander("Synthesis", expanded=True):
                syn = result["synthesis"]
                headline = syn.get("executive_summary", {}).get("headline")
                if headline:
                    st.markdown(f"**Headline:** {headline}")
                st.markdown("**Recommendations**")
                st.write(syn.get("recommendations", {}))
                st.markdown("**Risk Assessment**")
                st.write(syn.get("risk_assessment", {}))
                st.markdown("**Investment Thesis**")
                st.write(syn.get("investment_thesis", {}))

        # Fallback raw view (collapsed)
        with st.expander("Raw Data", expanded=False):
            st.json(result)
    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.warning("Displaying raw data instead:")
        st.json(result)