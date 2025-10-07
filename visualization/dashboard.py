from typing import Dict, Any, List
import plotly.graph_objects as go


def vector_store_overview_figure(stats: Dict[str, Any]) -> go.Figure:
    """Bar chart summarizing document counts per symbol and type.

    Gracefully handles minimal stats by showing total documents if per-symbol
    breakdown is unavailable.
    """
    by_symbol = stats.get("by_symbol", {})
    if by_symbol:
        symbols = list(by_symbol.keys())
        counts = [sum(cat.values()) for cat in by_symbol.values()]
        title = "Vector Store Document Counts"
    else:
        # Fallback: show total documents metric
        symbols = ["All"]
        counts = [stats.get("total_documents", 0)]
        title = "Vector Store Document Counts (Total)"

    fig = go.Figure([go.Bar(x=symbols, y=counts, name="Documents")])
    fig.update_layout(title=title, xaxis_title="Symbol", yaxis_title="Count")
    return fig


def agent_status_badges(agent_status: Dict[str, Any]) -> List[str]:
    """Return simple textual badges for agent statuses suitable for UI display."""
    badges = []
    for name, status in agent_status.items():
        s = status.get("status", "unknown")
        badges.append(f"{name}: {s}")
    return badges