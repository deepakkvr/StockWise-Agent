from typing import Dict, Any, Optional
from pathlib import Path


def generate_markdown_report(analysis: Dict[str, Any], symbol: Optional[str] = None) -> str:
    """Create a simple markdown report from an analysis dict."""
    sym = symbol or analysis.get("symbol", "N/A")
    confidence = analysis.get("confidence", "N/A")
    summary = analysis.get("summary", "")
    recommendation = analysis.get("recommendation", "")

    lines = [
        f"# StockWise Report: {sym}",
        "",
        f"- Confidence: {confidence}",
        "",
        "## Summary",
        summary or "No summary available.",
        "",
        "## Recommendation",
        recommendation or "No recommendation available.",
    ]
    return "\n".join(lines)


def save_markdown_report(markdown: str, output_path: str) -> str:
    """Persist the markdown report to disk and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return str(path)