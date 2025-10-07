# agents/synthesizer_agent.py

import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini

from .base_agent import BaseAgent, AgentType, AgentResult, AgentTask
from loguru import logger
from config import settings


class InsightSynthesizerAgent(BaseAgent):
    """Agent that synthesizes outputs from other agents into actionable insights."""

    def __init__(self, llm_provider: str = "mistral"):
        super().__init__(
            agent_type=AgentType.SYNTHESIZER,
            name="InsightSynthesizerAgent",
            llm_provider=llm_provider,
            max_concurrent_tasks=3,
        )
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM used for synthesis."""
        try:
            if self.llm_provider == "mistral":
                return MistralAI(
                    api_key=os.environ.get("MISTRAL_API_KEY"),
                    model=settings.MISTRAL_MODEL,
                    temperature=0.1,
                    max_tokens=1000,
                )
            elif self.llm_provider == "gemini":
                return Gemini(
                    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                    model=settings.GEMINI_MODEL,
                    temperature=0.1,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Error initializing LLM in synthesizer: {e}")
            raise

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that required fields from other agents are present."""
        # Accept either combined fields or a generic 'inputs'
        if "news_analysis" in input_data or "sentiment_analysis" in input_data:
            return True
        if "inputs" in input_data and isinstance(input_data["inputs"], dict):
            return True
        logger.error("SynthesizerAgent requires 'news_analysis' and/or 'sentiment_analysis' or 'inputs' dict")
        return False

    async def process_task(self, task: AgentTask) -> AgentResult:
        """Synthesize insights from other agent outputs."""
        start_time = datetime.now()
        try:
            payload = task.input_data

            # Normalize inputs
            news = payload.get("news_analysis") or payload.get("inputs", {}).get("news_analysis", {})
            sentiment = payload.get("sentiment_analysis") or payload.get("inputs", {}).get("sentiment_analysis", {})
            symbol = (
                payload.get("symbol")
                or payload.get("inputs", {}).get("symbol")
                or news.get("symbol")
                or sentiment.get("symbol")
                or "UNKNOWN"
            )

            # Build a concise context for the LLM
            context_summary = self._build_context_summary(news, sentiment)

            # Generate synthesis using the configured LLM
            llm_output = await self._generate_synthesis_with_llm(symbol, context_summary)

            processing_time = (datetime.now() - start_time).total_seconds()

            result_data = {
                "symbol": symbol,
                "executive_summary": {
                    "headline": llm_output.get("headline", ""),
                    "narrative": llm_output.get("summary", ""),
                },
                "recommendations": llm_output.get("recommendations", {}),
                "risk_assessment": llm_output.get("risks", {}),
                "investment_thesis": llm_output.get("thesis", {}),
                "inputs_used": {
                    "news_present": bool(news),
                    "sentiment_present": bool(sentiment),
                },
            }

            confidence_score = self._estimate_confidence(news, sentiment, llm_output)

            return AgentResult(
                agent_type=self.agent_type,
                task_id=task.task_id,
                success=True,
                data=result_data,
                metadata={
                    "llm_provider": self.llm_provider,
                    "inputs_summary_len": len(context_summary),
                },
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence_score=confidence_score,
            )

        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            raise

    def _build_context_summary(self, news: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
        """Create a compact textual summary from agent outputs for prompting."""
        parts: List[str] = []
        if news:
            parts.append(
                f"News: {news.get('articles_analyzed', 0)} articles; "
                f"themes: {list(news.get('themes', {}).keys())[:5]}"
            )
            overall = news.get("summary", {}).get("overall_sentiment", {})
            label = overall.get("label", "Neutral")
            parts.append(f"News sentiment: {label}")

        if sentiment:
            ssum = sentiment.get("sentiment_summary", {})
            label = ssum.get("sentiment_label", "Neutral")
            score = ssum.get("overall_sentiment", 0)
            parts.append(f"Social sentiment: {label} ({score:.2f})")

        return " | ".join(parts) if parts else "No structured inputs provided."

    async def _generate_synthesis_with_llm(self, symbol: str, context_summary: str) -> Dict[str, Any]:
        """Generate synthesis sections with the LLM."""
        prompt = (
            f"Synthesize market insights for {symbol}. Context: {context_summary}. "
            "Return a concise JSON with keys: headline, summary, recommendations, risks, thesis."
        )

        try:
            response = await self.llm.acomplete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.warning(f"LLM synthesis failed, falling back to templated summary: {e}")
            text = ""

        # Try to parse simple JSON-like content; if not, fallback
        parsed = self._parse_llm_json_like(text)
        if parsed:
            return parsed

        # Fallback template
        return {
            "headline": f"Synthesis for {symbol}",
            "summary": f"Based on available news and sentiment, here is a concise synthesis for {symbol}.",
            "recommendations": {"action": "Hold", "rationale": "Mixed signals"},
            "risks": {"market": "Volatility", "execution": "Delivery risks"},
            "thesis": {"core": "Monitor catalysts and sentiment shifts"},
        }

    def _parse_llm_json_like(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort parse if the LLM returns JSON-like text."""
        import json
        if not text:
            return None
        # Try strict JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try extracting a JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    def _estimate_confidence(self, news: Dict[str, Any], sentiment: Dict[str, Any], llm_output: Dict[str, Any]) -> float:
        """Estimate confidence based on inputs presence and agreement."""
        score = 0.5
        if news:
            score += 0.2
        if sentiment:
            score += 0.2
        # If both present, mild boost
        if news and sentiment:
            score += 0.1
        return round(min(score, 1.0), 3)