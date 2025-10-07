# rag/retrieval_system.py

import asyncio
from datetime import datetime, timedelta
from typing import List, Callable, Awaitable, Optional, Dict, Any
from loguru import logger
from config import settings

class RealTimeRetrievalSystem:
    """Schedules periodic data collection and vector store updates.

    This module provides a lightweight scheduler that repeatedly calls a provided
    asynchronous update function with a list of stock symbols. It tracks the last
    run timestamp and respects `settings.UPDATE_INTERVAL_HOURS` for cadence.

    Expected `update_fn` signature:
        async def update_fn(symbols: List[str]) -> Dict[str, Any]
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        update_fn: Optional[Callable[[List[str]], Awaitable[Dict[str, Any]]]] = None,
        interval_hours: Optional[int] = None,
    ):
        self.symbols = symbols or settings.DEFAULT_STOCKS
        self.update_fn = update_fn
        self.interval_hours = interval_hours or settings.UPDATE_INTERVAL_HOURS
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.last_run: Optional[datetime] = None

    async def start(self):
        """Start periodic updates in the background."""
        if not self.update_fn:
            raise ValueError("update_fn must be provided before starting the scheduler")

        if self._task and not self._task.done():
            logger.info("Retrieval system already running")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Real-time retrieval system started")

    async def stop(self):
        """Stop periodic updates."""
        if self._task:
            self._stop_event.set()
            await self._task
            logger.info("Real-time retrieval system stopped")

    async def run_once(self) -> Dict[str, Any]:
        """Run a single update cycle immediately."""
        if not self.update_fn:
            raise ValueError("update_fn must be provided to run updates")
        logger.info(f"Running single update cycle for {len(self.symbols)} symbols")
        result = await self.update_fn(self.symbols)
        self.last_run = datetime.now()
        return result

    async def _run_loop(self):
        """Internal loop respecting the configured interval."""
        # Run immediately on start
        try:
            await self.run_once()
        except Exception as e:
            logger.error(f"Initial update cycle failed: {e}")

        # Subsequent runs on schedule
        interval_seconds = max(1, int(self.interval_hours * 3600))
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

            if self._stop_event.is_set():
                break

            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Scheduled update cycle failed: {e}")

    def set_symbols(self, symbols: List[str]):
        """Update the tracked symbols list."""
        self.symbols = symbols
        logger.info(f"Updated tracked symbols: {', '.join(self.symbols)}")

    def next_run_eta(self) -> Optional[timedelta]:
        """Return time remaining until next scheduled run."""
        if not self.last_run:
            return None
        next_run = self.last_run + timedelta(hours=self.interval_hours)
        return max(timedelta(0), next_run - datetime.now())