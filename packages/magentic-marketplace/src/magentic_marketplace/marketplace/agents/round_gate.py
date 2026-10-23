"""Round gate: a round's customers wait until every seller of the round is ready."""

import asyncio


class RoundGate:
    """Holds one round's customers until all of that round's sellers have started up."""

    def __init__(self, seller_count: int, timeout: float = 300.0):
        """Gate for `seller_count` sellers; customers stop waiting after `timeout` seconds."""
        self.seller_count = seller_count
        self.timeout = timeout
        self._ready_count = 0
        self._all_ready = asyncio.Event()
        # No sellers means nothing to wait for
        if seller_count <= 0:
            self._all_ready.set()

    def seller_ready(self) -> None:
        """Mark one seller as ready, or as done trying when its start-up failed."""
        self._ready_count += 1
        if self._ready_count >= self.seller_count:
            self._all_ready.set()

    async def wait(self) -> bool:
        """Wait until every seller is ready; return False if the time limit ran out first."""
        try:
            await asyncio.wait_for(self._all_ready.wait(), timeout=self.timeout)
        except TimeoutError:
            return False
        return True
