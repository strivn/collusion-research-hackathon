"""Round gate: a round's customers wait until every seller of the round is ready."""


class RoundGate:
    """Holds one round's customers until all of that round's sellers have started up."""

    def __init__(self, seller_count: int, timeout: float = 300.0):
        """Gate for `seller_count` sellers; customers stop waiting after `timeout` seconds."""
        self.seller_count = seller_count
        self.timeout = timeout

    def seller_ready(self) -> None:
        """Mark one seller as ready, or as done trying when its start-up failed."""
        raise NotImplementedError("RoundGate.seller_ready")

    async def wait(self) -> bool:
        """Wait until every seller is ready; return False if the time limit ran out first."""
        raise NotImplementedError("RoundGate.wait")
