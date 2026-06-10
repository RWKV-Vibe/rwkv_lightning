import threading


class InferenceCancelled(Exception):
    pass


class PrefillBszLimitExceeded(Exception):
    def __init__(self, request_bsz: int, max_bsz: int):
        self.request_bsz = int(request_bsz)
        self.max_bsz = int(max_bsz)
        super().__init__(
            f"request bsz={self.request_bsz} exceeds max prefill bsz={self.max_bsz}"
        )


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()
