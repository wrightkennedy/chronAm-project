"""Custom exceptions used across ChronAm tools."""


class OperationCancelledError(Exception):
    """Raised when a long-running operation is cancelled by the user."""

    def __init__(self, message: str = "Operation cancelled."):
        super().__init__(message)
