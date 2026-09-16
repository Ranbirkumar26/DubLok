from __future__ import annotations


class AppError(Exception):
    """Base class for user-facing pipeline failures."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(AppError):
    pass


class DependencyMissingError(AppError):
    pass


class ProcessingError(AppError):
    pass
