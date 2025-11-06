"""
Custom exception types for the ServiceTitan API client.

These exceptions allow callers to distinguish between failures
occurring during authentication and those arising from API requests.
"""

class ServiceTitanError(Exception):
    """Base exception for all ServiceTitan client errors."""


class ServiceTitanAuthError(ServiceTitanError):
    """Raised when authentication or token retrieval fails."""


class ServiceTitanAPIError(ServiceTitanError):
    """Raised when an HTTP request to the ServiceTitan API returns an error status."""