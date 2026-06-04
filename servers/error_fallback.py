"""
Fallback stubs for FailureKind, MCPToolError, and JsonFormatter.
Imported by server subprocesses when client.metrics is not on sys.path.
"""
from enum import Enum


class FailureKind(Enum):
    RETRYABLE      = "retryable"
    USER_ERROR     = "user_error"
    UPSTREAM_ERROR = "upstream_error"
    INTERNAL_ERROR = "internal_error"


class MCPToolError(Exception):
    def __init__(self, kind, message, detail=None):
        self.kind    = kind
        self.message = message
        self.detail  = detail or {}
        super().__init__(message)


JsonFormatter = None
