"""jsonptr — resolve RFC 6901 JSON Pointers against decoded JSON documents."""
from jsonptr.errors import JSONPointerError
from jsonptr.pointer import resolve

__all__ = ["resolve", "JSONPointerError"]
