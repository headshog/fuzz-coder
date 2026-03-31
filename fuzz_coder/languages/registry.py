from __future__ import annotations

from .base import LanguageProfile
from .c_cpp import C_CPP_PROFILE
from .java import JAVA_PROFILE


_PROFILES = {
    "c_cpp": C_CPP_PROFILE,
    "java": JAVA_PROFILE,
}


def get_language_profile(name: str) -> LanguageProfile:
    """Lookup language profile used by indexing pipeline."""
    if name not in _PROFILES:
        raise ValueError(f"Unsupported language profile: {name}")
    return _PROFILES[name]


def get_supported_language_names():
    """Return supported language profile names."""
    return sorted(_PROFILES.keys())
