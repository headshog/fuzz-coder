from __future__ import annotations

from .base import LanguageFrontend, LanguageProfile
from .c_cpp import C_CPP_FRONTEND, C_CPP_PROFILE
from .java import JAVA_FRONTEND, JAVA_PROFILE


_PROFILES = {
    "c_cpp": C_CPP_PROFILE,
    "java": JAVA_PROFILE,
}

_FRONTENDS = {
    "c_cpp": C_CPP_FRONTEND,
    "java": JAVA_FRONTEND,
}


def get_language_profile(name: str) -> LanguageProfile:
    """Lookup language profile used by indexing pipeline."""
    if name not in _PROFILES:
        raise ValueError(f"Unsupported language profile: {name}")
    return _PROFILES[name]


def get_language_frontend(name: str) -> LanguageFrontend:
    """Lookup language frontend hooks used by tree-sitter/regex extraction."""
    if name not in _FRONTENDS:
        raise ValueError(f"Unsupported language frontend: {name}")
    return _FRONTENDS[name]


def get_supported_language_names():
    """Return supported language profile names."""
    return sorted(_PROFILES.keys())
