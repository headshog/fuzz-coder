from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    supported_ext: Set[str]
    control_keywords: Set[str]
    stdin_patterns: List[str]
    file_input_patterns: List[str]
    api_call_patterns: List[str]
    output_patterns: List[str]
    memory_management_patterns: List[str]
    error_handling_patterns: List[str]
    type_patterns: Dict[str, List[str]]


class LanguageFrontend(ABC):
    """Language-specific parsing hooks used by index/core.py."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def get_tree_sitter_raw_language(self) -> Any:
        """Return tree-sitter raw language object/capsule or None if unavailable."""
        return None

    @abstractmethod
    def parse_tree_sitter_functions(
        self,
        *,
        parser_utils: Any,
        source_bytes: bytes,
        root: Any,
        control_keywords: Set[str],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def preprocess_regex_signature(self, compact_signature: str) -> str:
        """Language hook to normalize regex-signature candidate before parsing."""
        return compact_signature

    def extra_regex_bad_prefixes(self) -> Sequence[str]:
        """Language-specific prefixes that should be rejected in regex fallback."""
        return ()

    def merge_regex_bad_prefixes(self, default_bad_prefixes: Tuple[str, ...]) -> Tuple[str, ...]:
        extra = tuple(self.extra_regex_bad_prefixes())
        if not extra:
            return default_bad_prefixes
        return default_bad_prefixes + extra
