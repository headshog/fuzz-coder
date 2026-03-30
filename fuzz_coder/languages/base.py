from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


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
