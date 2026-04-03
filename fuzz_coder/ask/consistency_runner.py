from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


CRITICAL_ISSUES = {
    "target_call_arity_mismatch",
    "file_data_not_used_in_target_call",
    "observed_call_not_target",
    "observed_call_arity_mismatch",
    "target_call_not_found",
}


@dataclass
class CandidateEval:
    answer: str
    verification: Dict[str, Any]
    score: float
    is_good: bool
    label: str


def _as_set(v):
    if v is None:
        return set()
    if isinstance(v, set):
        return v
    try:
        return set(v)
    except Exception:
        return set()


def should_run_consistency_runner(verification: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    confidence_level = str(verification.get("confidence_level", "high"))
    if confidence_level in {"low", "medium"}:
        reasons.append(f"confidence_{confidence_level}")
    consistency = _as_set(verification.get("consistency_issues"))
    if consistency:
        reasons.append("consistency_issues")
    if not bool(verification.get("is_valid", False)):
        reasons.append("invalid_example")
    return bool(reasons), reasons


def _build_issue_summary(verification: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "missing_requirements",
        "consistency_issues",
        "hallucinated",
        "out_of_context",
        "file_mismatches",
        "signature_mismatches",
        "context_mismatches",
    ]:
        val = _as_set(verification.get(key))
        if val:
            parts.append(f"- {key}: {sorted(val)}")
    if not parts:
        parts.append("- none")
    return "\n".join(parts)


def build_candidate_prompts(
    *,
    original_prompt: str,
    question: str,
    first_answer: str,
    verification: Dict[str, Any],
) -> List[str]:
    issues = _build_issue_summary(verification)

    prompt_a = (
        f"{original_prompt}\n\n"
        "### Consistency Candidate Pass A\n"
        "Rewrite the answer to satisfy strict grounding checks.\n"
        "MUST:\n"
        "- exact target signature and arity\n"
        "- valid file references with line ranges\n"
        "- observed call tied to target function\n"
        "- if file-based request: argv[1]-derived data must reach target call args\n"
        "Question:\n"
        f"{question}\n\n"
        "Current issues:\n"
        f"{issues}\n\n"
        "Previous answer:\n"
        f"{first_answer}\n\n"
        "Return only corrected final answer."
    )

    prompt_b = (
        f"{original_prompt}\n\n"
        "### Consistency Candidate Pass B\n"
        "Produce an independent corrected answer emphasizing concrete evidence and data-flow.\n"
        "MUST include:\n"
        "- target File + Signature\n"
        "- caller File + Signature (or explicit not found)\n"
        "- Observed call for target (or explicit not found)\n"
        "- for argv[1]/file request: target call uses argv[1]-derived bytes/variables\n"
        "Question:\n"
        f"{question}\n\n"
        "Current issues:\n"
        f"{issues}\n\n"
        "Previous answer:\n"
        f"{first_answer}\n\n"
        "Return only corrected final answer."
    )
    return [prompt_a, prompt_b]


def score_example_verification(verification: Dict[str, Any]) -> float:
    score = 0.0
    is_valid = bool(verification.get("is_valid", False))
    confidence_score = float(verification.get("confidence_score", 0.0))

    score += 100.0 if is_valid else -40.0
    score += confidence_score * 30.0

    missing = _as_set(verification.get("missing_requirements"))
    consistency = _as_set(verification.get("consistency_issues"))
    hallucinated = _as_set(verification.get("hallucinated"))
    out_of_context = _as_set(verification.get("out_of_context"))
    file_mismatch = _as_set(verification.get("file_mismatches"))
    sig_mismatch = _as_set(verification.get("signature_mismatches"))
    ctx_mismatch = _as_set(verification.get("context_mismatches"))

    score -= 8.0 * len(missing)
    score -= 6.0 * len(consistency)
    score -= 7.0 * len(hallucinated)
    score -= 5.0 * len(out_of_context)
    score -= 4.0 * len(file_mismatch)
    score -= 4.0 * len(sig_mismatch)
    score -= 5.0 * len(ctx_mismatch)

    if bool(verification.get("target_call_present", False)):
        score += 8.0
    if bool(verification.get("target_call_arity_match", False)):
        score += 6.0
    if int(verification.get("file_matches", 0)) > 0:
        score += 5.0
    if int(verification.get("signature_matches", 0)) > 0:
        score += 5.0
    if bool(verification.get("requires_file_data", False)) and bool(verification.get("target_uses_file_data", False)):
        score += 10.0
    if bool(verification.get("target_signature_present", False)):
        score += 3.0
    if bool(verification.get("code_block_present", False)):
        score += 3.0

    critical = consistency & CRITICAL_ISSUES
    score -= 20.0 * len(critical)

    return score


def is_good_candidate(verification: Dict[str, Any]) -> bool:
    if not bool(verification.get("is_valid", False)):
        return False
    if str(verification.get("confidence_level", "low")) == "low":
        return False
    consistency = _as_set(verification.get("consistency_issues"))
    if consistency & CRITICAL_ISSUES:
        return False
    return True


def pick_best_candidate(candidates: Sequence[CandidateEval]) -> CandidateEval | None:
    if not candidates:
        return None
    # Deterministic: highest score, then lexicographic label.
    return sorted(candidates, key=lambda c: (-c.score, c.label))[0]

