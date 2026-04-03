from fuzz_coder.ask.consistency_runner import (
    CandidateEval,
    pick_best_candidate,
    score_example_verification,
    should_run_consistency_runner,
)


def test_should_run_consistency_runner_on_medium_confidence():
    verification = {
        "is_valid": True,
        "confidence_level": "medium",
        "consistency_issues": set(),
    }
    should_run, reasons = should_run_consistency_runner(verification)
    assert should_run is True
    assert "confidence_medium" in reasons


def test_pick_best_candidate_is_deterministic_on_equal_score():
    verification = {
        "is_valid": True,
        "confidence_level": "high",
        "confidence_score": 0.9,
        "missing_requirements": set(),
        "consistency_issues": set(),
        "hallucinated": set(),
        "out_of_context": set(),
        "file_mismatches": set(),
        "signature_mismatches": set(),
        "context_mismatches": set(),
        "target_call_present": True,
        "target_call_arity_match": True,
        "file_matches": 1,
        "signature_matches": 1,
        "requires_file_data": True,
        "target_uses_file_data": True,
        "target_signature_present": True,
        "code_block_present": True,
    }
    score = score_example_verification(verification)

    c2 = CandidateEval(answer="A2", verification=verification, score=score, is_good=True, label="candidate_2")
    c1 = CandidateEval(answer="A1", verification=verification, score=score, is_good=True, label="candidate_1")
    best = pick_best_candidate([c2, c1])

    assert best is not None
    assert best.label == "candidate_1"
    assert best.answer == "A1"
