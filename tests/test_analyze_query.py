from fuzz_coder.ask.core import QueryPlanner


def test_analyze_query_detects_fuzz_and_path_filter_ru():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "Напиши список функций пригодных для фаззинга из директории src/parsers"

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "listing"
    assert analysis["is_listing"] is True
    assert analysis["needs_fuzz_targets"] is True
    assert analysis["path_filters"] == ["src/parsers"]


def test_analyze_query_detects_path_filter_en_module():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "List functions from module common/ggml that parse headers"

    analysis = planner.analyze_query(q)

    assert analysis["is_listing"] is True
    assert analysis["needs_parse_like"] is True
    assert analysis["path_filters"] == ["common/ggml"]


def test_analyze_query_path_filter_trims_relative_clause():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "List functions from directory src/parsers which are good for fuzzing"

    analysis = planner.analyze_query(q)

    assert analysis["path_filters"] == ["src/parsers"]


def test_analyze_query_without_path_filter():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "Какие функции читают из stdin?"

    analysis = planner.analyze_query(q)

    assert analysis["needs_stdin"] is True
    assert analysis["path_filters"] == []


def test_analyze_query_detects_exclude_previously_listed():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "Дай список других функций для фаззинга"

    analysis = planner.analyze_query(q, context_history=[("Q1", "A1")])

    assert analysis["exclude_previously_listed"] is True
    assert analysis["follow_up"] is True


def test_analyze_query_does_not_mark_relative_that_clause_as_follow_up():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "Write a list of functions that can be used for fuzzing"

    analysis = planner.analyze_query(q, context_history=[("Q1", "A1")])

    assert analysis["is_listing"] is True
    assert analysis["needs_fuzz_targets"] is True
    assert analysis["follow_up"] is False


def test_analyze_query_marks_that_list_as_follow_up():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "Give me other functions from that list"

    analysis = planner.analyze_query(q, context_history=[("Q1", "A1")])

    assert analysis["exclude_previously_listed"] is True
    assert analysis["follow_up"] is True


def test_analyze_query_example_generation_is_not_listing_and_not_broad_fuzz():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "llama_sampler_init_grammar_lazy_patterns": [1],
            "main": [2],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Give an example calling llama_sampler_init_grammar_lazy_patterns from main function that is best for fuzzing"

    analysis = planner.analyze_query(q, context_history=[("Q1", "A1")])

    assert analysis["query_type"] == "example_generation"
    assert analysis["is_listing"] is False
    assert analysis["exclude_previously_listed"] is False
    assert analysis["needs_fuzz_targets"] is False
    assert analysis["follow_up"] is False
    assert "llama_sampler_init_grammar_lazy_patterns" in analysis["query_function_candidates"]
    assert "llama_sampler_init_grammar_lazy_patterns" in analysis["function_names"]
