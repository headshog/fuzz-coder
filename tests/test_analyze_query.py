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


def test_analyze_query_resolves_plain_function_name_from_symbols():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "split": [1],
            "decode_binary_blob": [2],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Give an example of split function"

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "split" in analysis["query_function_candidates"]
    assert "split" in analysis["function_names"]


def test_analyze_query_resolves_simple_lowercase_function_name_parse():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "parse": [11],
            "parse_json_payload": [12],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Show example of parse function from main"

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "parse" in analysis["query_function_candidates"]
    assert "parse" in analysis["function_names"]
    # Query has imperative "show" and common "from/main"; they must not become targets.
    assert "show" not in analysis["function_names"]
    assert analysis["primary_function_name"] == "parse"


def test_analyze_query_can_resolve_main_even_if_common_word():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "main": [1],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Show example of main function"

    analysis = planner.analyze_query(q)

    assert "main" in analysis["query_function_candidates"]
    assert "main" in analysis["function_names"]


def test_analyze_query_sets_memory_and_error_flags_and_negative_output():
    planner = QueryPlanner(special_indices={}, symbols={}, call_graph={}, called_by={}, meta=[])
    q = "List functions with memory management and error handling but not output/write-like"

    analysis = planner.analyze_query(q)

    assert analysis["is_listing"] is True
    assert analysis["needs_memory_mgmt"] is True
    assert analysis["needs_error_handling"] is True
    assert analysis["exclude_output"] is True
    # Negative constraint should not force output-positive retrieval branch.
    assert analysis["needs_output"] is False


def test_analyze_query_does_not_treat_imperative_words_as_function_targets():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "write": [1],
            "is": [2],
            "main": [3],
            "ma_device_init__dsound": [4],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Write an example of ma_device_init__dsound function that is called from main function"

    analysis = planner.analyze_query(q)

    assert "ma_device_init__dsound" in analysis["function_names"]
    assert "main" in analysis["function_names"]
    assert "write" not in analysis["function_names"]
    assert "is" not in analysis["function_names"]
    assert analysis["primary_function_name"] == "ma_device_init__dsound"


def test_analyze_query_example_primary_prefers_call_target_over_main_context():
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

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "main" in analysis["function_names"]
    assert "llama_sampler_init_grammar_lazy_patterns" in analysis["function_names"]
    assert analysis["primary_function_name"] == "llama_sampler_init_grammar_lazy_patterns"


def test_analyze_query_default_v2_filters_generic_symbol_like_data(monkeypatch):
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "process_request": [1],
            "main": [2],
            "data": [3],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Write an example of process_request function that is called from main function and its parameters are constructed from data given from file in argv[1]"

    monkeypatch.delenv("FC_QUERY_ANALYZER_LEGACY", raising=False)
    monkeypatch.delenv("FC_QUERY_ANALYZER_SHADOW", raising=False)
    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "process_request" in analysis["function_names"]
    assert "main" in analysis["function_names"]
    assert "data" not in analysis["function_names"]


def test_analyze_query_shadow_mode_attaches_diff_without_switching_result(monkeypatch):
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "process_request": [1],
            "main": [2],
            "data": [3],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Write an example of process_request function that is called from main function and its parameters are constructed from data given from file in argv[1]"

    monkeypatch.setenv("FC_QUERY_ANALYZER_LEGACY", "1")
    monkeypatch.setenv("FC_QUERY_ANALYZER_SHADOW", "1")
    analysis = planner.analyze_query(q)

    assert "data" in analysis["function_names"]
    assert "_shadow_diff" in analysis
    assert "function_names" in analysis["_shadow_diff"]
    assert "data" not in analysis["_shadow_diff"]["function_names"]["v2"]


def test_analyze_query_default_v2_keeps_explicit_symbol_even_if_common_word(monkeypatch):
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "write": [1],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Give an example of `write` function"

    monkeypatch.delenv("FC_QUERY_ANALYZER_LEGACY", raising=False)
    monkeypatch.delenv("FC_QUERY_ANALYZER_SHADOW", raising=False)
    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "write" in analysis["query_function_candidates"]
    assert "write" in analysis["function_names"]


def test_analyze_query_path_filter_trims_trailing_function_word():
    planner = QueryPlanner(
        special_indices={},
        symbols={"split": [1], "main": [2]},
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Write an example of split from module vendor/cpp-httplib function "
        "that is called from main function and its parameters are constructed "
        "from data given from file in argv[1]"
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert analysis["path_filters"] == ["vendor/cpp-httplib"]


def test_analyze_query_path_filter_supports_from_path_then_module_order():
    planner = QueryPlanner(
        special_indices={},
        symbols={"split": [1], "main": [2]},
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Write an example of split function that is called from main function "
        "and its parameters are constructed from data given from file in argv[1] "
        "from vendor/cpp-httplib/httplib.cpp module"
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert analysis["path_filters"] == ["vendor/cpp-httplib/httplib.cpp"]


def test_analyze_query_path_filters_do_not_include_main_from_main_function_phrase():
    planner = QueryPlanner(
        special_indices={},
        symbols={"split": [1], "main": [2]},
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Write an example of split function. In the generated snippet, define a standalone main() "
        "and call split from it. Construct parameters from file argv[1] from module vendor/cpp-httplib/httplib.cpp"
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "main" not in analysis["path_filters"]


def test_analyze_query_alias_standalone_main_does_not_treat_main_or_construct_as_target():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "process_request": [1],
            "main": [2],
            "construct": [3],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Write an example of process_request function. "
        "In the generated snippet, define a standalone main() and call process_request from it. "
        "Construct its parameters from data given from file in argv[1]"
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert "process_request" in analysis["function_names"]
    assert "main" not in analysis["function_names"]
    assert "construct" not in analysis["function_names"]


def test_analyze_query_path_filter_ignores_alias_sentence_fragment_between_from_and_module():
    planner = QueryPlanner(
        special_indices={},
        symbols={"split": [1], "main": [2]},
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Write an example of split function. In the generated snippet, define a standalone main() "
        "and call split from it. Construct its parameters from data given from file in argv[1] "
        "from module vendor/cpp-httplib/httplib.cpp"
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "example_generation"
    assert analysis["path_filters"] == ["vendor/cpp-httplib/httplib.cpp"]
