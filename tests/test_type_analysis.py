from fuzz_coder.ask.core import QueryPlanner, build_type_analysis_from_context


def test_analyze_query_explain_alias_text_switches_to_type_analysis_for_unresolved_symbol():
    planner = QueryPlanner(
        special_indices={},
        symbols={
            "llama_params_fit": [1],
        },
        call_graph={},
        called_by={},
        meta=[],
    )
    q = (
        "Analyze function parameter semantics for ColorMapObject: for each parameter, explain its role, "
        "expected data format/range, whether it is input/output/inout, where values usually come from in the codebase, "
        "and provide evidence from signature, call sites, and docs (file:line). "
        "If unknown, say explicitly \"unknown from provided context\"."
    )

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "type_analysis"
    assert analysis["needs_type_semantics"] is True
    assert analysis["primary_type_name"] == "ColorMapObject"
    assert "ColorMapObject" in analysis["type_names"]
    assert analysis["function_names"] == []


def test_analyze_query_explicit_struct_fields_intent_is_type_analysis():
    planner = QueryPlanner(
        special_indices={},
        symbols={},
        call_graph={},
        called_by={},
        meta=[],
    )
    q = "Explain struct ColorMapObject and describe what its fields mean."

    analysis = planner.analyze_query(q)

    assert analysis["query_type"] == "type_analysis"
    assert analysis["needs_type_semantics"] is True
    assert analysis["primary_type_name"] == "ColorMapObject"


def test_build_type_analysis_from_context_uses_type_index_field_evidence():
    frags = [
        {
            "name": "DGifDecompressInput",
            "signature": "DGifDecompressInput(GifFileType *GifFile, int *Code)",
            "file": "/repo/giflib/lib/dgif_lib.c",
            "start_line": 975,
            "end_line": 1021,
            "parameters": [
                {"name": "GifFile", "type": "GifFileType *", "raw": "GifFileType *GifFile"},
                {"name": "Code", "type": "int *", "raw": "int *Code"},
            ],
            "code": "int DGifDecompressInput(GifFileType *GifFile, int *Code) { return 0; }",
        }
    ]
    analysis = {
        "primary_type_name": "GifFileType",
        "type_names": ["GifFileType"],
    }
    type_init_index = {
        "types": {
            "GifFileType": [
                {
                    "kind": "pointer_call",
                    "expr": "DGifOpenFileName(path, &Error)",
                    "evidence": [{"file": "/repo/giflib/lib/dgif_lib.c", "line": 120}],
                }
            ]
        },
        "struct_field_writes": {
            "GifFileType": [
                {
                    "path": "Error",
                    "access": "arrow",
                    "count": 3,
                    "sample_expr": "D_GIF_ERR_NOT_READABLE",
                    "evidence": [{"file": "/repo/giflib/lib/dgif_lib.c", "line": 1001}],
                }
            ]
        },
        "function_effects": {},
        "required_fields_by_function": {},
        "init_recipes_by_type": {},
    }

    text = build_type_analysis_from_context(
        frags,
        analysis=analysis,
        type_init_index=type_init_index,
    )

    assert "Type: `GifFileType`" in text
    assert "Field Semantics:" in text
    assert "`Error`" in text
    assert "Observed Initialization Patterns:" in text
    assert "DGifOpenFileName(path, &Error)" in text
