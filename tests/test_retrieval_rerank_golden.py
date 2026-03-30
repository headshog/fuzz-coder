from hybrid_code.ask import core


def _chunk(
    idx,
    name,
    file,
    code,
    signature,
    parameters,
    has_stdin=False,
    has_file_input=False,
    has_api_call=False,
    has_output=False,
    uses_memory_management=False,
    has_error_handling=False,
):
    return {
        "id": idx,
        "name": name,
        "file": file,
        "start_line": 1,
        "end_line": 30,
        "code": code,
        "signature": signature,
        "parameters": parameters,
        "has_stdin": has_stdin,
        "has_file_input": has_file_input,
        "has_api_call": has_api_call,
        "has_output": has_output,
        "uses_memory_management": uses_memory_management,
        "has_error_handling": has_error_handling,
        "input_sources": [],
    }


def test_golden_retrieval_rerank_with_path_filter():
    meta = [
        _chunk(
            0,
            "parse_json_payload",
            "/repo/src/parsers/json_parser.cpp",
            "bool parse_json_payload(const std::string &s){ parse_header(s); return true; }",
            "parse_json_payload(const std::string & s)",
            [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
            has_file_input=True,
            uses_memory_management=True,
        ),
        _chunk(
            1,
            "validate_header",
            "/repo/src/parsers/header.cpp",
            "bool validate_header(const uint8_t *data,size_t n){ if(n<4) throw; return true; }",
            "validate_header(const uint8_t * data, size_t n)",
            [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            uses_memory_management=True,
            has_error_handling=True,
        ),
        _chunk(
            2,
            "write_log",
            "/repo/src/io/logger.cpp",
            "void write_log(const std::string &s){ printf(\"%s\", s.c_str()); }",
            "write_log(const std::string & s)",
            [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
            has_output=True,
        ),
        _chunk(
            3,
            "decode_binary_blob",
            "/repo/src/parsers/binary_decoder.cpp",
            "int decode_binary_blob(const uint8_t *buf,size_t n){ memcpy(dst,buf,n); return parse(buf,n); }",
            "decode_binary_blob(const uint8_t * buf, size_t n)",
            [
                {"name": "buf", "type": "const uint8_t *", "raw": "const uint8_t * buf"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            uses_memory_management=True,
        ),
    ]

    special_indices = {
        "stdin": [],
        "file_input": [0],
        "api_calls": [],
        "output": [2],
        "memory_management": [0, 1, 3],
        "error_handling": [1],
        "by_type": {
            "string": [0, 2],
            "byte_array": [1, 3],
            "integer": [1, 3],
        },
    }
    symbols = {c["name"]: [c["id"]] for c in meta}
    planner = core.QueryPlanner(special_indices, symbols, call_graph={}, called_by={}, meta=meta)

    query = "List functions good for fuzzing from module src/parsers"
    analysis = planner.analyze_query(query)

    assert analysis["path_filters"] == ["src/parsers"]
    candidate_ids = planner.get_search_candidates(analysis, k=20)
    assert {0, 1, 3}.issubset(set(candidate_ids))

    lex = {
        "list": [0, 1, 2, 3],
        "functions": [0, 1, 2, 3],
        "fuzzing": [0, 1, 3],
        "module": [0, 1, 3],
        "src": [0, 1, 3],
        "parsers": [0, 1, 3],
    }
    lex_ids, _ = core.lexical_search(query, lex, meta)
    fused_ids, _ = core.reciprocal_rank_fusion([candidate_ids, lex_ids], rrf_k=50)

    reranked = core.rerank_chunks(query, fused_ids[:20], meta, analysis=analysis)
    ranked_pool = [x["idx"] for x in reranked]
    final_ids = core.prefilter_listing_candidates(ranked_pool, meta, analysis)

    assert final_ids
    assert 2 not in final_ids
    assert all("src/parsers" in meta[i]["file"] for i in final_ids)

    best = max(final_ids, key=lambda i: core.fuzz_target_score(meta[i]))
    assert final_ids[0] == best
