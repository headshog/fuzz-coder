from fuzz_coder.ask.example_grounding import ExampleGroundingResolver


def _resolver(meta, symbols=None, call_graph=None, called_by=None):
    return ExampleGroundingResolver(
        meta=meta,
        symbols=symbols or {},
        call_graph=call_graph or {},
        called_by=called_by or {},
    )


def test_grounding_resolver_finds_process_request_from_main():
    meta = [
        {
            "id": 0,
            "name": "process_request",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 7990,
            "end_line": 8242,
            "signature": "process_request(Stream &strm, const std::string &remote_addr, int remote_port)",
            "parameters": [
                {"name": "strm", "type": "Stream &", "raw": "Stream & strm"},
                {"name": "remote_addr", "type": "const std::string &", "raw": "const std::string & remote_addr"},
                {"name": "remote_port", "type": "int", "raw": "int remote_port"},
            ],
            "code": "bool process_request(Stream &strm, const std::string &remote_addr, int remote_port) { return true; }",
        },
        {
            "id": 1,
            "name": "main",
            "file": "/repo/tools/server_main.cpp",
            "start_line": 12,
            "end_line": 120,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "  std::ifstream in(argv[1], std::ios::binary);\n"
                "  Stream s{};\n"
                "  std::string remote = \"127.0.0.1\";\n"
                "  int port = 8080;\n"
                "  server.process_request(s, remote, port);\n"
                "  return 0;\n"
                "}\n"
            ),
        },
    ]
    symbols = {"process_request": [0], "main": [1]}
    call_graph = {
        "0": {"called_by": [1], "resolved_calls": []},
        "1": {"called_by": [], "resolved_calls": [0]},
    }
    analysis = {
        "query_type": "example_generation",
        "primary_function_name": "process_request",
        "function_names": ["process_request", "main"],
        "needs_file": True,
    }

    ctx = _resolver(meta, symbols=symbols, call_graph=call_graph).resolve(
        frags=[meta[0]],
        analysis=analysis,
        candidate_ids=[0, 1],
        top_k=20,
    )

    assert ctx["target"]["name"] == "process_request"
    assert ctx["caller"]["name"] == "main"
    assert "process_request(" in (ctx["observed_call"] or {}).get("expr", "")


def test_grounding_resolver_finds_ma_device_init_dsound_from_main():
    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/player/main.cpp",
            "start_line": 10,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "  ma_device device{};\n"
                "  ma_device_config cfg{};\n"
                "  ma_device_descriptor pb{};\n"
                "  ma_device_descriptor cp{};\n"
                "  ma_device_init__dsound(&device, &cfg, &pb, &cp);\n"
                "  return 0;\n"
                "}\n"
            ),
        },
        {
            "id": 1,
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture)",
            "parameters": [
                {"name": "pDevice", "type": "ma_device*", "raw": "ma_device* pDevice"},
                {"name": "pConfig", "type": "const ma_device_config*", "raw": "const ma_device_config* pConfig"},
                {"name": "pDescriptorPlayback", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorPlayback"},
                {"name": "pDescriptorCapture", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorCapture"},
            ],
            "code": "ma_result ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture) { return MA_SUCCESS; }",
        },
    ]
    symbols = {"main": [0], "ma_device_init__dsound": [1]}
    call_graph = {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    }
    analysis = {
        "query_type": "example_generation",
        "primary_function_name": "ma_device_init__dsound",
        "function_names": ["ma_device_init__dsound", "main"],
        "needs_file": True,
    }

    ctx = _resolver(meta, symbols=symbols, call_graph=call_graph).resolve(
        frags=[meta[1]],
        analysis=analysis,
        candidate_ids=[1, 0],
        top_k=20,
    )

    assert ctx["target"]["name"] == "ma_device_init__dsound"
    assert ctx["caller"]["name"] == "main"
    assert "ma_device_init__dsound(" in (ctx["observed_call"] or {}).get("expr", "")


def test_grounding_resolver_split_without_real_caller_returns_no_caller():
    meta = [
        {
            "id": 0,
            "name": "split",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 1309,
            "end_line": 1329,
            "signature": "split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn)",
            "parameters": [
                {"name": "b", "type": "const char *", "raw": "const char * b"},
                {"name": "e", "type": "const char *", "raw": "const char * e"},
                {"name": "d", "type": "char", "raw": "char d"},
                {"name": "m", "type": "size_t", "raw": "size_t m"},
                {"name": "fn", "type": "std::function<void(const char *, const char *)>", "raw": "std::function<void(const char *, const char *)> fn"},
            ],
            "code": "void split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn) {}",
        },
        {
            "id": 1,
            "name": "helper",
            "file": "/repo/vendor/cpp-httplib/helper.cpp",
            "start_line": 1,
            "end_line": 30,
            "signature": "helper()",
            "parameters": [],
            "code": "void helper() { int x = 0; (void)x; }",
        },
    ]
    symbols = {"split": [0], "helper": [1]}
    call_graph = {
        "0": {"called_by": [], "resolved_calls": []},
        "1": {"called_by": [], "resolved_calls": []},
    }
    analysis = {
        "query_type": "example_generation",
        "primary_function_name": "split",
        "function_names": ["split"],
        "needs_file": True,
        "path_filters": ["vendor/cpp-httplib"],
    }

    ctx = _resolver(meta, symbols=symbols, call_graph=call_graph).resolve(
        frags=[meta[0]],
        analysis=analysis,
        candidate_ids=[0, 1],
        top_k=20,
    )

    assert ctx["target"]["name"] == "split"
    assert ctx["caller"] is None
    assert ctx["observed_call"] is None


def test_grounding_resolver_prefers_non_self_caller_when_available():
    meta = [
        {
            "id": 0,
            "name": "process_request",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 7990,
            "end_line": 8242,
            "signature": "process_request(Stream &strm, const std::string &remote_addr, int remote_port)",
            "parameters": [
                {"name": "strm", "type": "Stream &", "raw": "Stream & strm"},
                {"name": "remote_addr", "type": "const std::string &", "raw": "const std::string & remote_addr"},
                {"name": "remote_port", "type": "int", "raw": "int remote_port"},
            ],
            "code": "bool process_request(Stream &strm, const std::string &remote_addr, int remote_port) { return true; }",
        },
        {
            # same-name overload-like caller candidate (should be fallback only)
            "id": 1,
            "name": "process_request",
            "file": "/repo/vendor/cpp-httplib/impl.cpp",
            "start_line": 9000,
            "end_line": 9100,
            "signature": "process_request(Stream &strm, Request &req, Response &res, bool close_connection, Error &error)",
            "parameters": [],
            "code": "bool process_request(Stream &strm, Request &req, Response &res, bool close_connection, Error &error) { return process_request(strm, \"127.0.0.1\", 80); }",
        },
        {
            "id": 2,
            "name": "main",
            "file": "/repo/tools/server_main.cpp",
            "start_line": 12,
            "end_line": 120,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": "int main(int argc, char ** argv) { Stream s{}; process_request(s, \"127.0.0.1\", 8080); return 0; }",
        },
    ]
    symbols = {"process_request": [0, 1], "main": [2]}
    call_graph = {
        "0": {"called_by": [1, 2], "resolved_calls": []},
        "1": {"called_by": [], "resolved_calls": [0]},
        "2": {"called_by": [], "resolved_calls": [0]},
    }
    analysis = {
        "query_type": "example_generation",
        "primary_function_name": "process_request",
        "function_names": ["process_request", "main"],
        "needs_file": True,
    }

    ctx = _resolver(meta, symbols=symbols, call_graph=call_graph).resolve(
        frags=[meta[0]],
        analysis=analysis,
        candidate_ids=[0, 1, 2],
        top_k=20,
    )

    assert ctx["target"]["name"] == "process_request"
    assert ctx["caller"]["name"] == "main"
