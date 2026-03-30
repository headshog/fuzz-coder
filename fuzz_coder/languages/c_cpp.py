from __future__ import annotations

from .base import LanguageProfile


C_CPP_PROFILE = LanguageProfile(
    name="c_cpp",
    supported_ext={".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"},
    control_keywords={
        "if", "for", "while", "switch", "return", "sizeof", "catch",
        "new", "delete", "throw", "else", "do", "class", "struct",
        "namespace", "template", "typedef", "using", "enum", "union"
    },
    stdin_patterns=[
        r"\b(cin)\s*>>",
        r"\b(scanf|getchar|getc|gets|gets_s)\s*\(",
        r"\b(fgets|fscanf)\s*\([^)]*(stdin)\b",
        r"\bstd::getline\s*\(\s*(std::)?cin\b",
        r"\bgetline\s*\(\s*(std::)?cin\b",
        r"\bread\s*\(\s*(0|STDIN_FILENO)\b",
        r"\b(std::)?cin\b",
        r"\b(System\.Console\.Read)\b",
        r"\b(Console\.Read|ReadLine|ReadKey)\b",
        r"\b(input|raw_input)\s*\(",
        r"\b(sys\.stdin|process\.stdin)\b",
    ],
    file_input_patterns=[
        r"\bf(open|fopen|ifstream|fstream)\s*\(",
        r"\b(fread|fgets|fscanf|fgetc|getc)\s*\(",
        r"\b(std::)?(ifstream|fstream|ofstream)\b",
        r"\b(File\.Open|File\.Read|StreamReader)\b",
        r"\b(fs\.readFileSync?|fs\.createReadStream)\b",
        r"\b(Path\.OpenText|File\.ReadAllText)\b",
    ],
    api_call_patterns=[
        r"\b(http_client|HttpClient|curl_easy|wget)\b",
        r"\b(requests\.(get|post|put|delete|patch))\b",
        r"\b(fetch|axios|XMLHttpRequest)\b",
        r"\b(urllib\.(request|urlopen))\b",
        r"\b(httplib::Client|boost::beast)\b",
    ],
    output_patterns=[
        r"\b(cout|printf|fprintf|sprintf)\b",
        r"\b(std::)?(cout|cerr|clog|print|writeln)\b",
        r"\b(Console\.Write|System\.out)\b",
    ],
    memory_management_patterns=[
        r"\b(malloc|calloc|realloc|free)\b",
        r"\b(new|delete)\b",
        r"\b(shared_ptr|unique_ptr|weak_ptr)\b",
    ],
    error_handling_patterns=[
        r"\b(throw|try|catch|finally)\b",
        r"\b(errno|perror|strerror)\b",
        r"\b(assert|static_assert)\b",
    ],
    type_patterns={
        "byte_array": [r"\b(uint8_t|unsigned\s+char|char\s*\*|std::vector<uint8_t>|QByteArray|ByteBuffer)\b"],
        "string": [r"\b(std::string|char\s*\*|const\s+char\s*\*|QString|std::wstring)\b"],
        "integer": [r"\b(int|long|short|int32_t|int64_t|size_t|ssize_t)\b"],
        "float": [r"\b(float|double|long\s+double)\b"],
        "pointer": [r"\w+\s*\*\s*\w+"],
        "reference": [r"\w+\s*&\s*\w+"],
        "template": [r"\b(std::vector|std::map|std::set|std::unordered_map|std::array)\b"],
    },
)
