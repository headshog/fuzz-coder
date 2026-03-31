from __future__ import annotations

from .base import LanguageProfile


JAVA_PROFILE = LanguageProfile(
    name="java",
    supported_ext={".java"},
    control_keywords={
        "if", "for", "while", "switch", "return", "catch",
        "new", "throw", "else", "do", "class", "interface",
        "enum", "record", "package", "import", "var",
    },
    stdin_patterns=[
        r"\b(System\.in)\b",
        r"\bScanner\s*\(\s*System\.in\s*\)",
        r"\bBufferedReader\s*\(\s*new\s+InputStreamReader\s*\(\s*System\.in\s*\)\s*\)",
        r"\b(console\s*\.\s*readLine|Console\s*\.\s*readLine)\s*\(",
    ],
    file_input_patterns=[
        r"\b(FileInputStream|FileReader|BufferedReader|DataInputStream)\b",
        r"\b(Files\.readAllBytes|Files\.readString|Files\.newBufferedReader)\s*\(",
        r"\b(Paths\.get|Path\.of)\s*\(",
    ],
    api_call_patterns=[
        r"\b(HttpClient|HttpRequest|HttpURLConnection|URLConnection)\b",
        r"\b(OkHttpClient|Retrofit|WebClient|RestTemplate)\b",
    ],
    output_patterns=[
        r"\b(System\.out\.(print|println|printf))\s*\(",
        r"\b(Logger|log4j|slf4j)\b",
    ],
    memory_management_patterns=[
        r"\b(ByteBuffer\.(allocate|allocateDirect)|Unsafe|DirectByteBuffer)\b",
        r"\b(new\s+byte\s*\[|ByteArray(Input|Output)Stream)\b",
    ],
    error_handling_patterns=[
        r"\b(try|catch|finally|throw|throws)\b",
        r"\b(assert)\b",
    ],
    type_patterns={
        "byte_array": [r"\b(byte\s*\[\]|ByteBuffer|ByteArray(Input|Output)Stream)\b"],
        "string": [r"\b(String|CharSequence|StringBuilder|StringBuffer)\b"],
        "integer": [r"\b(byte|short|int|long|Integer|Long|Short|BigInteger)\b"],
        "float": [r"\b(float|double|Float|Double|BigDecimal)\b"],
        "pointer": [],
        "reference": [],
        "template": [r"\b(List|Set|Map|ArrayList|HashMap|HashSet)\s*<"],
    },
)
