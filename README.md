# Улучшенная система для работы с кодовой базой

Я полностью переработал вашу систему, добавив все современные best practices для точного поиска и анализа кода.

## Что было улучшено

### 1. **Tree-sitter AST парсинг** (`index_hybrid_code.py`)
- Точное извлечение функций с параметрами через AST
- Правильное определение сигнатур функций
-Fallback на regex если tree-sitter недоступен

### 2. **Детекция типов ввода**
Система автоматически определяет:
- Функции, читающие со **stdin** (cin, scanf, fgets, input(), etc.)
- Функции, читающие из **файлов** (fopen, ifstream, File.Read, etc.)
- Функции, делающие **API вызовы** (requests, fetch, curl, etc.)

### 3. **Call Graph** (граф вызовов)
- Строится граф кто кого вызывает
- Можно искать callers/callees функций
- Помогает находить связанные функции

### 4. **Query Planner** (`ask_hybrid_code.py`)
Анализирует вопрос и выбирает стратегию:
- Распознаёт вопросы про stdin/file/api
- Распознаёт вопросы про параметры функций
- Распознаёт listing queries ("перечисли функции...")
- Использует специальные индексы для разных типов запросов

### 5. **Multi-signal Reranking**
Реранкер использует несколько сигналов:
- Keyword overlap score
- Parameter match score (важно для вопросов про входные данные)
- Input type match score (stdin/file/api)
- Опционально: cross-encoder для ещё лучшей точности

### 6. **Enhanced Prompts**
- Структурированный контекст с параметрами и типами ввода
- Специальные инструкции для listing queries
- Чёткие указания быть точным

## Установка

```bash
# Запустить установку
bash install.sh

# Или вручную:
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers tree_sitter tree_sitter_cpp faiss-cpu cross-encoder requests

# Установить Ollama и модель
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
```

## Использование

### 1. Индексация кода

```bash
# Проиндексировать ZIP с исходниками
python index_hybrid_code.py \
    --src /path/to/code.zip \
    --out ./index_data \
    --embed_model intfloat/multilingual-e5-base
```

Что создаётся в `index_data/`:
- `semantic.faiss` - FAISS индекс для семантического поиска
- `meta.jsonl` - метаданные функций (с параметрами, типами ввода)
- `lexical_index.json` - обратный индекс для keyword search
- `special_indices.json` - индексы для stdin/file/api функций
- `symbols.json` - маппинг имён функций в ID
- `call_graph.json` - граф вызовов
- `called_by.json` - обратный граф

### 2. Поиск по коду

```bash
# Интерактивный режим
python ask_hybrid_code.py \
    --index_dir ./index_data \
    --model qwen2.5-coder:7b \
    --verbose
```

Примеры вопросов:
- "перечисли функции, которые получают данные со стандартного ввода"
- "какие функции читают из файлов?"
- "show all functions that have input parameters"
- "which functions call printf?"
- "find functions that make HTTP requests"

### 3. Ключевые флаги

- `--verbose` - показывает анализ запроса и процесс retrieval
- `--top_k` - сколько кандидатов брать изначально (по умолчанию 10)
- `--rerank_top_k` - сколько оставить после реранкинга (по умолчанию 5)

## Почему это работает лучше

### Было:
```
Вопрос: "перечисли примеры функций, которые получают на вход данные со стандартного ввода"
Ответ: функция "abc" - не имеет во входе параметров вообще
```

### Стало:
1. **Query Planner** распознаёт что вопрос про stdin
2. Берёт candidates из специального `special_indices["stdin"]`
3. **Reranker** даёт высокий boost функциям с `has_stdin=True`
4. **Prompt** содержит явные инструкции перечислять только подходящие функции
5. **Метаданные** включают параметры, так что LLM видит их

## Рекомендации по моделям

Для 24GB VRAM:
- **qwen2.5-coder:7b** - отлично подходит для кода, быстрый
- **codellama:7b** или **13b** - тоже хорошие варианты
- **deepseek-coder:6.7b** - excellent for code tasks

Если хотите "thinking mode" как у ChatGPT - можно использовать model with chain-of-thought prompting, но для задач поиска по коду это обычно избыточно. Лучше потратить ресурсы на:
1. Больший контекст (больше функций в prompt)
2. Лучший retrieval (cross-encoder reranker)
3. Более точный парсинг (tree-sitter)

## Архитектурные улучшения

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Query Planner │ ← Анализирует тип вопроса
            └───────┬────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ Special │ │Semantic │ │ Symbol   │
   │ Indices │ │ Search  │ │ Lookup   │
   └────┬────┘ └────┬────┘ └────┬─────┘
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
            ┌────────────────┐
            │   Reranker     │ ← Multi-signal scoring
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │ Enhanced Prompt│ ← Structured context
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │      LLM       │
            └───────┬────────┘
                    │
                    ▼
              Final Answer
```

## Дополнительные советы

1. **Для больших кодовых баз** (>100k функций):
   - Используйте `--top_k 20 --rerank_top_k 10`
   - Рассмотрите FAISS GPU версию

2. **Для максимальной точности**:
   - Всегда используйте `--verbose` чтобы видеть что происходит
   - Убедитесь что tree-sitter работает (будет писать "parser": "tree-sitter" в meta.jsonl)

3. **Если ответы всё ещё неточные**:
   - Проверьте что detetion input types работает (смотрите special_indices.json)
   - Попробуйте другую модель (qwen2.5-coder обычно лучшая для кода)
   - Увеличьте `--rerank_top_k`
