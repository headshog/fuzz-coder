# Продвинутая система анализа кода с AI

Эта система использует современные best practices для глубокого понимания кодовой базы и ответов на сложные вопросы.

## 🚀 Ключевые возможности

### 1. **Умный поиск с переиндексацией (Reranking)**
- Multi-signal scoring: keyword matching + parameter matching + type matching + input type detection
- Cross-encoder reranker для точного ранжирования результатов
- Семантический поиск через FAISS с векторными эмбеддингами

### 2. **Query Planner с анализом намерений**
Система автоматически определяет тип вопроса:
- **Listing queries**: "перечисли функции...", "какие функции..."
- **Type-specific**: "функции с массивом байтов", "принимающие строку"
- **Input-specific**: "читают из stdin", "работают с файлами"
- **Example generation**: "дай пример использования", "как вызвать"
- **Parameter analysis**: "какие параметры принимает...", "что означает каждый параметр"
- **Implementation explanation**: "как работает", "алгоритм"
- **Follow-up вопросы**: контекстные уточнения к предыдущим ответам

### 3. **Thinking Mode с Self-Verification** 🔥
Для сложных запросов модель использует пошаговое мышление:
1. UNDERSTAND - понимание вопроса
2. ANALYZE - анализ сигнатур функций
3. MATCH - сопоставление критериев
4. VERIFY - проверка типов и параметров
5. FORMULATE - формулировка ответа
6. **SELF-VERIFY** - критическая проверка ответа на галлюцинации

### 4. **Защита от галлюцинаций** 🔒
- Модель получает явные инструкции НЕ придумывать функции
- После генерации ответа происходит автоматическая верификация
- Все упомянутые функции сверяются с предоставленным контекстом
- При обнаружении галлюцинаций выводится предупреждение
- В verbose режиме показывается список потенциально выдуманных функций

### 5. **Контекстная память**
- Поддержка диалога с историей (последние 5 обменов)
- Понимание follow-up вопросов ("эти функции", "из списка выше")
- Консистентность ответов в рамках сессии

### 6. **Детекция типов данных**
Распознаёт запросы на русском и английском:
- Массивы байтов: `byte`, `uint8*`, `char*`, `buffer`, `массив байт`
- Строки: `string`, `char*`, `std::string`, `строка`
- Числа: `int`, `int32_t`, `size_t`, `число`
- Векторы/массивы: `vector`, `array`, `список`

### 7. **AST Parsing через Tree-sitter**
- Точное извлечение функций с параметрами
- Правильное определение сигнатур
- Call graph для отслеживания вызовов
- Поддержка профилей языков: `c_cpp`, `java`

### 8. **Расширенная индексация кода** 📊
Новые возможности индексации:
- **Output patterns**: детекция функций вывода (cout, printf)
- **Memory management**: функции с malloc/free/new/delete
- **Error handling**: try/catch/throw, errno, assert
- **Type-based indices**: предварительная классификация по типам
- **Feature tags**: дополнительные метки для точного поиска

## 📦 Установка

```bash
# Установка зависимостей
bash install.sh

# Или вручную:
pip install sentence-transformers faiss-cpu tree_sitter tree_sitter_cpp requests tqdm numpy
# Для Java AST-парсинга:
pip install tree_sitter_java
```

## 🔧 Использование

### Шаг 1: Индексация кодовой базы

```bash
python index_fuzz_coder.py --src /path/to/code.zip --out ./index_data

# Для Java:
python index_fuzz_coder.py --src /path/to/java_project --out ./index_data_java --language java
```

Или для распакованной директории:
```bash
python index_fuzz_coder.py --src /path/to/code_dir --out ./index_data
```

Что индексируется:
- Все функции с их сигнатурами и параметрами
- Типы входных данных (stdin/file/API)
- Call graph (кто кого вызывает)
- Семантические эмбеддинги для поиска
- Лексический индекс для keyword search
- Подсказки из docs/guides (`function_hints.json`), если в документации встречаются имена функций

### Шаг 2: Запуск интерактивного режима

```bash
python ask_fuzz_coder.py \
    --index_dir ./index_data \
    --model qwen2.5-coder:32b \
    --verbose
```

Параметры:
- `--index_dir`: директория с индексом
- `--model`: модель Ollama (по умолчанию qwen2.5-coder)
- `--embed_model`: embedding модель (по умолчанию multilingual-e5-base)
- `--language`: профиль языка индексатора (`c_cpp` или `java`)
- `--top_k`: сколько кандидатов искать (по умолчанию 10)
- `--rerank_top_k`: сколько топ результатов вернуть (по умолчанию 5)
- `--verbose`: показывать детали анализа

Дополнительно (через переменные окружения):
- `FC_QUERY_ANALYZER_LEGACY=1` — включить legacy-анализатор запросов (по умолчанию используется v2)
- `FC_QUERY_ANALYZER_SHADOW=1` — shadow-режим: сравнивать legacy и v2 без смены результата
- `FC_PIPELINE_SHADOW=1` — shadow-режим orchestration: сравнивать основной и shadow-пайплайны без смены ответа пользователю
- `FC_EXAMPLE_GROUNDING_LEGACY=1` — включить legacy-grounding для example_generation (по умолчанию используется v2-grounding)
- `FC_EXAMPLE_GROUNDING_SHADOW=1` — сравнивать legacy и v2 grounding в verbose-режиме без смены ответа

### Шаг 2.1 (опционально): Браузерный чат (Gradio)

Если нужен UI в браузере вместо консоли:

```bash
pip install gradio

# Вариант 1: автообнаружение index_data_* рядом с web_fuzz_coder.py
python web_fuzz_coder.py \
  --model qwen3-coder:30b \
  --host 0.0.0.0 \
  --port 8080

# Вариант 2: явный один индекс
python web_fuzz_coder.py \
  --index_dir ./index_data \
  --model qwen3-coder:30b \
  --host 0.0.0.0 \
  --port 8080

# Вариант 3: базовая авторизация (inline)
python web_fuzz_coder.py \
  --index_base_dir . \
  --model qwen3-coder:30b \
  --host 0.0.0.0 \
  --port 8080 \
  --auth_users "alice:secret,bob:secret2"

# Вариант 4: базовая авторизация из файла
python web_fuzz_coder.py \
  --index_base_dir . \
  --model qwen3-coder:30b \
  --host 0.0.0.0 \
  --port 8080 \
  --auth_users_file ./users.txt
```

Если найдены несколько папок `index_data_PROJECT`, в UI появится выпадающий список проекта.

Браузерный UI поддерживает базовую авторизацию и отдельную историю для каждого пользователя:
- При передаче `--auth_users` или `--auth_users_file` включается login/password.
- История каждого пользователя хранится отдельно в `--history_dir` (по умолчанию `.web_fuzz_histories`).
- После перезапуска сервера истории сохраняются.
- При включенной авторизации в UI появляется кнопка `Logout`.

Форматы `--auth_users_file`:
- `users.txt`:
  - `alice:secret`
  - `bob:secret2`
- `users.json`:
  - `{"alice":"secret","bob":"secret2"}`
  - или `[{"username":"alice","password":"secret"}]`

В браузерном чате поддерживаются те же alias-команды:
- `fuzz`
- `fuzz wide`
- `more fuzz`
- `more fuzz wide`
- `example FUNCTION_NAME`
- `explain FUNCTION_NAME`

### Шаг 2.2 (опционально): Автозапуск через systemd

В репозитории есть готовый unit-файл:
- `deploy/systemd/web_fuzz_coder.service`

Он уже настроен под команду:
- `/home/headshog/.venv/bin/python /home/headshog/coder/web_fuzz_coder.py --model qwen3-coder:30b --host 0.0.0.0 --port 8080 --auth_users_file /home/headshog/coder/users.txt`

Установка сервиса:

```bash
sudo cp deploy/systemd/web_fuzz_coder.service /etc/systemd/system/web_fuzz_coder.service
sudo systemctl daemon-reload
sudo systemctl enable --now web_fuzz_coder.service
```

Проверка:

```bash
sudo systemctl status web_fuzz_coder.service
journalctl -u web_fuzz_coder.service -f
```

### Шаг 3: Задавайте вопросы!

Примеры запросов:

#### Поиск функций по типам параметров
```
> Какие функции принимают на вход массив байтов?
> Перечисли функции, которые получают данные через параметры типа char*
> Найди функции с параметром uint8_t*
```

#### Поиск по источникам ввода
```
> Какие функции читают из стандартного ввода (stdin)?
> Покажи функции, работающие с файлами
> Есть ли функции с HTTP запросами?
```

#### Генерация примеров
```
> Дай пример кода, вызывающий функцию parseData
> Как использовать функцию readFile? Покажи пример
> Напиши пример вызова processBuffer с правильными типами
```

#### Анализ параметров функции
```
> Какие параметры принимает функция llama_params_fit?
> What parameters does process_request take and what does each parameter mean?
> Для функции foo: формат данных по каждому параметру и откуда обычно берутся значения
```

#### Follow-up вопросы (контекстные)
```
> Какие функции принимают массив байтов?
[получает список]
> А какие из них ещё читают из файлов?
[получает уточнённый список]
> Дай пример вызова первой функции из списка
[получает код примера]
```

#### Объяснение реализации
```
> Как работает функция decompressData?
> Объясни алгоритм функции parseHeader
> Что делает функция validateInput?
```

### Embedding модель:
- `intfloat/multilingual-e5-base` - поддерживает русский и английский
- `BAAI/bge-base-en-v1.5` - только английский, но точнее

## 💡 Советы для лучших результатов

1. **Используйте --verbose** для отладки - показывает как система понимает ваш вопрос

2. **Задавайте конкретные вопросы** - "функции с uint8_t*" вместо "функции с байтами"

3. **Для follow-up используйте контекстные ссылки** - "эти функции", "первая из списка"

4. **Комбинируйте критерии** - "функции с массивом байтов И чтением из файла"

5. **Проверяйте verbose вывод** - если система неправильно поняла тип запроса, перефразируйте

## 🔍 Архитектура системы

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Planner  │ ← Анализ намерений, детекция типов
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Candidate      │ ← Special indices + Semantic search
│  Retrieval      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Multi-Signal   │ ← Keyword + Param + Type + Input scoring
│  Reranker       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prompt Builder │ ← Thinking mode + Context history
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM (Ollama)   │ → Ответ пользователю
└─────────────────┘
```

## 🛠 Troubleshooting

### Модель отвечает неверно
- Проверьте `--verbose` - правильно ли определён тип запроса
- Увеличьте `--rerank_top_k` для большего контекста
- Попробуйте перефразировать вопрос более конкретно

### Медленная работа
- Уменьшите `--top_k` и `--rerank_top_k`
- Используйте smaller model (7b вместо 14b)
- Отключите `--verbose`

### Не находит нужные функции
- Проверьте что код корректно проиндексирован
- Попробуйте разные формулировки типов (uint8_t* vs char*)
- Используйте английские термины если русские не работают

## 📊 Сравнение с простой версией

| Возможность | Простая версия | Продвинутая версия |
|-------------|----------------|--------------------|
| Понимание типов параметров | ❌ | ✅ |
| Follow-up вопросы | ❌ | ✅ |
| Thinking mode | ❌ | ✅ |
| Детекция stdin/file/API | ✅ | ✅ Улучшенная |
| Reranking | Базовый | Multi-signal |
| Русский язык | Частично | Полная поддержка |
| Генерация примеров | Нет | ✅ |
| Контекстная память | Нет | 5 последних обменов |

## 📝 Примеры сессий

### Сессия 1: Поиск и примеры
```
> Какие функции принимают buffer или массив байтов?

[Query Analysis]
  Type: type_specific
  Requested types: ['byte_array']

[Ответ модели с списком функций...]

> Дай пример вызова первой функции из списка

[Query Analysis]
  Type: example_generation
  Follow-up: True

[Ответ модели с кодом примера...]
```

### Сессия 2: Комбинированный поиск
```
> Найди функции которые читают из stdin И принимают параметры

[Query Analysis]
  Type: input_specific
  Needs stdin: True
  Needs params: True

[Ответ с функциями удовлетворяющими обоим критериям...]

> А какие из них ещё работают с сокетами?

[Query Analysis]
  Type: input_specific
  Follow-up: True
  Needs API: True

[Уточнённый ответ...]
```

## 🎓 Best Practices использованные в системе

1. **Retrieval-Augmented Generation (RAG)** - поиск релевантного контекста перед генерацией
2. **Hybrid Search** - комбинация семантического и lexical поиска
3. **Cross-Encoder Reranking** - точное ранжирование кандидатов
4. **Query Intent Classification** - понимание типа вопроса
5. **Chain-of-Thought Prompting** - пошаговое мышление для сложных задач
6. **Conversation Memory** - контекст для follow-up вопросов
7. **Multi-Signal Scoring** - множество признаков для релевантности
8. **AST-based Code Parsing** - точное извлечение структуры кода

# Тесты

```bash
RUN_LLAMA_CPP_E2E=1 python3 -m pytest -q tests
```
