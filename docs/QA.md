# QA — процедуры, метрики качества и контроль покрытия

Документ описывает процесс контроля качества пяти ключевых когнитивных систем
AEGIS (`World Model`, `Cognitive Graph`, `Evolution Engine`, `Goal Intelligence`,
`Real-world Feedback Loop`) и то, как воспроизвести все проверки.

## 1. Уровни тестирования

| Уровень | Что проверяет | Файлы |
|---|---|---|
| Юнит-тесты | Логика каждой системы в изоляции (persistence в `tmp_path`) | `tests/test_world_model.py`, `test_cognitive_graph.py`, `test_evolution_engine.py`, `test_goal_intelligence.py`, `test_feedback_loop.py` |
| Интеграционные | 5 систем внутри реального tick-цикла `Substrate` (offline, без сети/LLM) | `tests/test_higher_systems_integration.py` |
| BDD (Gherkin) | Поведение системы на языке спецификации | `tests/features/higher_systems.feature` + `tests/test_bdd_higher_systems.py` |
| Регрессия ядра | Существующие 89 тестов ядра не сломаны интеграцией | `tests/test_*.py` (память, этика, self-mod, eval, …) |

## 2. Команды запуска

```bash
# Все зависимости для QA
pip install -r requirements-dev.txt

# Весь набор тестов
python -m pytest -q

# Только пять систем
python -m pytest tests/test_world_model.py tests/test_cognitive_graph.py \
    tests/test_evolution_engine.py tests/test_goal_intelligence.py \
    tests/test_feedback_loop.py tests/test_higher_systems_integration.py \
    tests/test_bdd_higher_systems.py -q

# Только Gherkin-сценарии
python -m pytest tests/test_bdd_higher_systems.py -q

# Покрытие с гейтом (fail_under=85 в pyproject.toml)
python -m coverage run -m pytest -q
python -m coverage report            # текстовый отчёт
python -m coverage html -d data/coverage_html   # HTML-отчёт

# Мутационное тестирование
python scripts/mutation_test.py                  # все 5 систем
python scripts/mutation_test.py evolution        # одна система по ключевому слову
```

## 3. Метрики качества (последний прогон)

### 3.1. Пять ключевых систем

| Система | Юнит-тесты | Покрытие (branch) | Мутационный балл |
|---|---:|---:|---:|
| `world_model.py` | 31 | 99% | **100%** (32/32) |
| `cognitive_graph.py` | 22 | 92% | **96.4%** (27/28)¹ |
| `evolution_engine.py` | 16 | 94% | **100%** (12/12) |
| `goal_intelligence.py` | 19 | 97% | **100%** (23/23) |
| `feedback_loop.py` | 17 | 92% | **100%** (15/15) |
| **Итого 5 систем** | **105** | **95%** | **~99%** |

### 3.2. Весь проект (после 2-й волны тестирования)

- **829 тестов, все зелёные.**
- **Общее покрытие проекта: 93%** (было 60%).
- **46 из 47 модулей ≥ 85%**; единственное исключение — `substrate.py` (73%),
  большой async-оркестратор: остаток покрывается косвенно через `test_eval_layer.py`
  на реальном солвере.

Покрытие по группам (branch): нейро/состояние 99%, meta/world 98%, сетевые (с моками)
97%, ethics/safety-core 95–100%, llm 95%, weight_modifier 97%, code_modifier 96%,
state_backup 100%.

Мутационное тестирование (`python scripts/mutation_test.py <module>`):

| Модуль | Мутационный балл | Доведён вручную |
|---|---:|:--:|
| `world_model` | **100%** (32/32) | да |
| `cognitive_graph` | **96.4%** (27/28)¹ | да |
| `evolution_engine` | **100%** (12/12) | да |
| `goal_intelligence` | **100%** (23/23) | да |
| `feedback_loop` | **100%** (15/15) | да |
| `event_bus` | **100%** (5/5) | да |
| `ethics_core` | **100%** (39/39) | да |
| `goal_engine` | **100%** (81/81) | да |
| `self_preservation` | **100%** (41/41) | да |
| `meta_regulation` | 62.9% (22/35) | базовый прогон |

Три safety-критичных модуля (`ethics_core`, `goal_engine`, `self_preservation`)
доведены до 100%: каждый выживший мутант превращён в отдельный тест — см.
`tests/test_ethics_core_mut.py`, `test_goal_engine_mut.py`, `test_self_preservation_mut.py`.
Остальные модули (`meta_regulation` и т.д.) — базовые прогоны; harness готов
(`TARGETS` в `scripts/mutation_test.py` расширяем), доводка итеративная.

**Улучшения harness по ходу доводки:** (1) исправлена индексация узлов — теперь
корректно мутируются вложенные BinOp одного типа на одной строке (`x/2*100`);
(2) отбрасываются вырожденные мутанты (замена оператора на себя); (3) восстановление
исходников бинарное (байт-в-байт, без трансляции окончаний строк).

¹ Единственный выживший мутант в `cognitive_graph.find_path` (`or → and` в проверке
существования узлов) — **доказуемо эквивалентный**: `add_edge` гарантирует, что оба
конца ребра существуют, поэтому BFS не может достичь несуществующего узла, и обе
версии условия дают одинаковый результат (`None`) на всех входах. Не является дефектом.

## 4. Контроль покрытия (quality gate)

Настроен в `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 85          # сборка падает, если покрытие ниже
show_missing = true
exclude_lines = ["logger.exception", "logger.warning", ...]  # обработчики ошибок
```

Правило: покрытие пяти систем не должно опускаться ниже **85%** (текущее — 95%).
Непокрытыми остаются только ветки обработки ошибок (`except → logger.warning`),
которые исключены из подсчёта осознанно.

## 5. Мутационное тестирование

Инструмент: собственный харнесс `scripts/mutation_test.py` (mutmut не работает на
Windows без WSL; mutatest 3.1.0 несовместим с Python 3.11).

**Как работает:** для каждого модуля перечисляются мутанты через AST — переворот
сравнений (`<` ↔ `≥`), арифметики (`+` ↔ `−`, `*` ↔ `/`), булевых операторов
(`and` ↔ `or`), булевых констант (`True` ↔ `False`). Каждый мутант записывается в
файл, прогоняются его тесты, файл всегда восстанавливается (`finally` + резервная
копия `.mut.bak` для устойчивости к жёсткому убийству процесса). Мутант **убит**,
если тесты упали, и **выжил**, если прошли — выживший указывает на пробел в
ассертах.

**Эквивалентные мутанты** исключаются по построению: булевы флаги
`exc_info`/`ensure_ascii`/`indent`/`sort_keys` в вызовах логирования и сериализации
не меняют логику.

**Гейт:** `scripts/mutation_test.py` возвращает ненулевой код выхода, если выжил
хотя бы один (не-эквивалентный) мутант — пригодно для CI.

## 6. Изоляция тестов

Все юнит-тесты пяти систем передают `store_path=tmp_path/...`, поэтому не читают и
не пишут в рабочий каталог `data/`. Интеграционные тесты нейтрализуют сеть
(`agent_system.run_due_agents`), LLM (`llm.enabled=False`) и sandbox
(`environment.step` заглушён), обеспечивая детерминированный offline-прогон.

## 7. Чек-лист перед мержем

1. `python -m pytest -q` — все тесты зелёные.
2. `python -m coverage run -m pytest -q && python -m coverage report` — покрытие ≥ 85%.
3. `python scripts/mutation_test.py` — мутационный балл 100% (кроме задокументированного эквивалентного мутанта).
4. Новые публичные методы систем имеют: юнит-тест, ассерт на границы, и (для нового поведения) Gherkin-сценарий.
5. Любой новый слой подключён к `Substrate` по чек-листу из `docs/АУДИТ.md` (атрибут → фаза tick → `full_status()` → persistence).
