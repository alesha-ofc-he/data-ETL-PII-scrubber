# Customer churn research

Воспроизводимая подготовка данных для модели оттока клиентов. Сейчас реализованы загрузка [UCI Iranian Churn](https://archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset), проверка схемы, контрольная сумма локального снимка данных, разбиение без пересечения одинаковых строк и исследовательский [ноутбук](research/01_data_and_model_research.ipynb).

Датасет содержит 3150 строк, из них 495 с оттоком. В train обнаружены 9 групп с одинаковыми доступными признаками и разными метками; ограничения и принятые решения описаны в [data card](docs/data_card.md). Модели пока не обучены.

## Первый запуск из корня репозитория

```bash
uv sync --all-extras
uv run retention-lab fetch-data --config configs/full.toml
uv run retention-lab validate-data --config configs/full.toml
uv run retention-lab split-data --config configs/full.toml
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run jupyter nbconvert --execute --to notebook --inplace research/01_data_and_model_research.ipynb
```

`uv` установит Python 3.12 при необходимости. Первый вызов `fetch-data` требует сеть, остальные используют локальную копию в игнорируемом Git каталоге `data/`. Ноутбук хранит исследование; расчёты данных размещены в `src/retention_lab/`. Ноутбук исполнен на snapshot с SHA-256 `1f7dfa2170e2a15d16cb9ca78e45e88b77cc60b9eb1c1c05d89fbb480185a45c`; три сохранённых графика проверены.

Источник распространяется по лицензии CC BY 4.0. Это телеком-данные без дат отдельных клиентских событий; они не позволяют честно заявлять качество на будущем календарном периоде или эффект кампании удержания.
