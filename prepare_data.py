
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "merged_raw.csv"
OUTPUT_FILE = BASE_DIR / "output" / "phq9_scored.csv"

df = pd.read_csv(INPUT_FILE)

# Девять вопросов, входящих в суммарный балл PHQ-9
phq_columns = [
    "DPQ010", "DPQ020", "DPQ030",
    "DPQ040", "DPQ050", "DPQ060",
    "DPQ070", "DPQ080", "DPQ090",
]

# Проверяем наличие необходимых столбцов
missing_columns = [
    col for col in phq_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Не найдены столбцы PHQ-9: {missing_columns}"
    )

# Убеждаемся, что ответы представлены числами
df[phq_columns] = df[phq_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Коды 7 и 9 означают отказ и неизвестный ответ.
# Они не должны участвовать в расчёте суммы.
df[phq_columns] = df[phq_columns].replace(
    [7, 9], float("nan")
)

# Дополнительная проверка: допустимы только ответы 0–3.
invalid_mask = (
    df[phq_columns].notna()
    & ~df[phq_columns].isin([0, 1, 2, 3])
)

if invalid_mask.any().any():
    raise ValueError(
        "Обнаружены неожиданные значения в ответах PHQ-9."
    )

# Сумма рассчитывается только при наличии всех 9 ответов.
df["PHQ9_total"] = df[phq_columns].sum(
    axis=1,
    min_count=9
)

# Сохраняем промежуточную таблицу.
# Исходный merged_raw.csv не изменяем.
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# Выводим результаты проверки
valid_scores = df["PHQ9_total"].dropna()

print("Подготовка PHQ-9 завершена.")
print(f"Всего строк: {len(df)}")
print(f"Валидных суммарных баллов: {len(valid_scores)}")
print(f"Отсутствующих суммарных баллов: {df['PHQ9_total'].isna().sum()}")

if not valid_scores.empty:
    print(f"Минимальный балл: {valid_scores.min()}")
    print(f"Максимальный балл: {valid_scores.max()}")
    print(f"Средний балл: {valid_scores.mean():.2f}")

    print("\nРаспределение PHQ9_total:")
    print(valid_scores.value_counts().sort_index())

print(f"\nФайл сохранён: {OUTPUT_FILE}")
