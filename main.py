
from pathlib import Path
import pandas as pd

# Папки проекта
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Файлы NHANES, которые объединяем на первом этапе
FILES = [
    "DEMO_L.XPT",
    "DPQ_L.XPT",
    "SLQ_L.XPT",
    "SMQ_L.XPT",
    "ALQ_L.XPT",
    "PAQ_L.XPT",
    "FSQ_L.XPT",
    "INQ_L.XPT",
]


def main():
    tables = []

    print("Проверка и чтение файлов NHANES...\n")

    for filename in FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"ФАЙЛ НЕ НАЙДЕН: {filename}")
            print(f"Проверь папку: {DATA_DIR}")
            return

        # Читаем файл формата SAS XPORT
        df = pd.read_sas(file_path, format="xport")

        # Убираем возможные пробелы в названиях столбцов
        df.columns = df.columns.str.strip()

        if "SEQN" not in df.columns:
            print(f"ОШИБКА: в {filename} нет столбца SEQN")
            return

        # Проверяем уникальность идентификаторов
        duplicates = df["SEQN"].duplicated().sum()

        print(f"Файл: {filename}")
        print(f"  Строк: {len(df)}")
        print(f"  Столбцов: {len(df.columns)}")
        print(f"  Дубликатов SEQN: {duplicates}")

        if duplicates > 0:
            print("  ОШИБКА: обнаружены повторяющиеся SEQN.")
            print("  Объединение остановлено, чтобы не размножать строки.")
            return

        tables.append((filename, df))

    # Объединяем таблицы по идентификатору участника.
    # outer сохраняет участников, даже если они отсутствуют
    # в некоторых анкетах.
    merged = tables[0][1].copy()

    for filename, df in tables[1:]:
        before = len(merged)

        merged = merged.merge(
            df,
            on="SEQN",
            how="outer",
            validate="one_to_one",
        )

        print(
            f"\nПосле добавления {filename}: "
            f"{before} -> {len(merged)} строк"
        )

    # Сортируем по идентификатору для удобства проверки
    merged = merged.sort_values("SEQN").reset_index(drop=True)

    # Сохраняем объединённую исходную таблицу
    output_file = OUTPUT_DIR / "merged_raw.csv"
    merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nОбъединение завершено.")
    print(f"Итоговый размер: {merged.shape[0]} строк, "
          f"{merged.shape[1]} столбцов")
    print(f"Файл сохранён: {output_file}")

    print("\nПервые строки таблицы:")
    print(merged.head())

    print("\nКоличество заполненных значений по файлам:")
    for filename, df in tables:
        columns = [col for col in df.columns if col != "SEQN"]
        available = merged[columns].notna().any(axis=1).sum()
        print(f"  {filename}: данные есть у {available} участников")


if __name__ == "__main__":
    main()
