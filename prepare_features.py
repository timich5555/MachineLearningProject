
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. ПУТИ К ФАЙЛАМ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "phq9_scored.csv"
OUTPUT_FILE = OUTPUT_DIR / "model_dataset.csv"

# Вспомогательный файл с описанием признаков
DICTIONARY_FILE = OUTPUT_DIR / "feature_dictionary.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def clean_byte_string(value):
    """
    Преобразует значения вроде b'W' в W.
    Пустые байтовые строки b'' превращает в пропуски.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")

    if isinstance(value, str):
        value = value.strip()

        # В CSV байтовые строки могут храниться буквально:
        # "b'W'", "b'D'", "b''".
        if value.startswith("b'") and value.endswith("'"):
            value = value[2:-1]

        if value == "":
            return np.nan

    return value


def to_numeric(series):
    """Преобразует столбец в числа, некорректные значения — в NaN."""
    return pd.to_numeric(series, errors="coerce")


def replace_special_codes(series, codes):
    """
    Заменяет специальные коды NHANES на пропуски.
    Применять только к тем столбцам, для которых
    эти коды подтверждены документацией.
    """
    result = to_numeric(series)
    return result.replace(codes, np.nan)


def weekly_frequency(frequency, unit):
    """
    Переводит частоту занятий в приблизительное
    число занятий за неделю.

    D = день, W = неделя, M = месяц, Y = год.
    """

    if pd.isna(frequency):
        return np.nan

    if frequency < 0:
        return np.nan

    if frequency == 0:
        return 0.0

    if pd.isna(unit):
        return np.nan

    unit = str(unit).strip().upper()

    factors = {
        "D": 7.0,
        "W": 1.0,
        "M": 12.0 / 52.0,
        "Y": 1.0 / 52.0,
    }

    if unit not in factors:
        return np.nan

    return frequency * factors[unit]


def activity_minutes_per_week(frequency, unit, duration):
    """
    Рассчитывает недельный объём физической активности.

    Если частота равна нулю, результат равен нулю.
    Если частота положительная, но длительность неизвестна,
    результат остаётся пропущенным.
    """

    if pd.isna(frequency) or frequency < 0:
        return np.nan

    if frequency == 0:
        return 0.0

    if pd.isna(duration) or duration <= 0:
        return np.nan

    sessions_per_week = weekly_frequency(frequency, unit)

    if pd.isna(sessions_per_week):
        return np.nan

    return sessions_per_week * duration


def activity_category(series):
    """
    Делит валидные значения на три группы по терцилям.
    Нулевые значения входят в нижнюю группу.
    Пропуски сохраняются.
    """

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="string"
    )

    valid = series.dropna()

    if valid.empty:
        return result

    lower_boundary = valid.quantile(1 / 3)
    upper_boundary = valid.quantile(2 / 3)

    result.loc[valid.index[valid <= lower_boundary]] = "low"

    result.loc[
        valid.index[
            (valid > lower_boundary) &
            (valid <= upper_boundary)
        ]
    ] = "medium"

    result.loc[valid.index[valid > upper_boundary]] = "high"

    return result


# ============================================================
# 3. ЧТЕНИЕ И ПРОВЕРКА ИСХОДНОГО CSV
# ============================================================

def main():

    if not INPUT_FILE.exists():
        print(f"ОШИБКА: не найден файл {INPUT_FILE}")
        print("Сначала запусти скрипт prepare_data.py.")
        return

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("Исходная таблица:", df.shape)

    required_columns = [
        "SEQN",
        "PHQ9_total",
    ]

    for column in required_columns:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

    # Убираем возможные очень маленькие значения,
    # которые фактически представляют нули.
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].map(clean_byte_string)

    # Приводим целевую переменную к числовому типу.
    df["PHQ9_total"] = to_numeric(df["PHQ9_total"])


    # ========================================================
    # 4. ДЕМОГРАФИЧЕСКИЕ ПРИЗНАКИ
    # ========================================================

    demographic_columns = [
        "RIDAGEYR",
        "RIAGENDR",
        "DMDEDUC2",
        "DMDMARTZ",
        "INDFMPIR",
        "DMDHHSIZ",
    ]

    for column in demographic_columns:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

        df[column] = to_numeric(df[column])

    # Образование: специальные коды.
    df["DMDEDUC2"] = df["DMDEDUC2"].replace(
        {7: np.nan, 9: np.nan}
    )

    # Семейное положение: специальные коды.
    df["DMDMARTZ"] = df["DMDMARTZ"].replace(
        {77: np.nan, 99: np.nan}
    )

    # RIDAGEYR, RIAGENDR, INDFMPIR и DMDHHSIZ
    # сохраняем в исходной числовой кодировке.


    # ========================================================
    # 5. СОН: СРЕДНЕВЗВЕШЕННОЕ ЗНАЧЕНИЕ
    # ========================================================

    sleep_columns = ["SLD012", "SLD013"]

    for column in sleep_columns:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

        df[column] = to_numeric(df[column])

    # Рассчитываем среднее только при наличии обоих ответов.
    sleep_valid = df["SLD012"].notna() & df["SLD013"].notna()

    df["average_sleep_hours"] = np.nan

    df.loc[sleep_valid, "average_sleep_hours"] = (
        5 * df.loc[sleep_valid, "SLD012"]
        + 2 * df.loc[sleep_valid, "SLD013"]
    ) / 7


    # ========================================================
    # 6. КУРЕНИЕ
    # ========================================================

    for column in ["SMQ020", "SMQ040"]:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

        df[column] = to_numeric(df[column])

    # SMQ020: 1 = да, 2 = нет.
    df["SMQ020"] = df["SMQ020"].replace(
        {1: 1, 2: 0, 7: np.nan, 9: np.nan}
    )

    # SMQ040: 1 = ежедневно, 2 = в некоторые дни,
    # 3 = совсем не курит.
    df["SMQ040"] = df["SMQ040"].replace(
        {1: 3, 2: 2, 3: 1, 7: np.nan, 9: np.nan}
    )


    # ========================================================
    # 7. АЛКОГОЛЬ
    # ========================================================

    alcohol_columns = ["ALQ121", "ALQ130", "ALQ142"]

    for column in alcohol_columns:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

        df[column] = to_numeric(df[column])

    # Сохраняем исходную частоту употребления,
    # чтобы корректно обработать ALQ130.
    original_alq121 = df["ALQ121"].copy()

    # Частота употребления в год.
    annual_frequency = {
        0: 0,
        10: 1,
        9: 4.5,
        8: 9,
        7: 12,
        6: 30,
        5: 52,
        4: 104,
        3: 182,
        2: 330,
        1: 365,
    }

    df["ALQ121"] = original_alq121.map(annual_frequency)

    # ALQ130: число порций в день употребления.
    df["ALQ130"] = df["ALQ130"].replace(
        {777: np.nan, 999: np.nan}
    )

    # Не употреблял алкоголь за последний год:
    # среднее число порций в день употребления
    # кодируем нулём согласно принятому решению.
    no_alcohol = original_alq121 == 0
    df.loc[no_alcohol, "ALQ130"] = 0

    # Если ALQ121 отсутствует, нельзя автоматически
    # делать вывод об отсутствии употребления.
    unknown_alcohol = original_alq121.isna()
    df.loc[unknown_alcohol, "ALQ130"] = np.nan

    # ALQ142: частота эпизодов интенсивного употребления.
    df["ALQ142"] = df["ALQ142"].map(annual_frequency)


    # ========================================================
    # 8. ФИЗИЧЕСКАЯ АКТИВНОСТЬ
    # ========================================================

    activity_columns = [
        "PAD790Q", "PAD790U", "PAD800",
        "PAD810Q", "PAD810U", "PAD820",
        "PAD680",
    ]

    for column in activity_columns:
        if column not in df.columns:
            print(f"ОШИБКА: отсутствует столбец {column}")
            return

    # Единицы частоты: b'W' -> W, b'' -> NaN.
    for column in ["PAD790U", "PAD810U"]:
        df[column] = df[column].map(clean_byte_string)

    # Числовые значения.
    for column in [
        "PAD790Q", "PAD800",
        "PAD810Q", "PAD820", "PAD680",
    ]:
        df[column] = to_numeric(df[column])

    # Специальные коды частоты.
    df["PAD790Q"] = df["PAD790Q"].replace(
        {7777: np.nan, 9999: np.nan}
    )

    df["PAD810Q"] = df["PAD810Q"].replace(
        {7777: np.nan, 9999: np.nan}
    )

    # Специальные коды длительности.
    df["PAD800"] = df["PAD800"].replace(
        {7777: np.nan, 9999: np.nan}
    )

    df["PAD820"] = df["PAD820"].replace(
        {7777: np.nan, 9999: np.nan}
    )

    # Специальные коды сидячего времени.
    df["PAD680"] = df["PAD680"].replace(
        {7777: np.nan, 9999: np.nan}
    )

    # Рассчитываем недельную активность.
    df["moderate_activity_min_week"] = [
        activity_minutes_per_week(f, u, d)
        for f, u, d in zip(
            df["PAD790Q"],
            df["PAD790U"],
            df["PAD800"],
        )
    ]

    df["vigorous_activity_min_week"] = [
        activity_minutes_per_week(f, u, d)
        for f, u, d in zip(
            df["PAD810Q"],
            df["PAD810U"],
            df["PAD820"],
        )
    ]

    df["sedentary_minutes"] = df["PAD680"]

    # Категории строим по валидным значениям.
    # Нулевые значения входят в нижнюю группу.
    df["moderate_activity_category"] = activity_category(
        df["moderate_activity_min_week"]
    )

    df["vigorous_activity_category"] = activity_category(
        df["vigorous_activity_min_week"]
    )


    # ========================================================
    # 9. ПРОДОВОЛЬСТВЕННАЯ ОБЕСПЕЧЕННОСТЬ
    # ========================================================

    if "FSDAD" not in df.columns:
        print("ОШИБКА: отсутствует столбец FSDAD")
        return

    df["FSDAD"] = to_numeric(df["FSDAD"])

    # Сохраняем категории 1–4 без искусственного
    # изменения расстояний между ними.


    # ========================================================
    # 10. ФИНАНСОВЫЕ АКТИВЫ
    # ========================================================

    if "INQ300" not in df.columns:
        print("ОШИБКА: отсутствует столбец INQ300")
        return

    df["INQ300"] = to_numeric(df["INQ300"])

    # Предполагаемая кодировка: 1 = да, 2 = нет.
    # Если кодбук показывает другое, эту строку
    # необходимо скорректировать.
    df["INQ300"] = df["INQ300"].replace(
        {1: 1, 2: 0, 7: np.nan, 9: np.nan}
    )


    # ========================================================
    # 11. ФОРМИРУЕМ ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
    # ========================================================

    feature_columns = [
        "RIDAGEYR",
        "RIAGENDR",
        "DMDEDUC2",
        "DMDMARTZ",
        "INDFMPIR",
        "DMDHHSIZ",
        "average_sleep_hours",
        "SMQ020",
        "SMQ040",
        "ALQ121",
        "ALQ130",
        "ALQ142",
        "moderate_activity_min_week",
        "vigorous_activity_min_week",
        "sedentary_minutes",
        "FSDAD",
        "INQ300",
    ]

    categorical_alternatives = [
        "moderate_activity_category",
        "vigorous_activity_category",
    ]

    # Сохраняем только выбранные признаки и целевую переменную.
    final_columns = [
        "SEQN",
        "PHQ9_total",
        *feature_columns,
        *categorical_alternatives,
    ]

    final_df = df[final_columns].copy()

    # Для обучения регрессии нужны известные целевые значения.
    rows_before = len(final_df)
    final_df = final_df.dropna(subset=["PHQ9_total"]).copy()
    rows_after = len(final_df)


    # ========================================================
    # 12. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Словарь признаков: название и смысл каждого поля.
    descriptions = {
        "SEQN": "Идентификатор участника NHANES",
        "PHQ9_total": "Целевая переменная: сумма девяти вопросов PHQ-9",
        "RIDAGEYR": "Возраст в годах; верхнее значение 80 означает 80 лет и старше",
        "RIAGENDR": "Пол по кодировке NHANES",
        "DMDEDUC2": "Уровень образования взрослых",
        "DMDMARTZ": "Семейное положение",
        "INDFMPIR": "Отношение семейного дохода к порогу бедности",
        "DMDHHSIZ": "Размер домохозяйства; 7 означает 7 человек и более",
        "average_sleep_hours": "Средняя продолжительность сна в сутки за неделю",
        "SMQ020": "Курил ли участник не менее 100 сигарет за жизнь: 0/1",
        "SMQ040": "Текущий статус курения: 1 = не курит, 2 = иногда, 3 = ежедневно",
        "ALQ121": "Приблизительное число случаев употребления алкоголя в год",
        "ALQ130": "Среднее число стандартных порций в день употребления",
        "ALQ142": "Приблизительное число эпизодов интенсивного употребления в год",
        "moderate_activity_min_week": "Минуты умеренной досуговой активности за неделю",
        "vigorous_activity_min_week": "Минуты интенсивной досуговой активности за неделю",
        "sedentary_minutes": "Минуты сидячего времени в обычный день",
        "moderate_activity_category": "Категория умеренной активности по терцилям",
        "vigorous_activity_category": "Категория интенсивной активности по терцилям",
        "FSDAD": "Категория продовольственной обеспеченности взрослых",
        "INQ300": "Ответ на вопрос о финансовых активах; кодировку проверить по кодбуку",
    }

    dictionary_df = pd.DataFrame(
        [
            {"column": column, "description": descriptions.get(column, "")}
            for column in final_columns
        ]
    )

    dictionary_df.to_csv(
        DICTIONARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 13. ОТЧЁТ О ПОДГОТОВКЕ
    # ========================================================

    print("\nПодготовка признаков завершена.")
    print(f"Строк до фильтрации целевой переменной: {rows_before}")
    print(f"Строк с валидным PHQ9_total: {rows_after}")
    print(f"Столбцов в итоговой таблице: {len(final_columns)}")

    print("\nКоличество пропусков по признакам:")
    print(final_df.isna().sum().to_string())

    print("\nРаспределение целевой переменной:")
    print(final_df["PHQ9_total"].describe())

    print("\nИтоговая таблица:")
    print(final_df.head())

    print(f"\nCSV с данными: {OUTPUT_FILE}")
    print(f"Словарь признаков: {DICTIONARY_FILE}")


if __name__ == "__main__":
    main()
