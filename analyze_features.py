
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "output" / "model_dataset.csv"
PLOTS_DIR = PROJECT_DIR / "output" / "plots"
SUMMARY_FILE = PROJECT_DIR / "output" / "group_means.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PHQ9_total"

# Настройки отображения
sns.set_theme(style="whitegrid", font="DejaVu Sans")

# Для русских подписей в графиках Matplotlib
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. RUSSIAN FEATURE NAMES
# ============================================================

FEATURE_NAMES = {
    "RIDAGEYR": "Возраст участника",
    "RIAGENDR": "Пол участника",
    "DMDEDUC2": "Уровень образования",
    "DMDMARTZ": "Семейное положение",
    "INDFMPIR": "Отношение дохода семьи к уровню бедности",
    "DMDHHSIZ": "Размер домохозяйства",
    "average_sleep_hours": "Средняя продолжительность сна",
    "SMQ020": "Курение: не менее 100 сигарет за жизнь",
    "SMQ040": "Текущая частота курения",
    "ALQ121": "Примерное число случаев употребления алкоголя в год",
    "ALQ130": "Число порций алкоголя за день употребления",
    "ALQ142": "Примерное число эпизодов интенсивного употребления алкоголя в год",
    "moderate_activity_min_week": "Умеренная физическая активность в неделю",
    "vigorous_activity_min_week": "Интенсивная физическая активность в неделю",
    "sedentary_minutes": "Время сидения в день",
    "FSDAD": "Продовольственная обеспеченность взрослых",
    "INQ300": "Наличие сбережений свыше 5000 долларов",
    "moderate_activity_category": "Категория умеренной физической активности",
    "vigorous_activity_category": "Категория интенсивной физической активности",
}

TARGET_NAME = "Суммарный балл PHQ-9"


# ============================================================
# 3. CATEGORICAL FEATURES AND VALUE LABELS
# ============================================================

# Эти признаки анализируются через swarmplot.
# Числовые коды заменяются понятными русскими подписями.

CATEGORICAL_LABELS = {
    "RIAGENDR": {
        1: "Мужской",
        2: "Женский",
    },
    "DMDEDUC2": {
        1: "Ниже 9 классов",
        2: "9–11 классов",
        3: "Школа / GED",
        4: "Колледж / associate",
        5: "Бакалавриат и выше",
    },
    "DMDMARTZ": {
        1: "В браке / партнёрство",
        2: "Вдовец / разведён / раздельно",
        3: "Никогда не состоял в браке",
    },
    "DMDHHSIZ": {
        1: "1 человек",
        2: "2 человека",
        3: "3 человека",
        4: "4 человека",
        5: "5 человек",
        6: "6 человек",
        7: "7 и более",
    },
    "SMQ020": {
        0: "Нет",
        1: "Да",
    },
    "SMQ040": {
        1: "Каждый день",
        2: "В некоторые дни",
        3: "В настоящее время не курит",
    },
    "FSDAD": {
        1: "Полная обеспеченность",
        2: "Пограничная обеспеченность",
        3: "Низкая обеспеченность",
        4: "Очень низкая обеспеченность",
    },
    "INQ300": {
        0: "Нет",
        1: "Да",
    },
    "moderate_activity_category": {
        "low": "Низкая",
        "medium": "Средняя",
        "high": "Высокая",
    },
    "vigorous_activity_category": {
        "low": "Низкая",
        "medium": "Средняя",
        "high": "Высокая",
    },
}

# Эти признаки анализируются через scatterplot.
CONTINUOUS_FEATURES = [
    "RIDAGEYR",
    "INDFMPIR",
    "average_sleep_hours",
    "ALQ121",
    "ALQ130",
    "ALQ142",
    "moderate_activity_min_week",
    "vigorous_activity_min_week",
    "sedentary_minutes",
]

CATEGORICAL_FEATURES = list(CATEGORICAL_LABELS.keys())


# ============================================================
# 4. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Не найден файл:\n{INPUT_FILE}\n\n"
        "Сначала запусти prepare_features.py и проверь, "
        "что model_dataset.csv находится в папке output."
    )

df = pd.read_csv(INPUT_FILE)

if TARGET not in df.columns:
    raise KeyError(f"В датасете отсутствует целевой признак {TARGET}.")

missing_features = [
    col for col in FEATURE_NAMES if col not in df.columns
]

if missing_features:
    raise KeyError(
        "В датасете отсутствуют признаки:\n"
        + ", ".join(missing_features)
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.loc[df[TARGET].notna()].copy()

print("=" * 70)
print("ВИЗУАЛЬНЫЙ АНАЛИЗ ДАТАСЕТА NHANES")
print("=" * 70)
print(f"Файл: {INPUT_FILE}")
print(f"Количество участников: {len(df)}")
print(f"Количество факторов: {len(FEATURE_NAMES)}")
print(f"Папка с графиками: {PLOTS_DIR}")


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    """Создать безопасное имя файла из названия признака."""
    replacements = {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ",": "",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def add_observation_legend(ax, label="Один участник"):
    """Добавить русскую легенду к графику."""
    handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="steelblue",
        markeredgecolor="steelblue",
        alpha=0.65,
        label=label,
    )

    ax.legend(handles=[handle], loc="best", frameon=True)


def save_figure(fig, filename):
    """Сохранить график и закрыть его."""
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def prepare_categorical_data(feature):
    """Подготовить категориальный признак и русские подписи."""
    data = df[[feature, TARGET]].dropna().copy()

    if data.empty:
        return data, []

    labels = CATEGORICAL_LABELS[feature]
    data[feature] = data[feature].map(labels)

    # Удаляем неизвестные коды, если они встретились.
    data = data.dropna(subset=[feature])

    order = [
        labels[value]
        for value in labels
        if labels[value] in data[feature].values
    ]

    return data, order


def grouped_means_for_categorical(feature):
    """Средний PHQ-9 и размер группы для категориального признака."""
    data, _ = prepare_categorical_data(feature)

    if data.empty:
        return []

    grouped = (
        data.groupby(feature, observed=True)[TARGET]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():
        results.append({
            "feature": feature,
            "название_фактора": FEATURE_NAMES[feature],
            "тип_группировки": "Категория",
            "группа": row[feature],
            "число_наблюдений": int(row["count"]),
            "средний_PHQ9": row["mean"],
            "стандартное_отклонение_PHQ9": row["std"],
            "медиана_PHQ9": row["median"],
        })

    return results


def grouped_means_for_continuous(feature, bins=5):
    """
    Разделить количественный признак на интервалы и рассчитать
    статистики PHQ-9 внутри каждого интервала.

    Для признаков с большим числом повторяющихся значений
    используется qcut с обработкой повторяющихся границ.
    """
    data = df[[feature, TARGET]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=[feature, TARGET])

    if data.empty or data[feature].nunique() < 2:
        return []

    try:
        data["группа"] = pd.qcut(
            data[feature],
            q=bins,
            duplicates="drop",
        )
    except ValueError:
        return []

    grouped = (
        data.groupby("группа", observed=True)[TARGET]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():
        results.append({
            "feature": feature,
            "название_фактора": FEATURE_NAMES[feature],
            "тип_группировки": "Квантильный интервал",
            "группа": str(row["группа"]),
            "число_наблюдений": int(row["count"]),
            "средний_PHQ9": row["mean"],
            "стандартное_отклонение_PHQ9": row["std"],
            "медиана_PHQ9": row["median"],
        })

    return results


# ============================================================
# 6. SCATTERPLOTS FOR CONTINUOUS FEATURES
# ============================================================

print("\nСоздание scatterplot...")

for feature in CONTINUOUS_FEATURES:
    data = df[[feature, TARGET]].copy()

    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna(subset=[feature, TARGET])

    if data.empty:
        print(f"Пропущен график: {feature} — нет данных")
        continue

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=data,
        x=feature,
        y=TARGET,
        ax=ax,
        s=25,
        alpha=0.35,
        color="steelblue",
        edgecolor=None,
        label="Участники выборки",
    )

    # Линия среднего PHQ-9 в квантильных группах.
    # Это вспомогательная линия для визуального сравнения,
    # а не результат проверки статистической значимости.
    try:
        grouped = data.copy()
        grouped["интервал"] = pd.qcut(
            grouped[feature],
            q=5,
            duplicates="drop",
        )

        trend = (
            grouped.groupby("интервал", observed=True)
            .agg(
                x_mean=(feature, "mean"),
                y_mean=(TARGET, "mean"),
            )
            .reset_index()
        )

        if len(trend) >= 2:
            ax.plot(
                trend["x_mean"],
                trend["y_mean"],
                marker="o",
                linewidth=2,
                color="darkorange",
                label="Средний PHQ-9 по интервалам",
            )
    except (ValueError, TypeError):
        pass

    ax.set_title(
        f"Зависимость PHQ-9 от фактора «{FEATURE_NAMES[feature]}»",
        fontsize=12,
    )
    ax.set_xlabel(FEATURE_NAMES[feature])
    ax.set_ylabel(TARGET_NAME)
    ax.legend(loc="best", frameon=True)

    save_figure(
        fig,
        f"scatter_{safe_filename(feature)}.png",
    )

    print(f"  Сохранён scatterplot: {feature}")


# ============================================================
# 7. SWARMPLOTS FOR CATEGORICAL FEATURES
# ============================================================

print("\nСоздание swarmplot...")

for feature in CATEGORICAL_FEATURES:
    data, order = prepare_categorical_data(feature)

    if data.empty or len(order) == 0:
        print(f"Пропущен график: {feature} — нет данных")
        continue

    fig, ax = plt.subplots(figsize=(11, 6))

    # Swarmplot может быть медленным на очень больших группах.
    # При необходимости ограничиваем только отображение точек:
    # таблица средних значений при этом использует все данные.
    plot_data = data

    max_points = 2500

    if len(data) > max_points:
        plot_data = data.sample(
            n=max_points,
            random_state=42,
        )

    sns.swarmplot(
        data=plot_data,
        x=feature,
        y=TARGET,
        order=order,
        ax=ax,
        size=3,
        color="steelblue",
        alpha=0.65,
    )

    # Среднее значение PHQ-9 по каждой категории.
    means = (
        data.groupby(feature, observed=True)[TARGET]
        .mean()
        .reindex(order)
    )

    x_positions = np.arange(len(order))

    ax.scatter(
        x_positions,
        means.values,
        marker="D",
        s=70,
        color="darkorange",
        zorder=5,
        label="Средний PHQ-9 в категории",
    )

    point_legend = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        color="steelblue",
        markersize=5,
        label="Участник выборки",
    )

    mean_legend = Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        color="darkorange",
        markersize=7,
        label="Среднее значение PHQ-9",
    )

    ax.legend(
        handles=[point_legend, mean_legend],
        loc="best",
        frameon=True,
    )

    ax.set_title(
        f"Распределение PHQ-9 по фактору «{FEATURE_NAMES[feature]}»",
        fontsize=12,
    )
    ax.set_xlabel(FEATURE_NAMES[feature])
    ax.set_ylabel(TARGET_NAME)
    ax.tick_params(axis="x", rotation=25)

    save_figure(
        fig,
        f"swarm_{safe_filename(feature)}.png",
    )

    print(f"  Сохранён swarmplot: {feature}")


# ============================================================
# 8. GROUP MEANS FOR ALL FEATURES
# ============================================================

print("\nРасчёт групповых средних значений PHQ-9...")

summary_rows = []

for feature in CATEGORICAL_FEATURES:
    summary_rows.extend(grouped_means_for_categorical(feature))

for feature in CONTINUOUS_FEATURES:
    summary_rows.extend(grouped_means_for_continuous(feature))

summary = pd.DataFrame(summary_rows)

if not summary.empty:
    summary = summary.sort_values(
        by=["feature", "средний_PHQ9"],
        ascending=[True, True],
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Сводная таблица сохранена: {SUMMARY_FILE}")
else:
    print("Не удалось рассчитать сводную таблицу.")


# ============================================================
# 9. SUMMARY REPORT
# ============================================================

print("\n" + "=" * 70)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 70)

print(f"Участников в анализе: {len(df)}")
print(f"Целевой признак: {TARGET_NAME}")
print(f"Графики: {PLOTS_DIR}")
print(f"Групповые средние: {SUMMARY_FILE}")

print("\nСтатистика целевого признака:")
print(df[TARGET].describe().round(3).to_string())

print("\nКоличество пропусков в факторах:")
for feature in FEATURE_NAMES:
    print(
        f"{FEATURE_NAMES[feature]}: "
        f"{df[feature].isna().sum()}"
    )

print("\nПервые строки таблицы групповых средних:")
if not summary.empty:
    print(summary.head(20).round(3).to_string(index=False))

print("\nГотово.")
