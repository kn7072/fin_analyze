import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Папка с данными и папка для сохранения графиков
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_history_charts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Настройки графиков
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 11


def load_all_data(data_dir):
    """Читает все CSV из папки и возвращает словарь {ticker: DataFrame}."""
    data = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка '{data_dir}' не найдена.")

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        ticker = filename.replace("_history.csv", "")
        file_path = os.path.join(data_dir, filename)

        df = pd.read_csv(file_path)
        if "TRADEDATE" not in df.columns or "CLOSE" not in df.columns:
            print(f"Пропускаем {filename}: нет нужных колонок.")
            continue

        # Приводим даты к datetime и сортируем
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df = df.sort_values("TRADEDATE").reset_index(drop=True)
        data[ticker] = df
        print(
            f"Загружено: {ticker} | Строк: {len(df)} | "
            f"Период: {df['TRADEDATE'].min().date()} -> {df['TRADEDATE'].max().date()}"
        )

    return data


def plot_individual(data):
    """Строит отдельный график для каждого индекса."""
    for ticker, df in data.items():
        fig, ax = plt.subplots()
        ax.plot(df["TRADEDATE"], df["CLOSE"], color="steelblue", linewidth=1.3)

        ax.set_title(f"Индекс {ticker}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Значение индекса")

        # Формат дат на оси X
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        # Добавляем подпись последнего значения
        last_date = df["TRADEDATE"].iloc[-1]
        last_value = df["CLOSE"].iloc[-1]
        ax.annotate(
            f"{last_value:.2f}",
            xy=(last_date, last_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color="darkred",
            fontweight="bold",
        )

        file_path = os.path.join(OUTPUT_DIR, f"{ticker}_chart.png")
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"Сохранён график: {file_path}")


def plot_normalized(data):
    """Строит сводный график: все индексы нормализованы к 100 на дату старта."""
    if len(data) < 2:
        print("Недостаточно данных для сводного графика.")
        return

    fig, ax = plt.subplots()

    for ticker, df in data.items():
        # Берём общую дату старта — максимальную из всех минимальных дат
        normalized = df.copy()
        normalized["TRADEDATE"] = pd.to_datetime(normalized["TRADEDATE"])

        # Нормализация к 100 на первой доступной точке
        base_value = normalized["CLOSE"].iloc[0]
        normalized["NORM"] = normalized["CLOSE"] / base_value * 100

        ax.plot(
            normalized["TRADEDATE"],
            normalized["NORM"],
            linewidth=1.5,
            label=f"{ticker} (старт: {base_value:.2f})",
        )

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(
        "Сравнение индексов (нормализовано к 100)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение (старт = 100)")
    ax.legend(loc="best")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    file_path = os.path.join(OUTPUT_DIR, "_comparison_normalized.png")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    print(f"Сохранён сводный график: {file_path}")


def plot_with_moving_averages(data, short_window=50, long_window=200):
    """Строит график с 50- и 200-дневными скользящими средними для каждого индекса."""
    for ticker, df in data.items():
        if len(df) < long_window:
            print(f"Пропускаем {ticker}: недостаточно данных для SMA-{long_window}.")
            continue

        df = df.copy()
        df["SMA50"] = df["CLOSE"].rolling(window=short_window).mean()
        df["SMA200"] = df["CLOSE"].rolling(window=long_window).mean()

        fig, ax = plt.subplots()
        ax.plot(
            df["TRADEDATE"],
            df["CLOSE"],
            color="steelblue",
            linewidth=1.2,
            label="Close",
        )
        ax.plot(
            df["TRADEDATE"],
            df["SMA50"],
            color="orange",
            linewidth=1.4,
            label=f"SMA {short_window}",
        )
        ax.plot(
            df["TRADEDATE"],
            df["SMA200"],
            color="red",
            linewidth=1.4,
            label=f"SMA {long_window}",
        )

        # Отмечаем пересечения (Golden Cross / Death Cross)
        diff = df["SMA50"] - df["SMA200"]
        cross_up = (diff > 0) & (diff.shift(1) <= 0)
        cross_down = (diff < 0) & (diff.shift(1) >= 0)

        ax.scatter(
            df.loc[cross_up, "TRADEDATE"],
            df.loc[cross_up, "SMA50"],
            color="green",
            s=40,
            zorder=5,
            label="Golden Cross",
        )
        ax.scatter(
            df.loc[cross_down, "TRADEDATE"],
            df.loc[cross_down, "SMA50"],
            color="black",
            s=40,
            zorder=5,
            label="Death Cross",
        )

        ax.set_title(
            f"{ticker} с SMA {short_window}/{long_window}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Дата")
        ax.set_ylabel("Значение")
        ax.legend(loc="best")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        file_path = os.path.join(OUTPUT_DIR, f"{ticker}_sma_chart.png")
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"Сохранён график с SMA: {file_path}")


if __name__ == "__main__":
    data = load_all_data(DATA_DIR)

    if not data:
        print("Нет данных для построения графиков.")
    else:
        plot_individual(data)  # Отдельные графики
        plot_normalized(data)  # Сводный нормализованный график
        plot_with_moving_averages(data)  # Графики со скользящими средними
        print("\nГотово! Все графики в папке:", OUTPUT_DIR)
