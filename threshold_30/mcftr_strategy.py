"""https://smart-lab.ru/blog/1149183.php?nomobile=1#1"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm

request_session = requests.Session()
request_session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)
# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "mcftr_analysis"
TICKER = "MCFTR"
START_DATE = "2003-01-01"  # Начинаем с 2003 года, чтобы захватить первые 5-летние окна
TRADING_DAYS = 252

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 11


# ---------- ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ С MOEX ----------
def download_moex_history(ticker, start_date=None):
    """
    Загружает историю дневных цен закрытия для указанного индекса с MOEX ISS API.
    """
    base_url = f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{ticker}.json"
    all_data = []
    start = 0

    # Определяем общее количество записей для прогресс-бара
    try:
        resp = request_session.get(base_url, params={"start": 0})
        resp.raise_for_status()
        total_records = resp.json()["history.cursor"]["data"][0][1]
        page_size = resp.json()["history.cursor"]["data"][0][2]
        total_pages = (total_records + page_size - 1) // page_size
    except Exception as e:
        print(f"Ошибка при первичном запросе: {e}")
        return pd.DataFrame()

    # Загружаем данные постранично
    with tqdm(total=total_pages, desc=f"Загрузка {ticker}") as pbar:
        while True:
            params = {"start": start}
            if start_date:
                params["from"] = start_date
            try:
                response = request_session.get(base_url, params=params)
                response.raise_for_status()
                json_data = response.json()
                history = json_data["history"]["data"]
                if not history:
                    break
                all_data.extend(history)
                start += page_size
                pbar.update(1)
            except Exception as e:
                print(f"Ошибка на странице {start}: {e}")
                break

    if not all_data:
        return pd.DataFrame()

    columns = json_data["history"]["columns"]
    df = pd.DataFrame(all_data, columns=columns)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    # Оставляем только нужные колонки
    return df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": ticker})


# ---------- ЗАГРУЗКА И СОХРАНЕНИЕ ----------
print("Начинаем загрузку исторических данных MCFTR...")
mcftr_df = download_moex_history(TICKER, START_DATE)

if mcftr_df.empty:
    print(
        "Не удалось загрузить данные. Проверьте подключение к интернету или доступность API MOEX."
    )
    exit()

# Сохраняем "сырые" данные в папку moex_history_data
raw_path = os.path.join(DATA_DIR, f"{TICKER}_history.csv")
mcftr_df.to_csv(raw_path, index=False, encoding="utf-8")
print(f"Данные сохранены в: {raw_path}")

# ---------- РАСЧЁТ 5-ЛЕТНЕЙ ДОХОДНОСТИ ----------
# Устанавливаем дату в качестве индекса для удобства
df = mcftr_df.set_index("TRADEDATE").copy()

# Пересэмплируем на конец месяца (последнее доступное значение в месяце)
monthly = df.resample("ME").last().dropna()

# Вычисляем 5-летнюю доходность: (Цена сейчас / Цена 5 лет назад) - 1
# Сдвиг на 60 месяцев (5 лет * 12 месяцев)
monthly["5Y_Return"] = monthly[TICKER].pct_change(periods=60)

# Убираем строки, где 5-летняя доходность ещё не рассчитана (первые 5 лет)
monthly = monthly.dropna(subset=["5Y_Return"])


# ---------- ГРАФИК 1: 5-ЛЕТНЯЯ ДОХОДНОСТЬ MCFTR ----------
fig, ax = plt.subplots()
ax.plot(
    monthly.index,
    monthly["5Y_Return"] * 100,
    color="steelblue",
    linewidth=1.8,
    label="5-летняя доходность MCFTR",
)
ax.axhline(
    y=30, color="green", linestyle="--", linewidth=1.5, label="Порог покупки (30%)"
)
ax.axhline(
    y=180, color="red", linestyle="--", linewidth=1.5, label="Порог продажи (180%)"
)
ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)

ax.set_title(
    "Динамика 5-летней доходности индекса MCFTR (с учётом дивидендов)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Дата")
ax.set_ylabel("Доходность за 5 лет, %")
ax.legend(loc="best")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mcftr_5y_return.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 2: НАКОПЛЕННАЯ ДОХОДНОСТЬ (EQUITY CURVE) ----------
# Рассчитываем накопленную доходность от начальной даты загрузки
df["Cumulative_Return"] = (1 + df[TICKER].pct_change().fillna(0)).cumprod()

fig, ax = plt.subplots()
ax.plot(
    df.index,
    df["Cumulative_Return"],
    color="darkorange",
    linewidth=1.8,
    label="MCFTR (накопленная доходность, начало = 1.0)",
)
ax.set_title(
    "Накопленная доходность индекса MCFTR (Buy & Hold)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Дата")
ax.set_ylabel("Стоимость портфеля (начало = 1.0)")
ax.legend(loc="best")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mcftr_cumulative.png"), dpi=150)
plt.close(fig)


# ---------- ВЫВОД СТАТИСТИКИ ----------
print("\n===== СТАТИСТИКА MCFTR =====")
print(f"Период данных: {df.index.min().date()} -> {df.index.max().date()}")
print(
    f"Итоговая накопленная доходность: {(df['Cumulative_Return'].iloc[-1] - 1) * 100:.2f}%"
)

# Вывод экстремумов 5-летней доходности
max_5y = monthly["5Y_Return"].max()
max_5y_date = monthly["5Y_Return"].idxmax().date()
min_5y = monthly["5Y_Return"].min()
min_5y_date = monthly["5Y_Return"].idxmin().date()

print(f"\nМаксимальная 5-летняя доходность: {max_5y * 100:.2f}% (на {max_5y_date})")
print(f"Минимальная 5-летняя доходность: {min_5y * 100:.2f}% (на {min_5y_date})")

print(f"\nГотово! Графики сохранены в папку: {OUTPUT_DIR}")
print("Файлы: mcftr_5y_return.png, mcftr_cumulative.png")
