"""https://smart-lab.ru/blog/1191085.php?nomobile=1#1
Стратегия на основе RVI + MA(4) для MCFTR.

Логика (по статье InvestiruyStabilno):
  - MA(4) по RVI пересекает RVI снизу вверх  → сигнал к покупке (страх уходит, ожидается рост)
  - MA(4) по RVI пересекает RVI сверху вниз  → сигнал к продаже (медвежий тренд усиливается)
  - Автор работает на месячном таймфрейме; в коде это включается флагом USE_MONTHLY.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "mcftr_analysis"
TICKER = "MCFTR"
RVI_TICKER = "RVI"
START_DATE = "2004-01-01"  # RVI рассчитывается с 2013 года
TRADING_DAYS = 252
CASH_RATE = 0.1  # безрисковая ставка в кэше (ОФЗ)

RVI_MA_PERIOD = 4  # период MA по RVI, как в статье
USE_MONTHLY = (
    True  # True — ресемплинг на месячный таймфрейм (как в статье); False — дневки
)

# Флаг принудительного обновления котировок.
# False — если локальный CSV существует, читаем его (при необходимости докачиваем новые даты).
# True  — всегда перекачиваем историю с MOEX ISS целиком.
FORCE_REFRESH = True  # False
# Докачивать только новые даты, если локальный файл уже есть (инкрементальное обновление).
INCREMENTAL_UPDATE = True

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (16, 14)
plt.rcParams["font.size"] = 10

request_session = requests.Session()
request_session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)


# ---------- ЗАГРУЗКА С MOEX ISS ----------
def _download_moex_range(ticker, start_date=None, is_index=True):
    """Скачивает историю с MOEX ISS начиная с start_date (или с начала)."""
    if is_index:
        base_url = (
            "https://iss.moex.com/iss/history/engines/stock/markets/index/"
            f"securities/{ticker}.json"
        )
    else:
        base_url = (
            "https://iss.moex.com/iss/history/engines/stock/markets/shares/"
            f"securities/{ticker}.json"
        )
    all_data, start = [], 0
    try:
        resp = request_session.get(base_url, params={"start": 0})
        resp.raise_for_status()
        cursor = resp.json()["history.cursor"]["data"][0]
        total_records, page_size = cursor[1], cursor[2]
        total_pages = (total_records + page_size - 1) // page_size
    except Exception as e:
        print(f"Ошибка при первичном запросе {ticker}: {e}")
        return pd.DataFrame()

    with tqdm(total=total_pages, desc=f"Загрузка {ticker}") as pbar:
        while True:
            params = {"start": start}
            if start_date:
                params["from"] = start_date
            try:
                r = request_session.get(base_url, params=params)
                r.raise_for_status()
                j = r.json()
                history = j["history"]["data"]
                if not history:
                    break
                all_data.extend(history)
                start += page_size
                pbar.update(1)
            except Exception as e:
                print(f"Ошибка на странице {start} для {ticker}: {e}")
                break

    if not all_data:
        return pd.DataFrame()

    columns = j["history"]["columns"]
    df = pd.DataFrame(all_data, columns=columns)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    close_col = "CLOSE" if "CLOSE" in df.columns else "LEGALCLOSEPRICE"
    return df[["TRADEDATE", close_col]].rename(columns={close_col: ticker})


def download_moex_history(ticker, start_date=None, is_index=True):
    """Возвращает историю котировок: из локального CSV, если он есть, иначе с MOEX ISS.

    Поведение:
      - FORCE_REFRESH=True  → всегда качает с ISS и перезаписывает локальный файл.
      - Файл отсутствует    → качает с ISS (начиная с start_date) и сохраняет.
      - Файл существует:
          * INCREMENTAL_UPDATE=True  → докачивает только даты после последней в файле.
          * INCREMENTAL_UPDATE=False → просто читает локальный файл без обращения к сети.
    """
    csv_path = os.path.join(DATA_DIR, f"{ticker}_history.csv")

    # 1. Принудительное обновление — качаем всё заново
    if FORCE_REFRESH or not os.path.exists(csv_path):
        print(
            f"[{ticker}] Локальный файл не найден или включён FORCE_REFRESH — качаем с ISS..."
        )
        fresh = _download_moex_range(ticker, start_date, is_index=is_index)
        if fresh.empty:
            # Если скачать не удалось, но файл есть — используем его как fallback
            if os.path.exists(csv_path):
                print(f"[{ticker}] Не удалось скачать, читаем локальный файл.")
                return pd.read_csv(csv_path, parse_dates=["TRADEDATE"])
            return fresh
        fresh.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[{ticker}] Данные сохранены: {csv_path}")
        return fresh

    # 2. Файл существует — читаем
    print(f"[{ticker}] Локальный файл найден: {csv_path}")
    local = pd.read_csv(csv_path, parse_dates=["TRADEDATE"])

    # 3. Инкрементальное обновление (если включено)
    if INCREMENTAL_UPDATE and not local.empty:
        last_date = local["TRADEDATE"].max()
        # Начинаем со следующего дня после последней даты
        next_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[{ticker}] Докачиваем данные с {next_date}...")
        new_data = _download_moex_range(ticker, next_date, is_index=is_index)
        if not new_data.empty:
            # Убираем возможные дубликаты по дате (на случай пересечения)
            new_data = new_data[new_data["TRADEDATE"] > last_date]
            if not new_data.empty:
                combined = (
                    pd.concat([local, new_data], ignore_index=True)
                    .drop_duplicates(subset=["TRADEDATE"], keep="last")
                    .sort_values("TRADEDATE")
                    .reset_index(drop=True)
                )
                combined.to_csv(csv_path, index=False, encoding="utf-8")
                print(
                    f"[{ticker}] Добавлено {len(new_data)} новых строк. Всего: {len(combined)}."
                )
                return combined
            else:
                print(f"[{ticker}] Новых данных нет.")
        else:
            print(f"[{ticker}] Новых данных нет (пустой ответ).")
    return local


# ---------- ЗАГРУЗКА ДАННЫХ ----------
print("Загрузка данных MCFTR...")
raw_df = download_moex_history(TICKER, START_DATE, is_index=True)
if raw_df.empty:
    print("Не удалось загрузить данные MCFTR.")
    exit()

print("Загрузка данных RVI...")
rvi_df = download_moex_history(RVI_TICKER, START_DATE, is_index=True)
if rvi_df.empty:
    print("Не удалось загрузить данные RVI. Проверьте тикер или доступность ISS.")
    exit()

# Объединение по дате
df = raw_df.merge(rvi_df, on="TRADEDATE", how="inner").set_index("TRADEDATE").copy()
df = df.sort_index()

# ---------- ОПЦИОНАЛЬНЫЙ РЕСЕМПЛИНГ НА МЕСЯЧНЫЙ ТАЙМФРЕЙМ ----------
if USE_MONTHLY:
    df_monthly = df.resample("ME").last().dropna()
else:
    df_monthly = df.copy()

# ---------- РАСЧЁТ СИГНАЛОВ НА ОСНОВЕ RVI + MA ----------
rvi_series_m = df_monthly[RVI_TICKER].copy()
rvi_ma_m = rvi_series_m.rolling(RVI_MA_PERIOD, min_periods=1).mean()

# Пересечение снизу вверх: вчера MA была ниже RVI, сегодня — выше
ma_cross_up = (rvi_ma_m > rvi_series_m) & (rvi_ma_m.shift(1) <= rvi_series_m.shift(1))
# Пересечение сверху вниз: вчера MA была выше RVI, сегодня — ниже
ma_cross_down = (rvi_ma_m < rvi_series_m) & (rvi_ma_m.shift(1) >= rvi_series_m.shift(1))

buy_condition = ma_cross_up
sell_condition = ma_cross_down

regime = 0  # 0 = кэш, 1 = в рынке
regimes = []
for i in range(len(df_monthly)):
    buy_sig = (
        bool(buy_condition.iloc[i]) if not pd.isna(buy_condition.iloc[i]) else False
    )
    sell_sig = (
        bool(sell_condition.iloc[i]) if not pd.isna(sell_condition.iloc[i]) else False
    )

    if regime == 0 and buy_sig:
        regime = 1
    elif regime == 1 and sell_sig:
        regime = 0
    regimes.append(regime)

df_monthly["regime"] = regimes
df_monthly["position"] = df_monthly["regime"].shift(1).fillna(0)  # без look-ahead
df_monthly["rvi_ma"] = rvi_ma_m

# Переносим позиции и MA обратно на дневной индекс
if USE_MONTHLY:
    pos_monthly = df_monthly["position"].shift(1).fillna(0)
    df["position"] = pos_monthly.reindex(df.index, method="ffill").fillna(0)
    df["rvi_ma"] = df_monthly["rvi_ma"].reindex(df.index, method="ffill")
else:
    df["position"] = df_monthly["position"]
    df["rvi_ma"] = df_monthly["rvi_ma"]

df["regime"] = df["position"].shift(-1).fillna(df["position"])
buy_signals = df[df["position"].diff() == 1]
sell_signals = df[df["position"].diff() == -1]

# ---------- ДОХОДНОСТИ ----------
df["ret_mcftr"] = df[TICKER].pct_change().fillna(0)
cash_daily = (1 + CASH_RATE) ** (1 / TRADING_DAYS) - 1
df["ret_strategy"] = (
    df["position"] * df["ret_mcftr"] + (1 - df["position"]) * cash_daily
)

df["equity_strategy"] = (1 + df["ret_strategy"]).cumprod()
df["equity_bh"] = (1 + df["ret_mcftr"]).cumprod()

df["dd_strategy"] = df["equity_strategy"] / df["equity_strategy"].cummax() - 1
df["dd_bh"] = df["equity_bh"] / df["equity_bh"].cummax() - 1


# ---------- МЕТРИКИ ----------
def metrics(equity, returns, name):
    tr = equity.iloc[-1] / equity.iloc[0] - 1
    ny = len(equity) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / ny) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return {
        "Стратегия": name,
        "Итог. доход": f"{tr * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Волатильность": f"{vol * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD": f"{dd * 100:.2f}%",
    }


print("\n===== МЕТРИКИ =====")
summary = pd.DataFrame(
    [
        metrics(
            df["equity_strategy"], df["ret_strategy"], f"RVI + MA({RVI_MA_PERIOD})"
        ),
        metrics(df["equity_bh"], df["ret_mcftr"], "Buy & Hold MCFTR"),
    ]
)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False, encoding="utf-8")
print(f"\nСделок: {len(buy_signals) + len(sell_signals)}")
print(f"Время в рынке: {df['position'].mean() * 100:.1f}%")
print(f"Таймфрейм сигналов: {'месячный (ресемплинг)' if USE_MONTHLY else 'дневной'}")


# ============================================================
# ОБЩИЙ ДАШБОРД (4 панели с общей осью X)
# ============================================================
fig, axes = plt.subplots(
    4,
    1,
    figsize=(16, 16),
    sharex=True,
    gridspec_kw={"height_ratios": [2.5, 2, 2, 1.2]},
)

# --- 1. Цена MCFTR + маркеры покупки/продажи ---
ax = axes[0]
ax.plot(df.index, df[TICKER], color="black", linewidth=1.2, label="MCFTR (лог. шкала)")
if not buy_signals.empty:
    ax.scatter(
        buy_signals.index,
        buy_signals[TICKER],
        color="green",
        s=120,
        marker="^",
        zorder=5,
        edgecolor="white",
        linewidth=1,
        label=f"Покупка ({len(buy_signals)})",
    )
if not sell_signals.empty:
    ax.scatter(
        sell_signals.index,
        sell_signals[TICKER],
        color="red",
        s=120,
        marker="v",
        zorder=5,
        edgecolor="white",
        linewidth=1,
        label=f"Продажа ({len(sell_signals)})",
    )
ax.set_yscale("log")
ax.set_ylabel("MCFTR")
ax.set_title(
    "1. MCFTR: цена и точки входа/выхода по стратегии RVI + MA",
    fontsize=13,
    fontweight="bold",
    loc="left",
)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 2. RVI + MA по RVI ---
ax = axes[1]
ax.plot(
    df.index,
    df[RVI_TICKER],
    color="purple",
    linewidth=1.3,
    label="RVI (индекс волатильности)",
)
ax.plot(
    df.index,
    df["rvi_ma"],
    color="green",
    linewidth=1.3,
    label=f"MA({RVI_MA_PERIOD}) по RVI",
)
if not buy_signals.empty:
    ax.scatter(
        buy_signals.index,
        buy_signals[RVI_TICKER],
        color="green",
        s=80,
        marker="^",
        zorder=5,
        edgecolor="white",
        linewidth=0.8,
        label="Пересечение вверх (покупка)",
    )
if not sell_signals.empty:
    ax.scatter(
        sell_signals.index,
        sell_signals[RVI_TICKER],
        color="red",
        s=80,
        marker="v",
        zorder=5,
        edgecolor="white",
        linewidth=0.8,
        label="Пересечение вниз (продажа)",
    )
ax.set_ylabel("RVI")
ax.set_title(
    f"2. RVI и MA({RVI_MA_PERIOD}): сигналы пересечения",
    fontsize=13,
    fontweight="bold",
    loc="left",
)
ax.legend(loc="upper left", ncol=3)
ax.grid(True, alpha=0.3)

# --- 3. Equity curve ---
ax = axes[2]
ax.plot(
    df.index,
    df["equity_strategy"],
    color="steelblue",
    linewidth=1.8,
    label=f"Стратегия RVI + MA({RVI_MA_PERIOD})",
)
ax.plot(
    df.index,
    df["equity_bh"],
    color="darkorange",
    linewidth=1.8,
    label="Buy & Hold MCFTR",
)
ax.set_yscale("log")
ax.set_ylabel("Стоимость портфеля (лог.)")
ax.set_title(
    "3. Накопленная доходность: стратегия vs Buy & Hold",
    fontsize=13,
    fontweight="bold",
    loc="left",
)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 4. Просадки ---
ax = axes[3]
ax.fill_between(
    df.index,
    df["dd_strategy"] * 100,
    0,
    color="steelblue",
    alpha=0.45,
    label="Стратегия",
)
ax.fill_between(
    df.index, df["dd_bh"] * 100, 0, color="darkorange", alpha=0.45, label="Buy & Hold"
)
ax.set_ylabel("Просадка, %")
ax.set_xlabel("Дата")
ax.set_title("4. Просадки", fontsize=13, fontweight="bold", loc="left")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), rotation=0)

fig.suptitle(
    f"MCFTR — стратегия на RVI + MA({RVI_MA_PERIOD})   |   "
    f"покупка: MA пересекает RVI снизу вверх, "
    f"продажа: MA пересекает RVI сверху вниз   |   "
    f"таймфрейм: {'месячный' if USE_MONTHLY else 'дневной'}",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "dashboard_rvi_ma.png"), dpi=150, bbox_inches="tight"
)
plt.close(fig)

print(f"\nГотово! Дашборд: {os.path.join(OUTPUT_DIR, 'dashboard_rvi_ma.png')}")
