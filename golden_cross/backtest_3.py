"""
Корректный бэктест стратегии High-Yield Golden Cross на российском рынке.

Исправления относительно "наивной" версии:
  1. Источник сигналов — RUCBHYTR (высокодоходные облигации), а не RGBI.
  2. Доходность в кэше — реальная ключевая ставка ЦБ РФ на каждый день, а не фиксированные 12%.
  3. Опциональный фильтр по RGBITR: игнорировать сигнал на покупку, если RGBITR < SMA200(RGBITR).
  4. Сдвиг сигнала на 1 день (без look-ahead bias).
  5. Сравнение с Buy & Hold на MCFTR.
"""

import os
import time
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_backtest_correct"
SIGNAL_TICKER = "RUCBHYTR"  # источник сигналов
TRADE_TICKER = "MCFTR"  # торгуемый актив
FILTER_TICKER = "RGBITR"  # опциональный фильтр (гособлигации, полная доходность)
SHORT_WINDOW = 50
LONG_WINDOW = 200
USE_FILTER = True  # True — включить фильтр по RGBITR
TRADING_DAYS = 252
START_DATE = "2019-12-30"  # начало истории RUCBHYTR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 11

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)


# ---------- ЗАГРУЗКА ДАННЫХ С MOEX ISS API ----------
def fetch_moex_history(ticker, start_date):
    """Скачивает полную историю индекса с MOEX ISS API."""
    print(f"  Загрузка {ticker}...")
    base_url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/index/"
        f"securities/{ticker}.json"
    )
    all_rows = []
    start_index = 0

    while True:
        params = {
            "from": start_date,
            "start": start_index,
            "iss.only": "history",
            "history.columns": "TRADEDATE,CLOSE",
        }
        try:
            r = session.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Ошибка: {e}")
            break

        history = data.get("history", {})
        columns = history.get("columns", [])
        rows = history.get("data", [])
        if not rows:
            break

        all_rows.extend([dict(zip(columns, row)) for row in rows])
        start_index += len(rows)
        time.sleep(0.05)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    return df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": ticker})


def fetch_key_rate_history():
    """
    Возвращает DataFrame с историей ключевой ставки ЦБ РФ.
    Данные вручную внесены из официальных источников.
    """
    # Данные: (дата начала действия, ставка в %)
    rates = [
        ("2019-12-16", 6.25),
        ("2020-02-10", 6.00),
        ("2020-04-27", 5.50),
        ("2020-06-22", 4.50),
        ("2020-07-27", 4.25),
        ("2021-03-22", 4.50),
        ("2021-04-26", 5.00),
        ("2021-06-15", 5.50),
        ("2021-07-26", 6.50),
        ("2021-09-13", 6.75),
        ("2021-10-25", 7.50),
        ("2021-12-20", 8.50),
        ("2022-02-14", 9.50),
        ("2022-02-28", 20.00),
        ("2022-04-11", 17.00),
        ("2022-05-04", 14.00),
        ("2022-05-27", 11.00),
        ("2022-06-14", 9.50),
        ("2022-07-25", 8.00),
        ("2022-09-19", 7.50),
        ("2023-07-24", 8.50),
        ("2023-08-15", 12.00),
        ("2023-09-18", 13.00),
        ("2023-10-30", 15.00),
        ("2023-12-18", 16.00),
        ("2024-07-29", 18.00),
        ("2024-09-16", 19.00),
        ("2024-10-28", 21.00),
        ("2025-06-09", 20.00),
        ("2025-07-28", 18.00),
        ("2025-09-15", 17.00),
        ("2025-10-27", 16.50),
        ("2025-12-22", 16.00),
    ]
    df = pd.DataFrame(rates, columns=["DATE", "RATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.sort_values("DATE").reset_index(drop=True)


def attach_daily_rate(df, rate_df):
    """Прикрепляет к каждому дню бэктеста действующую ключевую ставку."""
    df = df.sort_values("TRADEDATE").copy()
    rate_df = rate_df.sort_values("DATE").copy()
    # merge_asof: для каждой даты в df берём последнюю ставку, действовавшую на тот момент
    merged = pd.merge_asof(
        df,
        rate_df,
        left_on="TRADEDATE",
        right_on="DATE",
        direction="backward",
    )
    # Если ставка ещё не действовала — берём первую доступную
    merged["RATE"] = merged["RATE"].bfill().ffill()
    return merged.drop(columns=["DATE"])


# ---------- ЗАГРУЗКА ВСЕХ ДАННЫХ ----------
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ С MOEX ISS API")
print("=" * 60)

signal_df = fetch_moex_history(SIGNAL_TICKER, START_DATE)
trade_df = fetch_moex_history(TRADE_TICKER, START_DATE)
filter_df = fetch_moex_history(FILTER_TICKER, START_DATE) if USE_FILTER else None

if signal_df is None or trade_df is None:
    raise SystemExit("Не удалось загрузить базовые данные.")

rate_df = fetch_key_rate_history()

# ---------- ОБЪЕДИНЕНИЕ ----------
merged = pd.merge(signal_df, trade_df, on="TRADEDATE", how="inner")
if filter_df is not None:
    merged = pd.merge(merged, filter_df, on="TRADEDATE", how="inner")
merged = merged.sort_values("TRADEDATE").reset_index(drop=True)

# Прикрепляем ключевую ставку
merged = attach_daily_rate(merged, rate_df)

print(
    f"\nОбщий период: {merged['TRADEDATE'].min().date()} -> {merged['TRADEDATE'].max().date()}"
)
print(f"Дней в выборке: {len(merged)}")


# ---------- СИГНАЛЫ ----------
merged["SMA_short"] = merged[SIGNAL_TICKER].rolling(SHORT_WINDOW).mean()
merged["SMA_long"] = merged[SIGNAL_TICKER].rolling(LONG_WINDOW).mean()

# Базовый сигнал: 1 — держим акции, 0 — в кэше
merged["base_signal"] = np.where(merged["SMA_short"] > merged["SMA_long"], 1, 0)

# Опциональный фильтр по RGBITR
if USE_FILTER:
    merged["FILTER_SMA200"] = merged[FILTER_TICKER].rolling(LONG_WINDOW).mean()
    merged["filter_ok"] = merged[FILTER_TICKER] > merged["FILTER_SMA200"]
else:
    merged["filter_ok"] = True

# Итоговый режим: в акциях только если базовый сигнал = 1 И фильтр ОК
merged["regime"] = np.where((merged["base_signal"] == 1) & (merged["filter_ok"]), 1, 0)

# Сдвиг на 1 день — позицию на сегодня берём из сигнала вчерашнего дня
merged["position"] = merged["regime"].shift(1).fillna(0).astype(int)

# Дневная доходность MCFTR
merged["ret_mcftr"] = merged[TRADE_TICKER].pct_change().fillna(0)

# Дневная ставка кэша из ключевой ставки (простое приведение: rate / 365)
merged["cash_daily"] = merged["RATE"] / 100 / 365

# Доходность стратегии
merged["ret_strategy"] = (
    merged["position"] * merged["ret_mcftr"]
    + (1 - merged["position"]) * merged["cash_daily"]
)

# Кумулятивные кривые
merged["equity_strategy"] = (1 + merged["ret_strategy"]).cumprod()
merged["equity_bh"] = (1 + merged["ret_mcftr"]).cumprod()


# ---------- МЕТРИКИ ----------
def compute_metrics(equity, returns, name):
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(equity) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "Стратегия": name,
        "Итоговая доходность": f"{total_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Волатильность (год)": f"{vol * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Calmar": f"{calmar:.2f}",
    }, drawdown


metrics_strategy, dd_strategy = compute_metrics(
    merged["equity_strategy"],
    merged["ret_strategy"],
    "High-Yield Golden Cross" + (" + RGBITR фильтр" if USE_FILTER else ""),
)
metrics_bh, dd_bh = compute_metrics(
    merged["equity_bh"], merged["ret_mcftr"], "MCFTR Buy & Hold"
)

summary = pd.DataFrame([metrics_strategy, metrics_bh])
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
print("=" * 60)
print(summary.to_string(index=False))

n_trades = int((merged["position"].diff().abs() == 1).sum())
time_in_market = merged["position"].mean() * 100
print(f"\nСделок (переходов): {n_trades}")
print(f"Время в рынке: {time_in_market:.1f}%")
print(f"Время в кэше:  {100 - time_in_market:.1f}%")
print(f"Средняя ставка в кэше: {merged['RATE'].mean():.2f}%")

summary.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False, encoding="utf-8")


# ---------- ГРАФИК 1: EQUITY CURVE ----------
fig, ax = plt.subplots()
ax.plot(
    merged["TRADEDATE"],
    merged["equity_strategy"],
    color="steelblue",
    linewidth=1.8,
    label="High-Yield Golden Cross" + (" + RGBITR фильтр" if USE_FILTER else ""),
)
ax.plot(
    merged["TRADEDATE"],
    merged["equity_bh"],
    color="darkorange",
    linewidth=1.8,
    label="MCFTR Buy & Hold",
)
ax.set_title(
    "Equity Curve: стратегия vs Buy & Hold (начало = 1.0)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Дата")
ax.set_ylabel("Стоимость портфеля")
ax.legend(loc="best")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 2: ПРОСАДКИ ----------
fig, ax = plt.subplots()
ax.fill_between(
    merged["TRADEDATE"],
    dd_strategy * 100,
    0,
    color="steelblue",
    alpha=0.4,
    label="Стратегия",
)
ax.fill_between(
    merged["TRADEDATE"],
    dd_bh * 100,
    0,
    color="darkorange",
    alpha=0.4,
    label="Buy & Hold",
)
ax.set_title("Просадки (Drawdown), %", fontsize=14, fontweight="bold")
ax.set_xlabel("Дата")
ax.set_ylabel("Просадка, %")
ax.legend(loc="lower left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "drawdowns.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 3: СИГНАЛЫ + MCFTR + КЛЮЧЕВАЯ СТАВКА ----------
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    figsize=(15, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1.5, 1], "hspace": 0.08},
)

# --- Верхняя панель: RUCBHYTR + SMA + сигналы ---
ax1.plot(
    merged["TRADEDATE"],
    merged[SIGNAL_TICKER],
    color="gray",
    linewidth=1.2,
    label=SIGNAL_TICKER,
    zorder=2,
)
ax1.plot(
    merged["TRADEDATE"],
    merged["SMA_short"],
    color="orange",
    linewidth=1.5,
    label=f"SMA {SHORT_WINDOW}",
    zorder=3,
)
ax1.plot(
    merged["TRADEDATE"],
    merged["SMA_long"],
    color="red",
    linewidth=1.5,
    label=f"SMA {LONG_WINDOW}",
    zorder=3,
)

diff = merged["SMA_short"] - merged["SMA_long"]
cross_up = (diff > 0) & (diff.shift(1) <= 0)
cross_down = (diff < 0) & (diff.shift(1) >= 0)

ax1.scatter(
    merged.loc[cross_up, "TRADEDATE"],
    merged.loc[cross_up, "SMA_short"],
    color="green",
    s=90,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
    label="Golden Cross (покупка)",
)
ax1.scatter(
    merged.loc[cross_down, "TRADEDATE"],
    merged.loc[cross_down, "SMA_short"],
    color="red",
    s=90,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
    label="Death Cross (продажа)",
)

ax1.set_title(
    f"Сигналы {SIGNAL_TICKER} (SMA {SHORT_WINDOW}/{LONG_WINDOW})",
    fontsize=13,
    fontweight="bold",
)
ax1.set_ylabel(f"Значение {SIGNAL_TICKER}")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

# --- Средняя панель: MCFTR + закраска периодов в рынке ---
ax2.plot(
    merged["TRADEDATE"],
    merged[TRADE_TICKER],
    color="steelblue",
    linewidth=1.5,
    label=TRADE_TICKER,
    zorder=3,
)

in_market = merged["position"] == 1
dates = merged["TRADEDATE"].values
start_idx = None
for i in range(len(in_market)):
    if in_market[i] and start_idx is None:
        start_idx = i
    elif not in_market[i] and start_idx is not None:
        ax2.axvspan(dates[start_idx], dates[i - 1], color="green", alpha=0.10, zorder=1)
        start_idx = None
if start_idx is not None:
    ax2.axvspan(dates[start_idx], dates[-1], color="green", alpha=0.10, zorder=1)

ax2.scatter(
    merged.loc[cross_up, "TRADEDATE"],
    merged.loc[cross_up, TRADE_TICKER],
    color="green",
    s=70,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)
ax2.scatter(
    merged.loc[cross_down, "TRADEDATE"],
    merged.loc[cross_down, TRADE_TICKER],
    color="red",
    s=70,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)

ax2.set_title(
    f"{TRADE_TICKER} (зелёная заливка = стратегия держит акции)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_ylabel(f"Значение {TRADE_TICKER}")
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

# --- Нижняя панель: ключевая ставка ЦБ ---
ax3.plot(
    merged["TRADEDATE"],
    merged["RATE"],
    color="darkred",
    linewidth=1.5,
    label="Ключевая ставка ЦБ РФ, %",
)
ax3.fill_between(merged["TRADEDATE"], merged["RATE"], 0, color="darkred", alpha=0.15)
ax3.set_title(
    "Ключевая ставка ЦБ РФ (реальная доходность в кэше)", fontsize=12, fontweight="bold"
)
ax3.set_xlabel("Дата")
ax3.set_ylabel("Ставка, %")
ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)

ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

fig.suptitle(
    f"RUCBHYTR (сигналы) + MCFTR (торговля) + Ключевая ставка (кэш)  |  "
    f"Период: {merged['TRADEDATE'].min().date()} — {merged['TRADEDATE'].max().date()}  |  "
    f"Сделок: {n_trades}",
    fontsize=11,
    y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUTPUT_DIR, "signals_full.png"), dpi=150)
plt.close(fig)


# ---------- СОХРАНЕНИЕ ДАННЫХ ----------
merged.to_csv(
    os.path.join(OUTPUT_DIR, "backtest_data.csv"), index=False, encoding="utf-8"
)

print(f"\nГотово! Результаты в папке: {OUTPUT_DIR}")
print(
    "Файлы: equity_curve.png, drawdowns.png, signals_full.png, metrics.csv, backtest_data.csv"
)
