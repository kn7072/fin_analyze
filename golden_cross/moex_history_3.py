"""
Корректный бэктест стратегии High-Yield Golden Cross с визуализацией фильтра RGBITR.

Панели:
  1. RUCBHYTR + SMA 50/200 + сигналы Golden/Death Cross
  2. RGBITR + SMA 200 + отметки блокировки фильтра
  3. MCFTR + закраска периодов в рынке
  4. Ключевая ставка ЦБ РФ
"""

import os
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_backtest_correct"
SIGNAL_TICKER = "RUCBHYTR"
TRADE_TICKER = "MCFTR"
FILTER_TICKER = "RGBITR"
SHORT_WINDOW = 50
LONG_WINDOW = 200
USE_FILTER = True
TRADING_DAYS = 252
START_DATE = "2019-12-30"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (15, 14)
plt.rcParams["font.size"] = 11

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)


# ---------- ЗАГРУЗКА ----------
def fetch_moex_history(ticker, start_date):
    print(f"  Загрузка {ticker}...")
    base_url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/index/"
        f"securities/{ticker}.json"
    )
    all_rows, start_index = [], 0
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
        columns, rows = history.get("columns", []), history.get("data", [])
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
    df = df.sort_values("TRADEDATE").copy()
    rate_df = rate_df.sort_values("DATE").copy()
    merged = pd.merge_asof(
        df,
        rate_df,
        left_on="TRADEDATE",
        right_on="DATE",
        direction="backward",
    )
    merged["RATE"] = merged["RATE"].bfill().ffill()
    return merged.drop(columns=["DATE"])


# ---------- ЗАГРУЗКА ДАННЫХ ----------
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)
signal_df = fetch_moex_history(SIGNAL_TICKER, START_DATE)
trade_df = fetch_moex_history(TRADE_TICKER, START_DATE)
filter_df = fetch_moex_history(FILTER_TICKER, START_DATE)

if signal_df is None or trade_df is None:
    raise SystemExit("Не удалось загрузить базовые данные.")

rate_df = fetch_key_rate_history()

merged = pd.merge(signal_df, trade_df, on="TRADEDATE", how="inner")
if filter_df is not None:
    merged = pd.merge(merged, filter_df, on="TRADEDATE", how="inner")
merged = merged.sort_values("TRADEDATE").reset_index(drop=True)
merged = attach_daily_rate(merged, rate_df)

print(
    f"\nОбщий период: {merged['TRADEDATE'].min().date()} -> {merged['TRADEDATE'].max().date()}"
)
print(f"Дней: {len(merged)}")


# ---------- СИГНАЛЫ И ФИЛЬТР ----------
merged["SMA_short"] = merged[SIGNAL_TICKER].rolling(SHORT_WINDOW).mean()
merged["SMA_long"] = merged[SIGNAL_TICKER].rolling(LONG_WINDOW).mean()

merged["base_signal"] = np.where(merged["SMA_short"] > merged["SMA_long"], 1, 0)

if USE_FILTER and FILTER_TICKER in merged.columns:
    merged["FILTER_SMA200"] = merged[FILTER_TICKER].rolling(LONG_WINDOW).mean()
    merged["filter_ok"] = merged[FILTER_TICKER] > merged["FILTER_SMA200"]
else:
    merged["FILTER_SMA200"] = np.nan
    merged["filter_ok"] = True

merged["regime"] = np.where((merged["base_signal"] == 1) & (merged["filter_ok"]), 1, 0)
merged["position"] = merged["regime"].shift(1).fillna(0).astype(int)

# Метка: сигнал ВДО есть, но фильтр его заблокировал
merged["blocked_by_filter"] = (merged["base_signal"] == 1) & (~merged["filter_ok"])

# Доходности
merged["ret_mcftr"] = merged[TRADE_TICKER].pct_change().fillna(0)
merged["cash_daily"] = merged["RATE"] / 100 / 365
merged["ret_strategy"] = (
    merged["position"] * merged["ret_mcftr"]
    + (1 - merged["position"]) * merged["cash_daily"]
)

merged["equity_strategy"] = (1 + merged["ret_strategy"]).cumprod()
merged["equity_bh"] = (1 + merged["ret_mcftr"]).cumprod()


# ---------- МЕТРИКИ ----------
def compute_metrics(equity, returns, name):
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(equity) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "Стратегия": name,
        "Итоговая доходность": f"{total_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Волатильность (год)": f"{vol * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Calmar": f"{calmar:.2f}",
    }, dd


metrics_strategy, dd_strategy = compute_metrics(
    merged["equity_strategy"],
    merged["ret_strategy"],
    f"High-Yield Golden Cross{' + RGBITR фильтр' if USE_FILTER else ''}",
)
metrics_bh, dd_bh = compute_metrics(
    merged["equity_bh"], merged["ret_mcftr"], "MCFTR Buy & Hold"
)

summary = pd.DataFrame([metrics_strategy, metrics_bh])
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
print(summary.to_string(index=False))

n_trades = int((merged["position"].diff().abs() == 1).sum())
time_in_market = merged["position"].mean() * 100
n_blocked = int(merged["blocked_by_filter"].sum())

print(f"\nСделок: {n_trades}")
print(f"Время в рынке: {time_in_market:.1f}%")
print(f"Дней, когда фильтр блокировал сигнал ВДО: {n_blocked}")
summary.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False, encoding="utf-8")


# ---------- ГЛАВНЫЙ ГРАФИК: 4 ПАНЕЛИ ----------
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4,
    1,
    figsize=(15, 16),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 2, 1.8, 1], "hspace": 0.10},
)

dates = merged["TRADEDATE"].values

# ===== ПАНЕЛЬ 1: RUCBHYTR + SMA + сигналы =====
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
    s=110,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.7,
    label="Golden Cross (сигнал ВДО)",
)
ax1.scatter(
    merged.loc[cross_down, "TRADEDATE"],
    merged.loc[cross_down, "SMA_short"],
    color="red",
    s=110,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.7,
    label="Death Cross",
)

# Отмечаем блокированные дни — рисуем серую зону поверх
blocked = merged["blocked_by_filter"].values
start_idx = None
for i in range(len(blocked)):
    if blocked[i] and start_idx is None:
        start_idx = i
    elif not blocked[i] and start_idx is not None:
        ax1.axvspan(dates[start_idx], dates[i - 1], color="black", alpha=0.12, zorder=1)
        start_idx = None
if start_idx is not None:
    ax1.axvspan(dates[start_idx], dates[-1], color="black", alpha=0.12, zorder=1)

ax1.set_title(
    f"1. Источник сигналов: {SIGNAL_TICKER} (SMA {SHORT_WINDOW}/{LONG_WINDOW})  |  "
    f"Серые зоны — фильтр RGBITR блокирует покупку",
    fontsize=12,
    fontweight="bold",
)
ax1.set_ylabel(f"Значение {SIGNAL_TICKER}")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

# ===== ПАНЕЛЬ 2: RGBITR + SMA200 + отметки =====
if FILTER_TICKER in merged.columns:
    ax2.plot(
        merged["TRADEDATE"],
        merged[FILTER_TICKER],
        color="purple",
        linewidth=1.5,
        label=FILTER_TICKER,
        zorder=3,
    )
    ax2.plot(
        merged["TRADEDATE"],
        merged["FILTER_SMA200"],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"SMA {LONG_WINDOW}",
        zorder=3,
    )

    # Закраска периодов, когда RGBITR ниже своей SMA200 (фильтр выключен)
    below = (merged[FILTER_TICKER] < merged["FILTER_SMA200"]).values
    start_idx = None
    for i in range(len(below)):
        if below[i] and start_idx is None:
            start_idx = i
        elif not below[i] and start_idx is not None:
            ax2.axvspan(
                dates[start_idx], dates[i - 1], color="red", alpha=0.10, zorder=1
            )
            start_idx = None
    if start_idx is not None:
        ax2.axvspan(dates[start_idx], dates[-1], color="red", alpha=0.10, zorder=1)

    # Отмечаем точки, где фильтр блокировал (только начало каждой серии)
    blocked_dates = merged.loc[merged["blocked_by_filter"], "TRADEDATE"]
    blocked_values = merged.loc[merged["blocked_by_filter"], FILTER_TICKER]
    if len(blocked_dates) > 0:
        ax2.scatter(
            blocked_dates,
            blocked_values,
            color="black",
            s=20,
            marker="x",
            zorder=5,
            label="Фильтр блокирует",
            alpha=0.7,
        )

    ax2.set_title(
        f"2. Фильтр: {FILTER_TICKER} (SMA {LONG_WINDOW})  |  "
        f"Красная заливка = RGBITR ниже SMA{LONG_WINDOW} → фильтр выключен",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel(f"Значение {FILTER_TICKER}")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
else:
    ax2.text(
        0.5,
        0.5,
        "RGBITR не загружен",
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=14,
    )
    ax2.set_title("2. Фильтр RGBITR", fontsize=12, fontweight="bold")

# ===== ПАНЕЛЬ 3: MCFTR + закраска периодов в рынке =====
ax3.plot(
    merged["TRADEDATE"],
    merged[TRADE_TICKER],
    color="steelblue",
    linewidth=1.5,
    label=TRADE_TICKER,
    zorder=3,
)

in_market = merged["position"] == 1
start_idx = None
for i in range(len(in_market)):
    if in_market[i] and start_idx is None:
        start_idx = i
    elif not in_market[i] and start_idx is not None:
        ax3.axvspan(dates[start_idx], dates[i - 1], color="green", alpha=0.12, zorder=1)
        start_idx = None
if start_idx is not None:
    ax3.axvspan(dates[start_idx], dates[-1], color="green", alpha=0.12, zorder=1)

ax3.scatter(
    merged.loc[cross_up, "TRADEDATE"],
    merged.loc[cross_up, TRADE_TICKER],
    color="green",
    s=90,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)
ax3.scatter(
    merged.loc[cross_down, "TRADEDATE"],
    merged.loc[cross_down, TRADE_TICKER],
    color="red",
    s=90,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)

ax3.set_title(
    f"3. Торговля: {TRADE_TICKER}  |  Зелёные зоны = стратегия держит акции",
    fontsize=12,
    fontweight="bold",
)
ax3.set_ylabel(f"Значение {TRADE_TICKER}")
ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)

# ===== ПАНЕЛЬ 4: Ключевая ставка =====
ax4.plot(
    merged["TRADEDATE"],
    merged["RATE"],
    color="darkred",
    linewidth=1.5,
    label="Ключевая ставка ЦБ РФ, %",
)
ax4.fill_between(merged["TRADEDATE"], merged["RATE"], 0, color="darkred", alpha=0.15)
ax4.set_title(
    "4. Доходность в кэше: ключевая ставка ЦБ РФ", fontsize=12, fontweight="bold"
)
ax4.set_xlabel("Дата")
ax4.set_ylabel("Ставка, %")
ax4.legend(loc="upper left", fontsize=9, framealpha=0.9)

ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

fig.suptitle(
    f"Полная логика стратегии: сигналы RUCBHYTR + фильтр RGBITR + торговля MCFTR + ставка ЦБ  |  "
    f"Сделок: {n_trades}  |  Заблокировано дней: {n_blocked}",
    fontsize=11,
    y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUTPUT_DIR, "full_logic.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК: EQUITY CURVE + DRAWDOWN ----------
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(15, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
)

ax1.plot(
    merged["TRADEDATE"],
    merged["equity_strategy"],
    color="steelblue",
    linewidth=1.8,
    label="Стратегия",
)
ax1.plot(
    merged["TRADEDATE"],
    merged["equity_bh"],
    color="darkorange",
    linewidth=1.8,
    label="MCFTR Buy & Hold",
)
ax1.set_title("Equity Curve (начало = 1.0)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Стоимость портфеля")
ax1.legend(loc="best")

ax2.fill_between(
    merged["TRADEDATE"],
    dd_strategy * 100,
    0,
    color="steelblue",
    alpha=0.4,
    label="Стратегия",
)
ax2.fill_between(
    merged["TRADEDATE"],
    dd_bh * 100,
    0,
    color="darkorange",
    alpha=0.4,
    label="Buy & Hold",
)
ax2.set_title("Просадки (Drawdown), %", fontsize=13, fontweight="bold")
ax2.set_xlabel("Дата")
ax2.set_ylabel("Просадка, %")
ax2.legend(loc="lower left")
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "equity_drawdown.png"), dpi=150)
plt.close(fig)


# ---------- СОХРАНЕНИЕ ----------
merged.to_csv(
    os.path.join(OUTPUT_DIR, "backtest_data.csv"), index=False, encoding="utf-8"
)

print(f"\nГотово! Файлы в: {OUTPUT_DIR}")
print("  full_logic.png — все 4 панели")
print("  equity_drawdown.png — кривая доходности и просадки")
print("  metrics.csv, backtest_data.csv")
