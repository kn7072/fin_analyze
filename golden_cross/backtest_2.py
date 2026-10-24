"""
Совмещённый график:
  - Верхняя панель: RUCBHYTR + SMA 50 + SMA 200 + точки Golden/Death Cross
  - Нижняя панель: MCFTR (индекс акций)

Панели имеют общую ось X, поэтому сигналы сверху
визуально совпадают с движениями акций снизу.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_backtest"
SIGNAL_TICKER = "RUCBHYTR"
TRADE_TICKER = "MCFTR"
SHORT_WINDOW = 50
LONG_WINDOW = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.size"] = 11


# ---------- ЗАГРУЗКА ----------
def load_index(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_history.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    return df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": ticker})


signal_df = load_index(SIGNAL_TICKER)
trade_df = load_index(TRADE_TICKER)

merged = (
    pd.merge(signal_df, trade_df, on="TRADEDATE", how="inner")
    .sort_values("TRADEDATE")
    .reset_index(drop=True)
)

# ---------- СИГНАЛЫ ----------
merged["SMA_short"] = merged[SIGNAL_TICKER].rolling(SHORT_WINDOW).mean()
merged["SMA_long"] = merged[SIGNAL_TICKER].rolling(LONG_WINDOW).mean()

diff = merged["SMA_short"] - merged["SMA_long"]
merged["cross_up"] = (diff > 0) & (diff.shift(1) <= 0)
merged["cross_down"] = (diff < 0) & (diff.shift(1) >= 0)

# Позиция: 1 — держим акции, 0 — в кэше. Сдвиг на 1 день (без look-ahead).
merged["position"] = np.where(diff > 0, 1, 0)
merged["position"] = merged["position"].shift(1).fillna(0).astype(int)

# Закрашиваем периоды, когда стратегия держала акции
merged["in_market"] = merged["position"] == 1


# ---------- ГРАФИК ----------
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(15, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1.4], "hspace": 0.08},
)

# ===== ВЕРХНЯЯ ПАНЕЛЬ: RUCBHYTR + SMA + сигналы =====
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

# Точки пересечений
ax1.scatter(
    merged.loc[merged["cross_up"], "TRADEDATE"],
    merged.loc[merged["cross_up"], "SMA_short"],
    color="green",
    s=90,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
    label="Golden Cross (покупка)",
)
ax1.scatter(
    merged.loc[merged["cross_down"], "TRADEDATE"],
    merged.loc[merged["cross_down"], "SMA_short"],
    color="red",
    s=90,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
    label="Death Cross (продажа)",
)

ax1.set_title(
    f"Сигналы High-Yield Golden Cross: {SIGNAL_TICKER} (SMA {SHORT_WINDOW}/{LONG_WINDOW})",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylabel(f"Значение {SIGNAL_TICKER}")
ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)

# ===== НИЖНЯЯ ПАНЕЛЬ: MCFTR + закраска периодов в рынке =====
ax2.plot(
    merged["TRADEDATE"],
    merged[TRADE_TICKER],
    color="steelblue",
    linewidth=1.5,
    label=TRADE_TICKER,
    zorder=3,
)

# Закрашиваем периоды, когда стратегия держит акции
# Используем fill_between с ymin=min*0.98, ymax=max*1.02
y_min = merged[TRADE_TICKER].min() * 0.98
y_max = merged[TRADE_TICKER].max() * 1.02
in_market = merged["in_market"].values
dates = merged["TRADEDATE"].values

# Находим непрерывные интервалы in_market=True
start_idx = None
for i in range(len(in_market)):
    if in_market[i] and start_idx is None:
        start_idx = i
    elif not in_market[i] and start_idx is not None:
        ax2.axvspan(dates[start_idx], dates[i - 1], color="green", alpha=0.10, zorder=1)
        start_idx = None
if start_idx is not None:
    ax2.axvspan(dates[start_idx], dates[-1], color="green", alpha=0.10, zorder=1)

# Отмечаем на MCFTR точки входа/выхода
ax2.scatter(
    merged.loc[merged["cross_up"], "TRADEDATE"],
    merged.loc[merged["cross_up"], TRADE_TICKER],
    color="green",
    s=90,
    marker="^",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)
ax2.scatter(
    merged.loc[merged["cross_down"], "TRADEDATE"],
    merged.loc[merged["cross_down"], TRADE_TICKER],
    color="red",
    s=90,
    marker="v",
    zorder=5,
    edgecolor="black",
    linewidth=0.6,
)

ax2.set_title(
    f"{TRADE_TICKER} (зелёная заливка = стратегия держит акции)",
    fontsize=13,
    fontweight="bold",
)
ax2.set_xlabel("Дата")
ax2.set_ylabel(f"Значение {TRADE_TICKER}")
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)

# Оформление оси X
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

# Подписи с числами сделок
n_buy = int(merged["cross_up"].sum())
n_sell = int(merged["cross_down"].sum())
fig.suptitle(
    f"Сигналов на покупку: {n_buy}  |  Сигналов на продажу: {n_sell}  |  "
    f"Период: {merged['TRADEDATE'].min().date()} — {merged['TRADEDATE'].max().date()}",
    fontsize=11,
    y=0.995,
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
file_path = os.path.join(OUTPUT_DIR, "signals_overlay.png")
fig.savefig(file_path, dpi=150)
plt.close(fig)
print(f"Сохранён график: {file_path}")
