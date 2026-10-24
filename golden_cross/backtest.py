"""
Бэктест стратегии High-Yield Golden Cross на российском рынке.

Сигналы берутся с индекса высокодоходных облигаций (RUCBHYTR),
торговля ведётся по индексу акций полной доходности (MCFTR).

Логика:
  - 50-дневная SMA RUCBHYTR > 200-дневной  -> держим MCFTR
  - 50-дневная SMA RUCBHYTR < 200-дневной  -> выходим в кэш (или ОФЗ)
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_backtest"
SIGNAL_TICKER = "RUCBHYTR"  # источник сигналов
TRADE_TICKER = "MCFTR"  # торгуемый актив
SHORT_WINDOW = 50
LONG_WINDOW = 200
CASH_RATE = 0.16  # годовая ставка в кэше (например, 16% — примерно ставка ОФЗ).
# Поставьте 0.0, если хотите сидеть в кэше без дохода.
TRADING_DAYS = 252

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 7)
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

# Объединяем по датам, где есть оба индекса
merged = (
    pd.merge(signal_df, trade_df, on="TRADEDATE", how="inner")
    .sort_values("TRADEDATE")
    .reset_index(drop=True)
)
print(
    f"Общий период: {merged['TRADEDATE'].min().date()} -> {merged['TRADEDATE'].max().date()}"
)
print(f"Дней в выборке: {len(merged)}")


# ---------- СИГНАЛЫ ----------
merged["SMA_short"] = merged[SIGNAL_TICKER].rolling(SHORT_WINDOW).mean()
merged["SMA_long"] = merged[SIGNAL_TICKER].rolling(LONG_WINDOW).mean()

# Режим: 1 — держим акции, 0 — сидим в кэше
merged["regime"] = np.where(merged["SMA_short"] > merged["SMA_long"], 1, 0)

# Позицию на сегодня берём из вчерашнего сигнала — чтобы избежать look-ahead bias.
# Сигнал на закрытии дня N -> входим/выходим на закрытии дня N+1.
merged["position"] = merged["regime"].shift(1).fillna(0).astype(int)

# Доходность MCFTR (дневная)
merged["ret_mcftr"] = merged[TRADE_TICKER].pct_change().fillna(0)

# Дневная ставка кэша
cash_daily = (1 + CASH_RATE) ** (1 / TRADING_DAYS) - 1

# Доходность стратегии
merged["ret_strategy"] = (
    merged["position"] * merged["ret_mcftr"] + (1 - merged["position"]) * cash_daily
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
    merged["equity_strategy"], merged["ret_strategy"], "High-Yield Golden Cross"
)
metrics_bh, dd_bh = compute_metrics(
    merged["equity_bh"], merged["ret_mcftr"], "MCFTR Buy & Hold"
)

# Дополнительная статистика
n_trades = int((merged["position"].diff().abs() == 1).sum())
time_in_market = merged["position"].mean() * 100

summary = pd.DataFrame([metrics_strategy, metrics_bh])
print("\n===== РЕЗУЛЬТАТЫ БЭКТЕСТА =====")
print(summary.to_string(index=False))
print(f"\nСделок (переходов): {n_trades}")
print(f"Время в рынке: {time_in_market:.1f}%")
print(f"Время в кэше:  {100 - time_in_market:.1f}%")

# Сохраняем таблицу метрик
summary.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False, encoding="utf-8")


# ---------- ГРАФИК 1: EQUITY CURVE ----------
fig, ax = plt.subplots()
ax.plot(
    merged["TRADEDATE"],
    merged["equity_strategy"],
    color="steelblue",
    linewidth=1.8,
    label="High-Yield Golden Cross",
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
    label="High-Yield Golden Cross",
)
ax.fill_between(
    merged["TRADEDATE"],
    dd_bh * 100,
    0,
    color="darkorange",
    alpha=0.4,
    label="MCFTR Buy & Hold",
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


# ---------- ГРАФИК 3: ЦЕНА + СИГНАЛЫ ----------
fig, ax = plt.subplots()
ax.plot(
    merged["TRADEDATE"],
    merged[SIGNAL_TICKER],
    color="gray",
    linewidth=1.2,
    label=SIGNAL_TICKER,
)
ax.plot(
    merged["TRADEDATE"],
    merged["SMA_short"],
    color="orange",
    linewidth=1.3,
    label=f"SMA {SHORT_WINDOW}",
)
ax.plot(
    merged["TRADEDATE"],
    merged["SMA_long"],
    color="red",
    linewidth=1.3,
    label=f"SMA {LONG_WINDOW}",
)

diff = merged["SMA_short"] - merged["SMA_long"]
cross_up = (diff > 0) & (diff.shift(1) <= 0)
cross_down = (diff < 0) & (diff.shift(1) >= 0)

ax.scatter(
    merged.loc[cross_up, "TRADEDATE"],
    merged.loc[cross_up, "SMA_short"],
    color="green",
    s=50,
    zorder=5,
    label="Golden Cross (покупка)",
)
ax.scatter(
    merged.loc[cross_down, "TRADEDATE"],
    merged.loc[cross_down, "SMA_short"],
    color="black",
    s=50,
    zorder=5,
    label="Death Cross (продажа)",
)

ax.set_title(f"Сигналы на {SIGNAL_TICKER}", fontsize=14, fontweight="bold")
ax.set_xlabel("Дата")
ax.set_ylabel("Значение")
ax.legend(loc="best")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "signals.png"), dpi=150)
plt.close(fig)


# ---------- СОХРАНЯЕМ ДАННЫЕ БЭКТЕСТА ----------
merged.to_csv(
    os.path.join(OUTPUT_DIR, "backtest_data.csv"), index=False, encoding="utf-8"
)

print(f"\nГотово! Результаты в папке: {OUTPUT_DIR}")
print(
    "Файлы: equity_curve.png, drawdowns.png, signals.png, metrics.csv, backtest_data.csv"
)
