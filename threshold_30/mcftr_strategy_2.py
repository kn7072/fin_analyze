"""https://smart-lab.ru/blog/1149183.php?nomobile=1#1"""

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
START_DATE = "2003-01-01"  # чтобы захватить первые 5-летние окна
TRADING_DAYS = 252
LOOKBACK_DAYS = TRADING_DAYS * 5  # ~5 лет торгов
BUY_THRESHOLD = 0.30  # покупаем, когда 5Y доходность <= 30%
SELL_THRESHOLD = 1.80  # продаём, когда 5Y доходность >= 180%
CASH_RATE = 0.16  # безрисковая ставка в кэше (ОФЗ)

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
def download_moex_history(ticker, start_date=None):
    base_url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/index/"
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
        print(f"Ошибка при первичном запросе: {e}")
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
                print(f"Ошибка на странице {start}: {e}")
                break

    if not all_data:
        return pd.DataFrame()

    columns = j["history"]["columns"]
    df = pd.DataFrame(all_data, columns=columns)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    return df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": ticker})


print("Загрузка данных MCFTR...")
raw_df = download_moex_history(TICKER, START_DATE)
if raw_df.empty:
    print("Не удалось загрузить данные.")
    exit()

raw_path = os.path.join(DATA_DIR, f"{TICKER}_history.csv")
raw_df.to_csv(raw_path, index=False, encoding="utf-8")
print(f"Данные сохранены: {raw_path}")

df = raw_df.set_index("TRADEDATE").copy()

# ---------- 5-ЛЕТНЯЯ ДОХОДНОСТЬ (дневная) ----------
df["5Y_Return"] = df[TICKER] / df[TICKER].shift(LOOKBACK_DAYS) - 1

# ---------- СИГНАЛЫ (машина состояний) ----------
regime = 0  # 0 = кэш, 1 = в рынке
regimes = []
for val in df["5Y_Return"].values:
    if pd.isna(val):
        regimes.append(np.nan)
        continue
    if regime == 0 and val <= BUY_THRESHOLD:
        regime = 1
    elif regime == 1 and val >= SELL_THRESHOLD:
        regime = 0
    regimes.append(regime)
df["regime"] = regimes
df["position"] = df["regime"].shift(1).fillna(0)  # без look-ahead

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

# Точки сделок
buy_signals = df[df["position"].diff() == 1]
sell_signals = df[df["position"].diff() == -1]


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
        metrics(df["equity_strategy"], df["ret_strategy"], "5Y Return Strategy"),
        metrics(df["equity_bh"], df["ret_mcftr"], "Buy & Hold MCFTR"),
    ]
)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False, encoding="utf-8")
print(f"\nСделок: {len(buy_signals) + len(sell_signals)}")
print(f"Время в рынке: {df['position'].mean() * 100:.1f}%")


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
    "1. MCFTR: цена и точки входа/выхода по стратегии",
    fontsize=13,
    fontweight="bold",
    loc="left",
)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 2. 5-летняя доходность + пороги ---
ax = axes[1]
ax.plot(
    df.index,
    df["5Y_Return"] * 100,
    color="steelblue",
    linewidth=1.3,
    label="5-летняя доходность MCFTR",
)
ax.axhline(
    BUY_THRESHOLD * 100,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label=f"Порог покупки {BUY_THRESHOLD * 100:.0f}%",
)
ax.axhline(
    SELL_THRESHOLD * 100,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Порог продажи {SELL_THRESHOLD * 100:.0f}%",
)
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.fill_between(
    df.index,
    df["5Y_Return"] * 100,
    BUY_THRESHOLD * 100,
    where=(df["5Y_Return"] <= BUY_THRESHOLD),
    color="green",
    alpha=0.15,
)
ax.fill_between(
    df.index,
    df["5Y_Return"] * 100,
    SELL_THRESHOLD * 100,
    where=(df["5Y_Return"] >= SELL_THRESHOLD),
    color="red",
    alpha=0.15,
)
ax.set_ylabel("Доходность за 5 лет, %")
ax.set_title(
    "2. 5-летняя доходность MCFTR и пороги срабатывания",
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
    label="Стратегия 5Y Return",
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
    f"MCFTR — стратегия на 5-летней доходности   |   "
    f"покупка ≤ {BUY_THRESHOLD * 100:.0f}%, продажа ≥ {SELL_THRESHOLD * 100:.0f}%",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dashboard.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nГотово! Дашборд: {os.path.join(OUTPUT_DIR, 'dashboard.png')}")
