"""
https://smart-lab.ru/my/SPKOLOTILIN/blog/page4/#1

Визуализация 4 стратегий накопления из статьи.

Стратегии:
  1. «Везунчик»       — покупка на годовом минимуме
  2. «Робот»          — ежемесячные покупки по 8 333 руб. (DCA)
  3. «Ранняя птичка»  — покупка в первый день года
  4. «Неудачник»      — покупка на годовом максимуме

Плюс: облигации (9.1%), депозит (8.7%), инфляция (8.1%).
"""

"""
Визуализация 4 стратегий накопления из статьи.
С улучшенным отображением дат на оси X.
"""

import os
import time

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------- НАСТРОЙКА ШРИФТА ----------
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- НАСТРОЙКИ ----------
DATA_DIR = "moex_history_data"
OUTPUT_DIR = "moex_dca_charts"
TICKER = "MCFTR"
START_DATE = "2006-01-01"
END_DATE = "2026-03-31"

ANNUAL_AMOUNT = 100_000
MONTHLY_AMOUNT = ANNUAL_AMOUNT / 12

RATE_BONDS = 9.1
RATE_DEPOSIT = 8.7
RATE_INFLATION = 8.1

CUR = "руб."

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 11

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)


# ---------- ЗАГРУЗКА ДАННЫХ ----------
def fetch_moex_history(ticker, start_date):
    print(f"  Загрузка {ticker} с {start_date}...")
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


cache_file = os.path.join(DATA_DIR, f"{TICKER}_history.csv")
if os.path.exists(cache_file):
    print(f"Читаем данные из кэша: {cache_file}")
    df = pd.read_csv(cache_file)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    if "CLOSE" in df.columns:
        df = df[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": TICKER})
else:
    df = fetch_moex_history(TICKER, START_DATE)
    if df is not None:
        df.to_csv(cache_file, index=False, encoding="utf-8")

if df is None or df.empty:
    raise SystemExit("Не удалось получить данные MCFTR.")

df = df[df["TRADEDATE"] <= END_DATE].reset_index(drop=True)
print(
    f"\nПериод данных: {df['TRADEDATE'].min().date()} -> {df['TRADEDATE'].max().date()}"
)
print(f"Дней: {len(df)}")
df = df.dropna(subset=[TICKER]).reset_index(drop=True)


# ---------- РАСЧЁТ СТРАТЕГИЙ ----------
df["YEAR"] = df["TRADEDATE"].dt.year

yearly_low_date = df.loc[df.groupby("YEAR")[TICKER].idxmin()][["YEAR", "TRADEDATE"]]
buy_dates_lucky = set(yearly_low_date["TRADEDATE"].dt.normalize())

yearly_high_date = df.loc[df.groupby("YEAR")[TICKER].idxmax()][["YEAR", "TRADEDATE"]]
buy_dates_unlucky = set(yearly_high_date["TRADEDATE"].dt.normalize())

first_day_per_year = df.groupby("YEAR")["TRADEDATE"].min().reset_index()
buy_dates_early = set(first_day_per_year["TRADEDATE"].dt.normalize())

df["YEAR_MONTH"] = df["TRADEDATE"].dt.to_period("M")
first_day_per_month = df.groupby("YEAR_MONTH")["TRADEDATE"].min().reset_index()
buy_dates_robot = set(first_day_per_month["TRADEDATE"].dt.normalize())


def simulate_strategy(df, buy_dates_set, amount_per_buy):
    units, invested = 0.0, 0.0
    equity_curve, buys = [], []
    for _, row in df.iterrows():
        date_norm = row["TRADEDATE"].normalize()
        price = row[TICKER]
        if date_norm in buy_dates_set:
            units += amount_per_buy / price
            invested += amount_per_buy
            buys.append((row["TRADEDATE"], price, amount_per_buy))
        equity_curve.append(
            {
                "TRADEDATE": row["TRADEDATE"],
                "equity": units * price,
                "invested": invested,
                "price": price,
            }
        )
    return pd.DataFrame(equity_curve), buys


equity_lucky, buys_lucky = simulate_strategy(df, buy_dates_lucky, ANNUAL_AMOUNT)
equity_unlucky, buys_unlucky = simulate_strategy(df, buy_dates_unlucky, ANNUAL_AMOUNT)
equity_early, buys_early = simulate_strategy(df, buy_dates_early, ANNUAL_AMOUNT)
equity_robot, buys_robot = simulate_strategy(df, buy_dates_robot, MONTHLY_AMOUNT)


# ---------- МЕТРИКИ ----------
def compute_irr(buys, final_value):
    if not buys or final_value <= 0:
        return np.nan
    first_date = min(b[0] for b in buys)
    last_date = max(b[0] for b in buys)
    cash_flows = [(b[0], -b[2]) for b in buys] + [(last_date, final_value)]

    def npv(r):
        return sum(
            a / (1 + r) ** ((d - first_date).days / 365.25) for d, a in cash_flows
        )

    lo, hi = -0.5, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize(name, equity_df, buys):
    final_value = equity_df["equity"].iloc[-1]
    total_invested = equity_df["invested"].iloc[-1]
    irr = compute_irr(buys, final_value)
    return {
        "Стратегия": name,
        "Взносов": len(buys),
        "Всего вложено": f"{total_invested:,.0f} {CUR}",
        "Итоговый капитал": f"{final_value:,.0f} {CUR}",
        "Прибыль": f"{final_value - total_invested:,.0f} {CUR}",
        "Рост, раз": f"{final_value / total_invested:.2f}x",
        "IRR": f"{irr * 100:.2f}%",
    }


summary = pd.DataFrame(
    [
        summarize("1. Везунчик (на минимумах)", equity_lucky, buys_lucky),
        summarize("2. Робот (ежемесячно, DCA)", equity_robot, buys_robot),
        summarize("3. Ранняя птичка (начало года)", equity_early, buys_early),
        summarize("4. Неудачник (на максимумах)", equity_unlucky, buys_unlucky),
    ]
)
print("\n" + "=" * 90)
print("РЕЗУЛЬТАТЫ СТРАТЕГИЙ")
print("=" * 90)
print(summary.to_string(index=False))
summary.to_csv(
    os.path.join(OUTPUT_DIR, "dca_summary.csv"), index=False, encoding="utf-8"
)


# ========== ФУНКЦИЯ ДЛЯ ОФОРМЛЕНИЯ ОСИ ДАТ ==========
def format_date_axis(ax, show_months=True):
    """
    Настраивает ось X: годы крупно, месяцы мелко (каждые 3 месяца).
    """
    # Основные метки — каждый год
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if show_months:
        # Дополнительные метки — каждые 3 месяца (январь, апрель, июль, октябрь)
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    # Сетка
    ax.grid(True, which="major", axis="x", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)

    # Поворот подписей
    for label in ax.get_xticklabels(which="major"):
        label.set_rotation(0)
        label.set_ha("center")
    if show_months:
        for label in ax.get_xticklabels(which="minor"):
            label.set_rotation(45)
            label.set_ha("right")
            label.set_fontsize(8)


# ---------- ГРАФИК 1: EQUITY CURVES ----------
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    equity_lucky["TRADEDATE"],
    equity_lucky["equity"],
    color="gold",
    linewidth=2.2,
    label="1. Везунчик (на минимумах)",
)
ax.plot(
    equity_robot["TRADEDATE"],
    equity_robot["equity"],
    color="steelblue",
    linewidth=2.2,
    label="2. Робот (ежемесячно)",
)
ax.plot(
    equity_early["TRADEDATE"],
    equity_early["equity"],
    color="green",
    linewidth=1.8,
    label="3. Ранняя птичка (начало года)",
)
ax.plot(
    equity_unlucky["TRADEDATE"],
    equity_unlucky["equity"],
    color="red",
    linewidth=1.8,
    label="4. Неудачник (на максимумах)",
)

first_date = equity_robot["TRADEDATE"].iloc[0]
last_date = equity_robot["TRADEDATE"].iloc[-1]
n_years = (last_date - first_date).days / 365.25

for rate, color, label in [
    (RATE_BONDS, "darkorange", "Облигации (9.1%)"),
    (RATE_DEPOSIT, "brown", "Депозит (8.7%)"),
    (RATE_INFLATION, "gray", "Инфляция (8.1%)"),
]:
    years_elapsed = (last_date - equity_robot["TRADEDATE"]).dt.days / 365.25
    future_values = equity_robot["invested"] * (1 + rate / 100) ** years_elapsed
    ax.plot(
        equity_robot["TRADEDATE"],
        future_values,
        color=color,
        linewidth=1.3,
        linestyle="--",
        alpha=0.7,
        label=label,
    )

ax.set_title(
    "Сравнение стратегий накопления в MCFTR (2006–2026)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Дата (год-месяц)")
ax.set_ylabel(f"Стоимость портфеля, {CUR}")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))

format_date_axis(ax, show_months=True)
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "strategies_equity.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 2: MCFTR СО СТРЕЛКАМИ ПОКУПОК ----------
fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

configs = [
    ("1. Везунчик (покупка на минимумах года)", buys_lucky, "gold", axes[0]),
    ("2. Робот (ежемесячные покупки)", buys_robot, "steelblue", axes[1]),
    ("3. Ранняя птичка (покупка в начале года)", buys_early, "green", axes[2]),
    ("4. Неудачник (покупка на максимумах года)", buys_unlucky, "red", axes[3]),
]

for title, buys, color, ax in configs:
    ax.plot(df["TRADEDATE"], df[TICKER], color="lightgray", linewidth=1.2, zorder=1)
    ax.plot(df["TRADEDATE"], df[TICKER], color="gray", linewidth=1.2, zorder=2)

    if buys:
        buy_dates = [b[0] for b in buys]
        buy_prices = [b[1] for b in buys]

        ax.scatter(
            buy_dates,
            buy_prices,
            color=color,
            s=50,
            zorder=5,
            edgecolor="black",
            linewidth=0.6,
            marker="^",
            label=f"Покупка ({len(buys)} шт.)",
        )

        # Подписи с датами под каждой точкой
        # (для «Робота» их слишком много — показываем только каждый 6-й)
        step = 1 if len(buys) <= 30 else 6
        for i, (d, p, _) in enumerate(buys):
            if i % step == 0:
                ax.annotate(
                    d.strftime("%Y-%m"),
                    xy=(d, p),
                    xytext=(0, -14),
                    textcoords="offset points",
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )

    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(f"{TICKER}")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    format_date_axis(ax, show_months=True)

axes[-1].set_xlabel("Дата (год-месяц)")
fig.suptitle(
    "Точки покупок для каждой стратегии (подписи — даты)",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUTPUT_DIR, "buy_points.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 3: ИТОГОВЫЙ КАПИТАЛ ----------
fig, ax = plt.subplots(figsize=(12, 7))

names = [
    "1. Везунчик",
    "2. Робот",
    "3. Ранняя птичка",
    "4. Неудачник",
    "Облигации",
    "Депозит",
    "Инфляция",
]
final_values = [
    equity_lucky["equity"].iloc[-1],
    equity_robot["equity"].iloc[-1],
    equity_early["equity"].iloc[-1],
    equity_unlucky["equity"].iloc[-1],
    equity_robot["invested"].iloc[-1] * (1 + RATE_BONDS / 100) ** n_years,
    equity_robot["invested"].iloc[-1] * (1 + RATE_DEPOSIT / 100) ** n_years,
    equity_robot["invested"].iloc[-1] * (1 + RATE_INFLATION / 100) ** n_years,
]
colors_bar = ["gold", "steelblue", "green", "red", "darkorange", "brown", "gray"]

bars = ax.barh(names, final_values, color=colors_bar, edgecolor="black", linewidth=0.6)

for bar, val in zip(bars, final_values):
    ax.text(
        val * 1.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val / 1e6:.2f}M {CUR}",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

ax.set_title("Итоговый капитал к марту 2026 года", fontsize=14, fontweight="bold")
ax.set_xlabel(f"Капитал, {CUR}")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "final_capital.png"), dpi=150)
plt.close(fig)


# ---------- ГРАФИК 4: ТАБЛИЦА С ДАТАМИ ВСЕХ СДЕЛОК ----------
# Сохраняем отдельный CSV с датами покупок для каждой стратегии
buys_lucky_df = pd.DataFrame(buys_lucky, columns=["DATE", "PRICE", "AMOUNT"])
buys_robot_df = pd.DataFrame(buys_robot, columns=["DATE", "PRICE", "AMOUNT"])
buys_early_df = pd.DataFrame(buys_early, columns=["DATE", "PRICE", "AMOUNT"])
buys_unlucky_df = pd.DataFrame(buys_unlucky, columns=["DATE", "PRICE", "AMOUNT"])

with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "buys_dates.xlsx")) as writer:
    buys_lucky_df.to_excel(writer, sheet_name="1_Везунчик", index=False)
    buys_robot_df.to_excel(writer, sheet_name="2_Робот", index=False)
    buys_early_df.to_excel(writer, sheet_name="3_Ранняя_птичка", index=False)
    buys_unlucky_df.to_excel(writer, sheet_name="4_Неудачник", index=False)

print(f"\nГотово! Файлы в папке: {OUTPUT_DIR}")
print("  strategies_equity.png — сравнение всех стратегий")
print("  buy_points.png        — MCFTR с точками покупок и датами")
print("  final_capital.png     — итоговый капитал")
print("  dca_summary.csv       — таблица с результатами")
print("  buys_dates.xlsx       — даты всех покупок по стратегиям")
