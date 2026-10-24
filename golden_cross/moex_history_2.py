"""
https://www.moex.com/a2193?ysclid=mus6kpyymy885100309
https://smart-lab.ru/mobile/topic/1222085/#1
https://github.com/ntlekhugov/moex-data-fetcher#1

В русской версии руководства (файл 6523) есть аналогичный раздел:
    iss.only=block1,block2 – ответ может содержать несколько блоков данных и этот параметр позволяет выбрать только нужные;
    <block>.columns=<id>,<id> – состав полей в ответе для конкретного блока, например boards.columns=boardid,board_title

    .

🧩 Как это применяется к истории индексов

Эндпоинт истории индексов (/iss/history/engines/stock/markets/index/securities/{ticker}.json) возвращает блок с именем history. Поэтому:
    iss.only=history — оставляет в ответе только этот блок (отбрасывает history.cursor и другие возможные блоки);
    history.columns=TRADEDATE,SECID,CLOSE — оставляет только указанные поля внутри блока history.

"""

import os
import time

import pandas as pd
import requests

# Список инструментов для воссоздания стратегии Джея Кэппела на МосБирже
# MCFTR - Индекс акций полной доходности (включает дивиденды)
# RGBI - Индекс государственных облигаций (главный макро-индикатор)
# RUCBHYTR - Индекс высокодоходных корпоративных облигаций (доступен с 2019 года)
TICKERS = ["MCFTR", "RGBI", "RUCBHYTR"]
START_DATE = "2006-01-01"

request_session = requests.Session()
request_session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }
)


def fetch_moex_history(ticker, start_date):
    """Скачивает полную историю индекса с MOEX ISS API."""
    print(f"Начинаем выгрузку истории для {ticker}...")

    base_url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/index/"
        f"securities/{ticker}.json"
    )

    all_data = []
    start_index = 0

    while True:
        params = {
            "from": start_date,
            "start": start_index,
            "iss.only": "history",
            "history.columns": "TRADEDATE,SECID,CLOSE",
            # ❌ убрали iss.json=extended — используем стандартный формат
        }

        try:
            response = request_session.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
        except Exception as e:
            print(f"Ошибка при запросе данных для {ticker}: {e}")
            break

        # ✅ Стандартный формат: json_data = {"history": {"columns": [...], "data": [[...], ...]}}
        history = json_data.get("history", {})
        columns = history.get("columns", [])
        rows = history.get("data", [])

        if not rows:
            break

        # Собираем строки в список словарей
        chunk_records = [dict(zip(columns, row)) for row in rows]
        df_chunk = pd.DataFrame(chunk_records)
        all_data.append(df_chunk)

        start_index += len(rows)
        time.sleep(0.1)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.dropna(subset=["CLOSE"])
        final_df = final_df.sort_values(by="TRADEDATE").reset_index(drop=True)
        return final_df
    return None


output_dir = "moex_history_data"
os.makedirs(output_dir, exist_ok=True)

for ticker in TICKERS:
    df = fetch_moex_history(ticker, START_DATE)

    if df is not None and not df.empty:
        file_path = os.path.join(output_dir, f"{ticker}_history.csv")
        df.to_csv(file_path, index=False, encoding="utf-8")
        print(
            f"Успешно сохранено: {file_path} | Строк: {len(df)} | "
            f"Период: {df['TRADEDATE'].min()} -> {df['TRADEDATE'].max()}"
        )
    else:
        print(f"Не удалось получить данные для тикера {ticker}")

print("\nВыгрузка полностью завершена! Все файлы лежат в папке 'moex_history_data'")
