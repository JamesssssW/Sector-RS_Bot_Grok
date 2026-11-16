import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ====================== 板塊定義 ======================
CORE_SECTORS = {
    'XLK': '科技',
    'XLF': '金融',
    'XLV': '醫療保健',
    'XLY': '非必需消費',
    'XLP': '必需消費',
    'XLE': '能源',
    'XLI': '工業',
    'XLU': '公用事業',
    'XLB': '原材料',
    'XLRE': '房地產',
    'XLC': '通訊服務',
    'SMH': '半導體',
    'XBI': '生物科技',
    'IGV': '軟體應用',
    'KRE': '地區銀行',
    'XHB': '房屋建築',
    'XRT': '零售',
    'IYT': '運輸',
    'ITA': '國防航天',
    'XME': '金屬礦業'
}

THEMATIC_SECTORS = {
    'CIBR': '網絡安全',
    'DRAM': '記憶體',
    'UFO': '太空探索',
    'LYTE': '光通信',
    'WGMI': '加密挖礦',
    'ICLN': '清潔能源',
    'TAN': '太陽能',
    'BOTZ': '機械人AI',
    'SOCL': '社交媒體',
    'FINX': '金融科技',
    'COPX': '銅礦金屬',
    'IHI': '醫療設備',
    'SKYY': '雲端運算',
    'PAVE': '基礎設施',
    'JETS': '航空旅遊'
}

SECTORS = {**CORE_SECTORS, **THEMATIC_SECTORS}
BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
MIN_BARS = 60          # 至少要有多少根 K 線才納入計算
MAX_RANK_GAP_DAYS = 4  # 歷史日期差距超過此天數就不算排名變化

# ====================== 歷史讀取 / 儲存（含日期校驗） ======================
def load_previous_history():
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return {}

        first_val = next(iter(data.values()))

        # 舊格式：純 int
        if isinstance(first_val, int):
            new_data = {}
            for k, v in data.items():
                new_data[k] = {
                    "ranks": [v],
                    "quadrants": [None],
                    "rank": v,
                    "quadrant": None,
                    "last_date": None
                }
            return new_data

        return data

    except Exception as e:
        print(f"讀取歷史紀錄失敗：{e}")
        return {}

def save_current_history(df_result, trade_date: str):
    prev_data = load_previous_history()
    history_data = {}

    for idx, row in df_result.iterrows():
        ticker = row['ticker']
        curr_rank = int(idx + 1)
        curr_quad = row['quadrant']

        item_hist = prev_data.get(ticker, {})
        ranks = item_hist.get('ranks', [])
        quadrants = item_hist.get('quadrants', [])

        ranks.append(curr_rank)
        quadrants.append(curr_quad)
        ranks = ranks[-30:]
        quadrants = quadrants[-30:]

        history_data[ticker] = {
            "ranks": ranks,
            "quadrants": quadrants,
            "rank": curr_rank,
            "quadrant": curr_quad,
            "last_date": trade_date
        }

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

def is_history_fresh(item_hist: dict, today: datetime) -> bool:
    """判斷歷史是否足夠新，可拿來算排名變化"""
    last_date_str = item_hist.get('last_date')
    if not last_date_str:
        return False
