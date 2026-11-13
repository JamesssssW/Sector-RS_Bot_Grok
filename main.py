import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# ====================== 設定 ======================
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

BENCHMARK = "SPY"
TICKERS = list(SECTORS.keys()) + [BENCHMARK]

# Discord Webhook（從環境變數讀取）
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

# ====================== 資料抓取 ======================
def get_data(period="1y"):
    data = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"]
    return data.dropna(how="all")

# ====================== 計算相對強度 ======================
def calc_returns(df: pd.DataFrame):
    """計算各週期報酬率"""
    periods = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "YTD": None,  # 特殊處理
    }

    results = []
    last = df.iloc[-1]

    for ticker in SECTORS.keys():
        row = {"Ticker": ticker, "Sector": SECTORS[ticker]}

        for name, days in periods.items():
            if name == "YTD":
                # 取今年第一個交易日
                ytd_start = df[df.index.year == df.index[-1].year].iloc[0]
                ret = (last[ticker] / ytd_start[ticker] - 1) * 100
                spy_ret = (last[BENCHMARK] / ytd_start[BENCHMARK] - 1) * 100
            else:
                if len(df) < days + 1:
                    ret = spy_ret = float("nan")
                else:
                    ret = (last[ticker] / df[ticker].iloc[-days-1] - 1) * 100
                    spy_ret = (last[BENCHMARK] / df[BENCHMARK].iloc[-days-1] - 1) * 100

            row[f"{name}"] = ret
            row[f"RS_{name}"] = ret - spy_ret   # 相對強度 = 板塊報酬 - SPY報酬

        results.append(row)

    return pd.DataFrame(results)

# ====================== 排名與格式化 ======================
def rank_and_format(df: pd.DataFrame):
    # 以 1M 相對強度為主排序
    df = df.sort_values("RS_1M", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    # 產生 Discord 訊息
    et = pytz.timezone("US/Eastern")
    now = datetime.now(et).strftime("%Y-%m-%d %H:%M ET")

    lines = [
        f"**美股板塊相對強度排名 (vs SPY)**",
        f"資料日期：{now}",
        "",
        "```",
        f"{'排名':<4} {'板塊':<22} {'1W RS':>8} {'1M RS':>8} {'3M RS':>8}",
        "-" * 52,
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['Rank']:<4} {row['Sector']:<22} "
            f"{row['RS_1W']:>+7.1f}% {row['RS_1M']:>+7.1f}% {row['RS_3M']:>+7.1f}%"
        )

    lines.append("```")
    lines.append("")
    lines.append("**說明**：RS = 板塊報酬 − SPY 報酬（正數表示跑贏大盤）")
    lines.append("排序依據：1個月相對強度 (RS_1M)")

    return "\n".join(lines), df

# ====================== 發送 Discord ======================
def send_discord(content: str):
    if not WEBHOOK_URL:
        print("未設定 DISCORD_WEBHOOK，只印出結果：")
        print(content)
        return

    payload = {
        "content": content,
        "username": "Sector RS Bot"
    }
    resp = requests.post(WEBHOOK_URL, json=payload)
    if resp.status_code in (200, 204):
        print("成功發送到 Discord")
    else:
        print(f"發送失敗：{resp.status_code} - {resp.text}")

# ====================== 主程式 ======================
def main():
    print("開始抓取資料...")
    df = get_data(period="1y")
    print("計算相對強度...")
    rs_df = calc_returns(df)
    msg, ranked = rank_and_format(rs_df)

    print(msg)
    send_discord(msg)

if __name__ == "__main__":
    main()