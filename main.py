import os
import yfinance as yf
import pandas as pd
import numpy as np
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

# 加權綜合 RS 權重（可自行調整）
WEIGHTS = {
    "1W": 0.20,
    "1M": 0.40,
    "3M": 0.40,
}

# RRG 計算參數
RRG_WINDOW = 14          # 用於標準化的滾動窗口
RS_MOM_LOOKBACK = 5      # RS-Momentum 回看天數

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

# ====================== 資料抓取 ======================
def get_data(period="1y"):
    data = yf.download(
        TICKERS,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True
    )["Close"]
    return data.dropna(how="all")

# ====================== 計算報酬與加權 RS ======================
def calc_returns(df: pd.DataFrame):
    periods = {
        "1W": 5,
        "1M": 21,
        "3M": 63,
    }

    results = []
    last = df.iloc[-1]

    for ticker in SECTORS.keys():
        row = {
            "Ticker": ticker,
            "Sector": SECTORS[ticker]
        }

        rs_values = {}

        for name, days in periods.items():
            if len(df) < days + 1:
                ret = spy_ret = np.nan
            else:
                ret = (last[ticker] / df[ticker].iloc[-days - 1] - 1) * 100
                spy_ret = (last[BENCHMARK] / df[BENCHMARK].iloc[-days - 1] - 1) * 100

            row[f"{name}"] = ret
            row[f"RS_{name}"] = ret - spy_ret
            rs_values[name] = row[f"RS_{name}"]

        # 加權綜合 RS
        composite = 0.0
        valid_weight = 0.0
        for name, w in WEIGHTS.items():
            if not np.isnan(rs_values[name]):
                composite += rs_values[name] * w
                valid_weight += w

        row["RS_Composite"] = composite / valid_weight if valid_weight > 0 else np.nan
        results.append(row)

    return pd.DataFrame(results)

# ====================== RRG 四象限計算 ======================
def calc_rrg(df: pd.DataFrame):
    """
    計算簡易版 RRG：
    - RS-Ratio  : 相對強度標準化
    - RS-Momentum : RS-Ratio 的變化率
    """
    rs_lines = {}
    for ticker in SECTORS.keys():
        rs = (df[ticker] / df[BENCHMARK]) * 100
        rs_lines[ticker] = rs

    rs_df = pd.DataFrame(rs_lines)

    # 標準化 RS-Ratio（類似 JDK 的做法）
    rs_ratio = {}
    rs_mom = {}

    for ticker in SECTORS.keys():
        series = rs_df[ticker].dropna()
        if len(series) < RRG_WINDOW + RS_MOM_LOOKBACK:
            rs_ratio[ticker] = np.nan
            rs_mom[ticker] = np.nan
            continue

        # 滾動標準化
        rolling_mean = series.rolling(RRG_WINDOW).mean()
        rolling_std = series.rolling(RRG_WINDOW).std(ddof=0)
        ratio = 100 + (series - rolling_mean) / rolling_std
        ratio = ratio.dropna()

        # RS-Momentum = 最近變化
        mom = (ratio / ratio.shift(RS_MOM_LOOKBACK) - 1) * 100
        mom = mom.dropna()

        rs_ratio[ticker] = ratio.iloc[-1]
        rs_mom[ticker] = mom.iloc[-1]

    return rs_ratio, rs_mom

def get_quadrant(rs_ratio: float, rs_mom: float) -> str:
    if np.isnan(rs_ratio) or np.isnan(rs_mom):
        return "N/A"
    if rs_ratio >= 100 and rs_mom >= 0:
        return "Leading"
    elif rs_ratio >= 100 and rs_mom < 0:
        return "Weakening"
    elif rs_ratio < 100 and rs_mom >= 0:
        return "Improving"
    else:
        return "Lagging"

# ====================== 顏色對應 ======================
QUADRANT_COLOR = {
    "Leading": 0x2ECC71,      # 綠
    "Improving": 0x3498DB,    # 藍
    "Weakening": 0xF1C40F,    # 黃
    "Lagging": 0xE74C3C,      # 紅
    "N/A": 0x95A5A6,
}

QUADRANT_EMOJI = {
    "Leading": "🟢",
    "Improving": "🔵",
    "Weakening": "🟡",
    "Lagging": "🔴",
    "N/A": "⚪",
}

# ====================== 產生 Discord Embed ======================
def build_embed(ranked_df: pd.DataFrame, rs_ratio: dict, rs_mom: dict):
    et = pytz.timezone("US/Eastern")
    now_str = datetime.now(et).strftime("%Y-%m-%d %H:%M ET")

    # 取最強板塊決定主色
    top_quad = ranked_df.iloc[0]["Quadrant"]
    color = QUADRANT_COLOR.get(top_quad, 0x5865F2)

    # 標題與描述
    title = "📊 美股板塊相對強度排名 (vs SPY)"
    description = (
        f"**資料時間**：{now_str}\n"
        f"**排序依據**：加權綜合 RS（1W 20% + 1M 40% + 3M 40%）\n"
        f"**RRG 象限**：Leading / Improving / Weakening / Lagging"
    )

    # 建立欄位（前 11 名全部列出）
    fields = []

    # 排名總表
    rank_lines = []
    for _, row in ranked_df.iterrows():
        emoji = QUADRANT_EMOJI.get(row["Quadrant"], "⚪")
        rank_lines.append(
            f"`{row['Rank']:2d}` {emoji} **{row['Sector']}** "
            f"({row['Ticker']})  "
            f"RS:{row['RS_Composite']:+.1f}%  "
            f"[{row['Quadrant']}]"

        )

    fields.append({
        "name": "🏆 綜合排名",
        "value": "\n".join(rank_lines),
        "inline": False
    })

    # 各週期 RS 細節
    detail_lines = []
    for _, row in ranked_df.iterrows():
        detail_lines.append(
            f"**{row['Sector']}**  "
            f"1W:{row['RS_1W']:+.1f}%  "
            f"1M:{row['RS_1M']:+.1f}%  "
            f"3M:{row['RS_3M']:+.1f}%"
        )

    fields.append({
        "name": "📈 各週期相對強度 (RS)",
        "value": "\n".join(detail_lines),
        "inline": False
    })

    # RRG 數值
    rrg_lines = []
    for _, row in ranked_df.iterrows():
        t = row["Ticker"]
        ratio = rs_ratio.get(t, np.nan)
        mom = rs_mom.get(t, np.nan)
        rrg_lines.append(
            f"**{row['Sector']}**  "
            f"RS-Ratio:{ratio:.1f}  "
            f"Mom dev_bravo.1f}"
        )

    fields.append({
        "name": "🧭 RRG 指標",
        "value": "\n".join(rrg_lines),
        "inline": False
    })

    # 象限說明
    fields.append({
        "name": "📖 象限說明",
        "value": (
            "🟢 **Leading**　：強勢且持續加速\n"
            "🔵 **Improving**：弱勢但正在轉強\n"
            "🟡 **Weakening**：強勢但動能減弱\n"
            "🔴 **Lagging**　：弱勢且持續走弱"
        ),
        "inline": False
    })

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Sector RS Bot • 資料來源 yfinance • 僅供參考"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    return embed

# ====================== 發送 Discord ======================
def send_discord(embed: dict):
    if not WEBHOOK_URL:
        print("未設定 DISCORD_WEBHOOK，只印出結果")
        print(embed)
        return

    payload = {
        "username": "Sector RS Bot",
        "embeds": [embed]
    }

    resp = requests.post(WEBHOOK_URL, json=payload, timeout=15)
    if resp.status_code in (200, 204):
        print("✅ 成功發送到 Discord")
    else:
        print(f"❌ 發送失敗：{resp.status_code}")
        print(resp.text)

# ====================== 主程式 ======================
def main():
    print("1️⃣ 抓取資料中...")
    df = get_data(period="1y")

    print("2️⃣ 計算加權相對強度...")
    rs_df = calc_returns(df)

    print("3️⃣ 計算 RRG 四象限...")
    rs_ratio, rs_mom = calc_rrg(df)

    # 合併象限資訊
    rs_df["RS_Ratio"] = rs_df["Ticker"].map(rs_ratio)
    rs_df["RS_Mom"] = rs_df["Ticker"].map(rs_mom)
    rs_df["Quadrant"] = rs_df.apply(
        lambda r: get_quadrant(r["RS_Ratio"], r["RS_Mom"]), axis=1
    )

    # 依加權綜合 RS 排序
    ranked = rs_df.sort_values("RS_Composite", ascending=False).reset_index(drop=True)
    ranked["Rank"] = ranked.index + 1

    print("4️⃣ 建立 Discord Embed...")
    embed = build_embed(ranked, rs_ratio, rs_mom)

    print("5️⃣ 發送結果...")
    send_discord(embed)

    # 終端機預覽
    print("\n===== 排名預覽 =====")
    print(ranked[["Rank", "Sector", "RS_Composite", "Quadrant", "RS_1W", "RS_1M", "RS_3M"]].to_string(index=False))

if __name__ == "__main__":
    main()
