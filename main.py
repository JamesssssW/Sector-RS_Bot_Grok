import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz
from pathlib import Path

# ====================== 設定 ======================
# 20 個板塊 / 主題 ETF
SECTORS = {
    # 11 大 GICS 板塊
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Comm. Services",
    # 額外 9 個熱門主題 / 產業
    "SMH": "Semiconductors",
    "XBI": "Biotech",
    "XRT": "Retail",
    "XHB": "Homebuilders",
    "KRE": "Regional Banks",
    "ITA": "Aerospace & Defense",
    "GDX": "Gold Miners",
    "TAN": "Solar",
    "BOTZ": "Robotics & AI",
}

BENCHMARK = "SPY"
TICKERS = list(SECTORS.keys()) + [BENCHMARK]

# 加權綜合 RS 權重
WEIGHTS = {
    "1W": 0.20,
    "1M": 0.40,
    "3M": 0.40,
}

# RRG 參數
RRG_WINDOW = 14
RS_MOM_LOOKBACK = 5

# 檔案路徑
PREV_RANK_FILE = Path("previous_ranking.json")

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
            if not np.isnan(rs_values.get(name, np.nan)):
                composite += rs_values[name] * w
                valid_weight += w

        row["RS_Composite"] = composite / valid_weight if valid_weight > 0 else np.nan
        results.append(row)

    return pd.DataFrame(results)

# ====================== RRG 四象限 ======================
def calc_rrg(df: pd.DataFrame):
    rs_lines = {}
    for ticker in SECTORS.keys():
        rs = (df[ticker] / df[BENCHMARK]) * 100
        rs_lines[ticker] = rs

    rs_df = pd.DataFrame(rs_lines)

    rs_ratio = {}
    rs_mom = {}

    for ticker in SECTORS.keys():
        series = rs_df[ticker].dropna()
        if len(series) < RRG_WINDOW + RS_MOM_LOOKBACK:
            rs_ratio[ticker] = np.nan
            rs_mom[ticker] = np.nan
            continue

        rolling_mean = series.rolling(RRG_WINDOW).mean()
        rolling_std = series.rolling(RRG_WINDOW).std(ddof=0)
        ratio = 100 + (series - rolling_mean) / rolling_std
        ratio = ratio.dropna()

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

# ====================== 前一日排名處理 ======================
def load_previous_ranking():
    if not PREV_RANK_FILE.exists():
        return {}
    try:
        with open(PREV_RANK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ranking", {})
    except Exception:
        return {}

def save_current_ranking(ranked_df: pd.DataFrame):
    ranking = {
        row["Ticker"]: int(row["Rank"])
        for _, row in ranked_df.iterrows()
    }
    data = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "ranking": ranking
    }
    with open(PREV_RANK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def calc_rank_change(current_rank: int, prev_rank: int | None) -> str:
    if prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank  # 正數代表排名上升
    if diff > 0:
        return f"↑{diff}"
    elif diff < 0:
        return f"↓{abs(diff)}"
    else:
        return "→"

# ====================== 顏色與 Emoji ======================
QUADRANT_COLOR = {
    "Leading": 0x2ECC71,
    "Improving": 0x3498DB,
    "Weakening": 0xF1C40F,
    "Lagging": 0xE74C3C,
    "N/A": 0x95A5A6,
}

QUADRANT_EMOJI = {
    "Leading": "🟢",
    "Improving": "🔵",
    "Weakening": "🟡",
    "Lagging": "🔴",
    "N/A": "⚪",
}

# ====================== 建立 Discord Embed ======================
def build_embed(ranked_df: pd.DataFrame, rs_ratio: dict, rs_mom: dict):
    et = pytz.timezone("US/Eastern")
    now_str = datetime.now(et).strftime("%Y-%m-%d %H:%M ET")

    top_quad = ranked_df.iloc[0]["Quadrant"]
    color = QUADRANT_COLOR.get(top_quad, 0x5865F2)

    title = "📊 美股板塊相對強度排名（20 板塊）"
    description = (
        f"**資料時間**：{now_str}\n"
        f"**排序依據**：加權綜合 RS（1W 20% + 1M 40% + 3M 40%）\n"
        f"**比較基準**：SPY　|　含前一日排名變化"
    )

    # 排名總表
    rank_lines = []
    for _, row in ranked_df.iterrows():
        emoji = QUADRANT_EMOJI.get(row["Quadrant"], "⚪")
        change = row["RankChange"]
        change_str = f"`{change}`" if change != "→" else "`→`"
        rank_lines.append(
            f"`{row['Rank']:2d}` {change_str} {emoji} **{row['Sector']}** "
            f"({row['Ticker']})  "
            f"RS:`{row['RS_Composite']:+.1f}%`"
        )

    fields = [
        {
            "name": "🏆 綜合排名（含升降）",
            "value": "\n".join(rank_lines),
            "inline": False
        }
    ]

    # 各週期 RS
    detail_lines = []
    for _, row in ranked_df.iterrows():
        detail_lines.append(
            f"**{row['Sector']}**  "
            f"1W:`{row['RS_1W']:+.1f}`  "
            f"1M:`{row['RS_1M']:+.1f}`  "
            f"3M:`{row['RS_3M']:+.1f}`"
        )

    fields.append({
        "name": "📈 各週期相對強度 (RS)",
        "value": "\n".join(detail_lines),
        "inline": False
    })

    # RRG
    rrg_lines = []
    for _, row in ranked_df.iterrows():
        t = row["Ticker"]
        ratio = rs_ratio.get(t, np.nan)
        mom = rs_mom.get(t, np.nan)
        rrg_lines.append(
            f"**{row['Sector']}**  "
            f"Ratio:`{ratio:.1f}`  Mom:`{mom:+.1f}`  [{row['Quadrant']}]"
        )

    fields.append({
        "name": "🧭 RRG 指標",
        "value": "\n".join(rrg_lines),
        "inline": False
    })

    fields.append({
        "name": "📖 說明",
        "value": (
            "🟢 Leading　強勢且加速　｜　🔵 Improving　弱勢但轉強\n"
            "🟡 Weakening 強勢但減弱　｜　🔴 Lagging　弱勢且走弱\n"
            "`↑n` 排名上升 n 名　`↓n` 下降 n 名　`→` 持平　`NEW` 新進榜"
        ),
        "inline": False
    })

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Sector RS Bot • 20 Sectors • Data: yfinance"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    return embed

# ====================== 發送 Discord ======================
def send_discord(embed: dict):
    if not WEBHOOK_URL:
        print("⚠️ 未設定 DISCORD_WEBHOOK，僅印出結果")
        print(json.dumps(embed, indent=2, ensure_ascii=False))
        return

    payload = {
        "username": "Sector RS Bot",
        "embeds": [embed]
    }
    resp = requests.post(WEBHOOK_URL, json=payload, timeout=20)
    if resp.status_code in (200, 204):
        print("✅ 成功發送到 Discord")
    else:
        print(f"❌ 發送失敗：{resp.status_code}")
        print(resp.text)

# ====================== 主程式 ======================
def main():
    print("1️⃣ 抓取資料...")
    df = get_data(period="1y")

    print("2️⃣ 計算加權相對強度...")
    rs_df = calc_returns(df)

    print("3️⃣ 計算 RRG...")
    rs_ratio, rs_mom = calc_rrg(df)

    rs_df["RS_Ratio"] = rs_df["Ticker"].map(rs_ratio)
    rs_df["RS_Mom"] = rs_df["Ticker"].map(rs_mom)
    rs_df["Quadrant"] = rs_df.apply(
        lambda r: get_quadrant(r["RS_Ratio"], r["RS_Mom"]), axis=1
    )

    # 排序
    ranked = rs_df.sort_values("RS_Composite", ascending=False).reset_index(drop=True)
    ranked["Rank"] = ranked.index + 1

    # 計算與前一日排名變化
    print("4️⃣ 比較前一日排名...")
    prev_ranking = load_previous_ranking()
    ranked["RankChange"] = ranked.apply(
        lambda r: calc_rank_change(r["Rank"], prev_ranking.get(r["Ticker"])),
        axis=1
    )

    # 儲存今日排名（供下次比較）
    save_current_ranking(ranked)

    print("5️⃣ 建立 Embed 並發送...")
    embed = build_embed(ranked, rs_ratio, rs_mom)
    send_discord(embed)

    # 終端機預覽
    print("\n===== 排名預覽 =====")
    cols = ["Rank", "RankChange", "Sector", "RS_Composite", "Quadrant", "RS_1W", "RS_1M", "RS_3M"]
    print(ranked[cols].to_string(index=False))

if __name__ == "__main__":
    main()
