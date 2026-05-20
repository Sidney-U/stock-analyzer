from __future__ import annotations

import http.client
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests
import urllib3


HISTORY_FETCH_ERROR = "历史行情获取失败：可能是网络、数据源或东方财富接口暂时不可用，请稍后重试。"
HISTORY_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]
NETWORK_EXCEPTIONS = (
    requests.exceptions.RequestException,
    urllib3.exceptions.ProtocolError,
    http.client.RemoteDisconnected,
)


@dataclass(frozen=True)
class AnalysisConfig:
    symbol: str
    start_date: str | None = None
    end_date: str | None = None
    adjust: str = "qfq"

    def normalized(self) -> "AnalysisConfig":
        start = validate_trade_date(self.start_date, "start_date")
        end = validate_trade_date(self.end_date, "end_date")
        if start and end and start > end:
            raise ValueError("开始日期不能晚于结束日期")
        return AnalysisConfig(
            symbol=normalize_symbol(self.symbol),
            start_date=start,
            end_date=end,
            adjust=self.adjust,
        )


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip()
    if not clean.isdigit() or len(clean) != 6:
        raise ValueError("股票代码应为 6 位数字，例如 600519")
    return clean


def validate_trade_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None

    clean = value.strip()
    if not clean:
        return None
    if len(clean) != 8 or not clean.isdigit():
        raise ValueError(f"{field_name} 应为 YYYYMMDD 格式，例如 20240101")
    try:
        datetime.strptime(clean, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} 不是有效日期：{clean}") from exc
    return clean


def default_start_date(days: int = 420) -> str:
    return (date.today() - timedelta(days=days)).strftime("%Y%m%d")


def default_end_date() -> str:
    return date.today().strftime("%Y%m%d")


def fetch_history(config: AnalysisConfig) -> pd.DataFrame:
    config = config.normalized()
    start = config.start_date or default_start_date()
    end = config.end_date or default_end_date()

    try:
        raw_df = fetch_history_from_eastmoney(config, start, end)
        return normalize_history_dataframe(raw_df, config.symbol)
    except Exception:
        try:
            raw_df = fetch_history_from_sina(config, start, end)
            return normalize_history_dataframe(raw_df, config.symbol)
        except Exception as exc:
            raise RuntimeError(HISTORY_FETCH_ERROR) from exc


def fetch_history_from_eastmoney(
    config: AnalysisConfig, start: str, end: str
) -> pd.DataFrame:
    return ak.stock_zh_a_hist(
        symbol=config.symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=config.adjust,
    )


def fetch_history_from_sina(
    config: AnalysisConfig, start: str, end: str
) -> pd.DataFrame:
    return ak.stock_zh_a_daily(
        symbol=market_symbol(config.symbol),
        start_date=start,
        end_date=end,
        adjust=config.adjust,
    )


def market_symbol(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def normalize_history_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError(f"未获取到 {symbol} 的历史行情数据")

    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    required_columns = ["open", "close", "high", "low", "volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = "、".join(missing_columns)
        raise RuntimeError(f"{symbol} 的历史行情缺少必要字段：{missing_text}")

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "amount" not in df.columns:
        df["amount"] = np.nan
    if "change" not in df.columns:
        df["change"] = df["close"].diff()
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100
    if "amplitude" not in df.columns:
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        raise RuntimeError(f"{symbol} 的历史行情数据不足，至少需要 2 个交易日")
    return df[HISTORY_COLUMNS]


def fetch_realtime(symbol: str) -> dict[str, object]:
    symbol = normalize_symbol(symbol)
    try:
        spot_df = ak.stock_zh_a_spot_em()
    except Exception:
        return {}
    if spot_df.empty:
        return {}

    code_column = "代码"
    matched = spot_df[spot_df[code_column].astype(str) == symbol]
    if matched.empty:
        return {}

    row = matched.iloc[0]
    return {str(key): row[key] for key in spot_df.columns}


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]

    result["MA5"] = close.rolling(window=5, min_periods=5).mean()
    result["MA20"] = close.rolling(window=20, min_periods=20).mean()
    result["MA60"] = close.rolling(window=60, min_periods=60).mean()
    result["RSI"] = calculate_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD_DIF"] = ema12 - ema26
    result["MACD_DEA"] = result["MACD_DIF"].ewm(span=9, adjust=False).mean()
    result["MACD_HIST"] = (result["MACD_DIF"] - result["MACD_DEA"]) * 2

    result["VOL_MA5"] = result["volume"].rolling(window=5, min_periods=5).mean()
    result["VOL_MA20"] = result["volume"].rolling(window=20, min_periods=20).mean()
    result["VOL_CHANGE"] = result["volume"].pct_change() * 100
    result["VOL_RATIO_20"] = result["volume"] / result["VOL_MA20"]

    return result


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_report(
    symbol: str,
    df: pd.DataFrame,
    realtime: dict[str, object],
    language: str = "zh",
) -> str:
    if language not in {"zh", "en"}:
        raise ValueError("language 仅支持 zh 或 en")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest

    stock_name = str(realtime.get("名称", ""))
    title_name = f"{symbol} {stock_name}".strip()
    trend_signal = judge_trend(latest, language)
    volume_signal = judge_volume(latest, language)
    rsi_signal = judge_rsi(latest["RSI"], language)
    macd_signal = judge_macd(latest, previous, language)
    stance = classify_conclusion(latest, previous, language)
    risk_items = build_risk_items(latest, df, language)
    summary_items = build_executive_summary(
        latest=latest,
        previous=previous,
        df=df,
        stance=stance,
        trend_signal=trend_signal,
        volume_signal=volume_signal,
        rsi_signal=rsi_signal,
        macd_signal=macd_signal,
        language=language,
    )

    realtime_lines = format_realtime_lines(realtime, language)
    metric_table = latest_metric_table(latest, language)

    if language == "en":
        report = [
            f"# A-Share Technical Observation Report: {title_name}",
            "",
            f"- Report date: {date.today().isoformat()}",
            f"- Historical data range: {df['date'].iloc[0].date()} to {latest['date'].date()}",
            "- Data source: AKShare historical and realtime market data interfaces",
            "- Note: This report provides observation signals only and does not offer buy or sell advice.",
            "",
            "## Executive Summary",
            "",
            *[f"- {item}" for item in summary_items],
            "",
            "## Conclusion Grade",
            "",
            f"- Current grade: {stance}",
            "- Interpretation: the grade reflects technical conditions only, not an investment recommendation.",
            "",
            "## Latest Indicator Snapshot",
            "",
            metric_table,
            "",
            "## Realtime Market Overview",
            "",
            *realtime_lines,
            "",
            "## Trend",
            "",
            f"- Latest close: {format_number(latest['close'])}",
            f"- MA5 / MA20 / MA60: {format_number(latest['MA5'])} / {format_number(latest['MA20'])} / {format_number(latest['MA60'])}",
            f"- Observation: {trend_signal}",
            "",
            "## Momentum",
            "",
            f"- RSI(14): {format_number(latest['RSI'])}. {rsi_signal}",
            f"- MACD DIF / DEA / histogram: {format_number(latest['MACD_DIF'])} / {format_number(latest['MACD_DEA'])} / {format_number(latest['MACD_HIST'])}",
            f"- MACD observation: {macd_signal}",
            "",
            "## Volume",
            "",
            f"- Latest volume: {format_number(latest['volume'], decimals=0)}",
            f"- Daily volume change: {format_percent(latest['VOL_CHANGE'])}",
            f"- Volume vs. 20-day average: {format_number(latest['VOL_RATIO_20'])}x",
            f"- Observation: {volume_signal}",
            "",
            "## Risk",
            "",
            *[f"- {item}" for item in risk_items],
            "",
            "## Recent 10 Trading Days",
            "",
            recent_table(df.tail(10), language),
            "",
        ]
        return "\n".join(report)

    report = [
        f"# A 股技术观察报告：{title_name}",
        "",
        f"- 报告日期：{date.today().isoformat()}",
        f"- 历史数据区间：{df['date'].iloc[0].date()} 至 {latest['date'].date()}",
        "- 数据来源：AKShare 历史行情与实时行情接口",
        "- 说明：本报告只输出观察信号，不构成任何投资建议，也不提供绝对买入或卖出结论。",
        "",
        "## Executive Summary",
        "",
        *[f"- {item}" for item in summary_items],
        "",
        "## 结论分级",
        "",
        f"- 当前分级：{stance}",
        "- 解读：分级仅反映技术面观察状态，不构成买入或卖出建议。",
        "",
        "## 最新指标速览",
        "",
        metric_table,
        "",
        "## 实时行情概览",
        "",
        *realtime_lines,
        "",
        "## 趋势",
        "",
        f"- 最新收盘价：{format_number(latest['close'])}",
        f"- MA5 / MA20 / MA60：{format_number(latest['MA5'])} / {format_number(latest['MA20'])} / {format_number(latest['MA60'])}",
        f"- 观察信号：{trend_signal}",
        "",
        "## 动量",
        "",
        f"- RSI(14)：{format_number(latest['RSI'])}，{rsi_signal}",
        f"- MACD DIF / DEA / 柱：{format_number(latest['MACD_DIF'])} / {format_number(latest['MACD_DEA'])} / {format_number(latest['MACD_HIST'])}",
        f"- MACD 观察信号：{macd_signal}",
        "",
        "## 成交量",
        "",
        f"- 最新成交量：{format_number(latest['volume'], decimals=0)}",
        f"- 成交量日变化：{format_percent(latest['VOL_CHANGE'])}",
        f"- 成交量相对 20 日均量：{format_number(latest['VOL_RATIO_20'])} 倍",
        f"- 观察信号：{volume_signal}",
        "",
        "## 风险",
        "",
        *[f"- {item}" for item in risk_items],
        "",
        "## 最近 10 个交易日指标",
        "",
        recent_table(df.tail(10)),
        "",
    ]
    return "\n".join(report)


def build_executive_summary(
    latest: pd.Series,
    previous: pd.Series,
    df: pd.DataFrame,
    stance: str,
    trend_signal: str,
    volume_signal: str,
    rsi_signal: str,
    macd_signal: str,
    language: str,
) -> list[str]:
    recent_high = df["high"].tail(60).max()
    recent_low = df["low"].tail(60).min()

    if language == "en":
        summary = [
            f"Conclusion grade is {stance}, based on moving averages, RSI, MACD and volume behavior.",
            f"Trend: {trend_signal}",
            f"Momentum: RSI is {format_number(latest['RSI'])}; {macd_signal}",
            f"Volume: {volume_signal}",
        ]
        if pd.notna(recent_high) and latest["close"] >= recent_high * 0.97:
            summary.append("Price is close to the recent 60-day high, so pullback volatility deserves attention.")
        elif pd.notna(recent_low) and latest["close"] <= recent_low * 1.03:
            summary.append("Price is close to the recent 60-day low, so trend continuation risk deserves attention.")
        else:
            summary.append("Risk view: no extreme 60-day high or low proximity is detected.")
        return summary

    summary = [
        f"结论分级为{stance}，综合参考均线、RSI、MACD 与成交量状态。",
        f"趋势：{trend_signal}",
        f"动量：RSI 为 {format_number(latest['RSI'])}；{macd_signal}",
        f"成交量：{volume_signal}",
    ]
    if pd.notna(recent_high) and latest["close"] >= recent_high * 0.97:
        summary.append("价格接近近 60 日高位，需留意回撤波动。")
    elif pd.notna(recent_low) and latest["close"] <= recent_low * 1.03:
        summary.append("价格接近近 60 日低位，需留意趋势延续风险。")
    else:
        summary.append("风险：暂未触发近 60 日高低位极端位置提示。")
    return summary


def build_dashboard_analysis(df: pd.DataFrame, period: str = "中线") -> dict[str, object]:
    if period not in {"短线", "中线", "长线"}:
        raise ValueError("分析周期仅支持：短线 / 中线 / 长线")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    lookback = {"短线": 20, "中线": 60, "长线": 120}[period]
    recent = df.tail(min(len(df), lookback))
    high_20 = df["high"].tail(20).max()
    low_20 = df["low"].tail(20).min()
    high_60 = df["high"].tail(60).max()
    low_60 = df["low"].tail(60).min()

    trend_grade = classify_conclusion(latest, previous, "zh")
    momentum = judge_momentum_state(df)
    volume_price = judge_volume_price_state(latest)
    support = {
        "近20日低点": low_20,
        "MA20": latest["MA20"],
        "MA60": latest["MA60"],
    }
    resistance = {
        "近20日高点": high_20,
        "近60日高点": high_60,
    }
    risk_control = min(
        value
        for value in [low_20 * 0.98, latest["MA60"] * 0.97]
        if pd.notna(value)
    )
    buy_low = max(low_20, latest["MA20"] * 0.98) if pd.notna(latest["MA20"]) else low_20
    buy_high = latest["MA20"] * 1.02 if pd.notna(latest["MA20"]) else low_20 * 1.03
    if buy_low > buy_high:
        buy_low, buy_high = buy_high, buy_low

    weakening_reasons = []
    if latest["RSI"] > 70:
        weakening_reasons.append("RSI 高于 70")
    if latest["MACD_HIST"] < previous["MACD_HIST"]:
        weakening_reasons.append("MACD 柱体动能衰减")
    if latest["close"] >= high_20 * 0.98:
        weakening_reasons.append("价格接近近20日高点")
    if latest["close"] >= high_60 * 0.98:
        weakening_reasons.append("价格接近近60日高点")

    analysis = {
        "period": period,
        "lookback_days": lookback,
        "latest_close": latest["close"],
        "pct_change": latest["pct_change"],
        "volume_change": latest["VOL_CHANGE"],
        "ma5": latest["MA5"],
        "ma20": latest["MA20"],
        "ma60": latest["MA60"],
        "rsi": latest["RSI"],
        "macd_dif": latest["MACD_DIF"],
        "macd_dea": latest["MACD_DEA"],
        "macd_hist": latest["MACD_HIST"],
        "high_20": high_20,
        "low_20": low_20,
        "high_60": high_60,
        "low_60": low_60,
        "trend_grade": trend_grade,
        "trend_text": judge_trend(latest),
        "momentum": momentum,
        "volume_price": volume_price,
        "support": support,
        "resistance": resistance,
        "risk_level": risk_control,
        "risk_text": (
            f"若收盘价跌破 MA20（{format_number(latest['MA20'])}）或 MA60"
            f"（{format_number(latest['MA60'])}），则趋势结构转弱风险上升；风险控制位参考"
            f" {format_number(risk_control)}。"
        ),
        "observation_buy_range": (buy_low, buy_high),
        "risk_control_level": risk_control,
        "reduce_watch_level": max(high_20, high_60),
        "trend_confirm_level": high_20,
        "reduce_reasons": weakening_reasons or ["价格处于压力位附近且量能扩张不足"],
        "scenarios": build_scenarios(latest, high_20),
        "risk_items": build_risk_items(latest, df),
        "period_high": recent["high"].max(),
        "period_low": recent["low"].min(),
    }
    return analysis


def judge_momentum_state(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    compare = df.iloc[-20] if len(df) >= 20 else df.iloc[0]
    price_made_progress = latest["close"] > compare["close"]
    macd_weaker = latest["MACD_HIST"] < compare["MACD_HIST"]
    rsi_weaker = latest["RSI"] < compare["RSI"]
    if price_made_progress and macd_weaker and rsi_weaker:
        return "背离"
    if latest["MACD_DIF"] > latest["MACD_DEA"] and latest["MACD_HIST"] > previous["MACD_HIST"]:
        return "增强"
    if latest["MACD_HIST"] < previous["MACD_HIST"] or latest["MACD_DIF"] < latest["MACD_DEA"]:
        return "衰减"
    return "中性"


def judge_volume_price_state(latest: pd.Series) -> str:
    price_up = latest["pct_change"] >= 0
    volume_up = latest["VOL_RATIO_20"] >= 1.2 or latest["VOL_CHANGE"] > 0
    if price_up and volume_up:
        return "放量上涨"
    if price_up and not volume_up:
        return "缩量上涨"
    if not price_up and volume_up:
        return "放量下跌"
    return "缩量下跌"


def build_scenarios(latest: pd.Series, high_20: float) -> dict[str, str]:
    return {
        "强势情景": (
            f"若价格有效突破近20日高点 {format_number(high_20)}，且成交量高于20日均量，"
            "则趋势确认信号增强。"
        ),
        "中性情景": (
            f"若价格围绕 MA20（{format_number(latest['MA20'])}）上下震荡，"
            "则市场处于方向选择阶段。"
        ),
        "弱势情景": (
            f"若价格跌破 MA60（{format_number(latest['MA60'])}）或 MACD 出现死叉，"
            "则趋势转弱风险上升。"
        ),
    }


def dashboard_metric_table(analysis: dict[str, object]) -> str:
    table = pd.DataFrame(
        [
            {
                "最新收盘价": format_number(analysis["latest_close"]),
                "涨跌幅": format_percent(analysis["pct_change"]),
                "成交量变化": format_percent(analysis["volume_change"]),
                "MA5": format_number(analysis["ma5"]),
                "MA20": format_number(analysis["ma20"]),
                "MA60": format_number(analysis["ma60"]),
                "RSI": format_number(analysis["rsi"]),
                "MACD DIF": format_number(analysis["macd_dif"]),
                "MACD DEA": format_number(analysis["macd_dea"]),
                "MACD Histogram": format_number(analysis["macd_hist"]),
                "近20日最高": format_number(analysis["high_20"]),
                "近20日最低": format_number(analysis["low_20"]),
                "近60日最高": format_number(analysis["high_60"]),
                "近60日最低": format_number(analysis["low_60"]),
            }
        ]
    )
    return table.to_markdown(index=False)


def build_dashboard_report(
    symbol: str,
    df: pd.DataFrame,
    analysis: dict[str, object] | None = None,
) -> str:
    analysis = analysis or build_dashboard_analysis(df)
    buy_low, buy_high = analysis["observation_buy_range"]
    support = analysis["support"]
    resistance = analysis["resistance"]
    scenarios = analysis["scenarios"]
    risk_items = analysis["risk_items"]
    reduce_reasons = "、".join(analysis["reduce_reasons"])

    report = [
        f"# A股技术分析 Dashboard 报告：{symbol}",
        "",
        "本报告定位为交易观察与风险提示，基于历史行情与技术指标生成，不构成任何投资建议，也不提供绝对买入或卖出结论。涉及后续走势的内容均采用条件情景表达。",
        "",
        "## Executive Summary",
        "",
        f"- 分析周期：{analysis['period']}，观察窗口约 {analysis['lookback_days']} 个交易日。",
        f"- 趋势判断：{analysis['trend_grade']}；当前均线与价格结构显示：{analysis['trend_text']}",
        f"- 动量判断：{analysis['momentum']}；量价关系：{analysis['volume_price']}。",
        f"- 观察买入区间：{format_number(buy_low)} - {format_number(buy_high)}；风险控制位：{format_number(analysis['risk_control_level'])}。",
        "- 所有价位均为历史数据和技术指标推导的参考区间，应结合实时行情、基本面与风险承受能力审慎评估。",
        "",
        "## 核心指标表",
        "",
        dashboard_metric_table(analysis),
        "",
        "## 趋势分析",
        "",
        f"- 趋势判断：{analysis['trend_grade']}。",
        f"- 均线结构：MA5 {format_number(analysis['ma5'])} / MA20 {format_number(analysis['ma20'])} / MA60 {format_number(analysis['ma60'])}。",
        f"- 解读：{analysis['trend_text']}",
        "",
        "## 动量分析",
        "",
        f"- 动量判断：{analysis['momentum']}。",
        f"- MACD DIF / DEA / Histogram：{format_number(analysis['macd_dif'])} / {format_number(analysis['macd_dea'])} / {format_number(analysis['macd_hist'])}。",
        f"- RSI：{format_number(analysis['rsi'])}。",
        "",
        "## 量价分析",
        "",
        f"- 量价关系：{analysis['volume_price']}。",
        f"- 成交量变化：{format_percent(analysis['volume_change'])}。",
        "",
        "## 支撑/压力位",
        "",
        f"- 支撑位：近20日低点 {format_number(support['近20日低点'])}，MA20 {format_number(support['MA20'])}，MA60 {format_number(support['MA60'])}。",
        f"- 压力位：近20日高点 {format_number(resistance['近20日高点'])}，近60日高点 {format_number(resistance['近60日高点'])}。",
        f"- 风险位：{analysis['risk_text']}",
        "",
        "## 情景推演",
        "",
        *[f"- {name}：{text}" for name, text in scenarios.items()],
        "",
        "## 交易观察",
        "",
        f"- 观察买入区间：{format_number(buy_low)} - {format_number(buy_high)}，该区间参考 MA20 附近、近20日低点上方以及 RSI 回落后的企稳区域。",
        f"- 风险控制位：{format_number(analysis['risk_control_level'])}，该位置参考 MA60 下方 2%-3% 以及近20日低点下方。",
        f"- 减仓观察位：{format_number(analysis['reduce_watch_level'])}，重点观察近20日/近60日高点、RSI > 70 或 MACD 动能衰减。当前触发关注因素：{reduce_reasons}。",
        f"- 趋势确认位：若有效突破近20日高点 {format_number(analysis['trend_confirm_level'])} 并伴随放量，则趋势确认度提升。",
        "",
        "## 风险提示",
        "",
        *[f"- {item}" for item in risk_items],
        "- 上述观察位不是确定性交易指令；若消息面、流动性、业绩或系统性风险发生变化，则技术结构可能快速失效。",
        "",
    ]
    return "\n".join(report)


def latest_metric_table(latest: pd.Series, language: str) -> str:
    if language == "en":
        table = pd.DataFrame(
            [
                {
                    "Latest Close": format_number(latest["close"]),
                    "MA5": format_number(latest["MA5"]),
                    "MA20": format_number(latest["MA20"]),
                    "MA60": format_number(latest["MA60"]),
                    "RSI": format_number(latest["RSI"]),
                    "MACD": format_number(latest["MACD_HIST"]),
                }
            ]
        )
    else:
        table = pd.DataFrame(
            [
                {
                    "最新价格": format_number(latest["close"]),
                    "MA5": format_number(latest["MA5"]),
                    "MA20": format_number(latest["MA20"]),
                    "MA60": format_number(latest["MA60"]),
                    "RSI": format_number(latest["RSI"]),
                    "MACD": format_number(latest["MACD_HIST"]),
                }
            ]
        )
    return table.to_markdown(index=False)


def classify_conclusion(latest: pd.Series, previous: pd.Series, language: str) -> str:
    score = 0
    if pd.notna(latest["MA60"]):
        if latest["close"] > latest["MA5"] > latest["MA20"] > latest["MA60"]:
            score += 2
        elif latest["close"] < latest["MA5"] < latest["MA20"] < latest["MA60"]:
            score -= 2
        elif latest["close"] > latest["MA20"]:
            score += 1
        elif latest["close"] < latest["MA20"]:
            score -= 1

    if latest["RSI"] >= 55:
        score += 1
    elif latest["RSI"] <= 45:
        score -= 1

    if latest["MACD_DIF"] > latest["MACD_DEA"] and latest["MACD_HIST"] >= previous["MACD_HIST"]:
        score += 1
    elif latest["MACD_DIF"] < latest["MACD_DEA"] and latest["MACD_HIST"] <= previous["MACD_HIST"]:
        score -= 1

    if latest["VOL_RATIO_20"] >= 1.5 and latest["pct_change"] > 0:
        score += 1
    elif latest["VOL_RATIO_20"] >= 1.5 and latest["pct_change"] < 0:
        score -= 1

    if score >= 2:
        return "偏强" if language == "zh" else "Bullish"
    if score <= -2:
        return "偏弱" if language == "zh" else "Weak"
    return "中性" if language == "zh" else "Neutral"


def format_realtime_lines(realtime: dict[str, object], language: str = "zh") -> list[str]:
    if not realtime:
        if language == "en":
            return ["- No realtime quote matched; the report is based on historical data."]
        return ["- 未匹配到实时行情记录，报告仍基于历史行情生成。"]

    fields = (
        [
            ("Name", "名称"),
            ("Last Price", "最新价"),
            ("Change %", "涨跌幅"),
            ("Volume", "成交量"),
            ("Turnover", "成交额"),
            ("Turnover Rate", "换手率"),
        ]
        if language == "en"
        else [
            ("名称", "名称"),
            ("最新价", "最新价"),
            ("涨跌幅", "涨跌幅"),
            ("成交量", "成交量"),
            ("成交额", "成交额"),
            ("换手率", "换手率"),
        ]
    )
    lines = []
    for label, key in fields:
        if key in realtime and pd.notna(realtime[key]):
            suffix = "%" if key in {"涨跌幅", "换手率"} else ""
            separator = ": " if language == "en" else "："
            lines.append(f"- {label}{separator}{realtime[key]}{suffix}")
    if language == "en":
        return lines or ["- Realtime quote fields are empty; the report is based on historical data."]
    return lines or ["- 实时行情字段为空，报告仍基于历史行情生成。"]


def judge_trend(row: pd.Series, language: str = "zh") -> str:
    close = row["close"]
    ma5 = row["MA5"]
    ma20 = row["MA20"]
    ma60 = row["MA60"]
    if pd.isna(ma60):
        if language == "en":
            return "History is insufficient to evaluate MA60, so the trend signal is less reliable."
        return "历史样本不足以完整评估 MA60，趋势信号偏弱。"
    if close > ma5 > ma20 > ma60:
        if language == "en":
            return "Price is above stacked moving averages, indicating a relatively strong short-to-medium-term trend."
        return "价格位于多条均线之上，短中期趋势相对偏强。"
    if close < ma5 < ma20 < ma60:
        if language == "en":
            return "Price is below stacked moving averages, indicating trend pressure."
        return "价格位于多条均线之下，短中期趋势承压。"
    if close > ma20 and ma5 > ma20:
        if language == "en":
            return "Price and MA5 are above MA20, suggesting trend improvement."
        return "短期价格和 MA5 站上 MA20，趋势有改善迹象。"
    if close < ma20 and ma5 < ma20:
        if language == "en":
            return "Price and MA5 are below MA20, so the trend view remains cautious."
        return "短期价格和 MA5 低于 MA20，趋势偏谨慎。"
    if language == "en":
        return "Moving averages are mixed, so the trend direction is not yet clear."
    return "均线排列交织，趋势方向暂不清晰，适合继续观察确认。"


def judge_volume(row: pd.Series, language: str = "zh") -> str:
    vol_ratio = row["VOL_RATIO_20"]
    price_change = row["pct_change"]
    if pd.isna(vol_ratio):
        if language == "en":
            return "Volume history is insufficient to assess volume conditions."
        return "历史成交量样本不足，暂不判断量能状态。"
    if vol_ratio >= 1.5 and price_change > 0:
        if language == "en":
            return "Price rose on higher volume, indicating more active participation, but persistence should be watched."
        return "放量上涨，说明资金活跃度提升，但需要观察持续性。"
    if vol_ratio >= 1.5 and price_change < 0:
        if language == "en":
            return "Price fell on higher volume, indicating heavier selling pressure or wider disagreement."
        return "放量下跌，说明抛压或分歧加大。"
    if vol_ratio <= 0.7:
        if language == "en":
            return "Volume is below the 20-day average, indicating weaker participation."
        return "成交量低于 20 日均量，市场参与度偏低。"
    if language == "en":
        return "Volume is near the 20-day average, with no obvious abnormal volume-price signal."
    return "成交量接近 20 日均量，量价关系暂无明显异常。"


def judge_rsi(rsi: float, language: str = "zh") -> str:
    if rsi >= 70:
        if language == "en":
            return "RSI is elevated, so short-term overheating risk has increased."
        return "RSI 处于偏高区间，短线过热风险上升。"
    if rsi <= 30:
        if language == "en":
            return "RSI is low, indicating weak short-term sentiment or a potential repair watchpoint."
        return "RSI 处于偏低区间，短线情绪偏弱或存在修复观察点。"
    if rsi >= 55:
        if language == "en":
            return "RSI is in a neutral-to-strong range."
        return "RSI 位于中性偏强区间。"
    if rsi <= 45:
        if language == "en":
            return "RSI is in a neutral-to-weak range."
        return "RSI 位于中性偏弱区间。"
    if language == "en":
        return "RSI is in a neutral range."
    return "RSI 位于中性区间。"


def judge_macd(latest: pd.Series, previous: pd.Series, language: str = "zh") -> str:
    dif = latest["MACD_DIF"]
    dea = latest["MACD_DEA"]
    hist = latest["MACD_HIST"]
    previous_hist = previous["MACD_HIST"]
    if dif > dea and hist > previous_hist:
        if language == "en":
            return "DIF is above DEA and the histogram is expanding, indicating improving momentum."
        return "DIF 位于 DEA 上方且柱体扩张，动能边际增强。"
    if dif < dea and hist < previous_hist:
        if language == "en":
            return "DIF is below DEA and the histogram is weakening, indicating fading momentum."
        return "DIF 位于 DEA 下方且柱体走弱，动能边际减弱。"
    if dif > dea:
        if language == "en":
            return "DIF is above DEA, but momentum change still needs confirmation from price and volume."
        return "DIF 位于 DEA 上方，但动能变化仍需结合量价确认。"
    if language == "en":
        return "DIF is below DEA, so a momentum repair signal is still pending."
    return "DIF 位于 DEA 下方，趋势修复信号仍需等待。"


def build_risk_items(latest: pd.Series, df: pd.DataFrame, language: str = "zh") -> list[str]:
    if language == "en":
        items = [
            "AKShare data comes from third-party market data sources and may be delayed, adjusted, or temporarily unavailable.",
            "Technical indicators summarize historical price and volume only and should not be used as standalone decision inputs.",
        ]
    else:
        items = [
            "AKShare 数据来自第三方行情源，可能存在延迟、字段调整或临时不可用。",
            "技术指标主要反映历史价格和成交量，不应单独作为决策依据。",
        ]

    recent_high = df["high"].tail(60).max()
    recent_low = df["low"].tail(60).min()
    close = latest["close"]
    if recent_high and close >= recent_high * 0.97:
        if language == "en":
            items.append("Price is close to the recent 60-day high; watch for elevated volatility and pullback risk.")
        else:
            items.append("价格接近近 60 日高位，需留意高位波动和回撤风险。")
    if recent_low and close <= recent_low * 1.03:
        if language == "en":
            items.append("Price is close to the recent 60-day low; watch for trend continuation and liquidity risk.")
        else:
            items.append("价格接近近 60 日低位，趋势延续和流动性风险权重上升。")
    if latest["RSI"] >= 70:
        if language == "en":
            items.append("When RSI is elevated, short-term volatility may increase.")
        else:
            items.append("RSI 偏高，短线波动风险权重上升。")
    if latest["VOL_RATIO_20"] >= 2:
        if language == "en":
            items.append("A large volume expansion often indicates greater market disagreement.")
        else:
            items.append("成交量显著放大，通常意味着市场分歧提升。")

    return items


def recent_table(df: pd.DataFrame, language: str = "zh") -> str:
    table = df[
        [
            "date",
            "close",
            "pct_change",
            "MA5",
            "MA20",
            "MA60",
            "RSI",
            "MACD_HIST",
            "VOL_RATIO_20",
        ]
    ].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    if language == "en":
        rename_map = {
            "date": "Date",
            "close": "Close",
            "pct_change": "Change %",
            "MA5": "MA5",
            "MA20": "MA20",
            "MA60": "MA60",
            "RSI": "RSI",
            "MACD_HIST": "MACD Hist",
            "VOL_RATIO_20": "Vol/MA20",
        }
    else:
        rename_map = {
            "date": "日期",
            "close": "收盘",
            "pct_change": "涨跌幅%",
            "MA5": "MA5",
            "MA20": "MA20",
            "MA60": "MA60",
            "RSI": "RSI",
            "MACD_HIST": "MACD柱",
            "VOL_RATIO_20": "量比20日",
        }
    table = table.rename(columns=rename_map)
    return table.to_markdown(index=False, floatfmt=".2f")


def format_number(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def format_percent(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def save_report(report: str, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
