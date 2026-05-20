from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from ashare_analyzer.analyzer import (
    AnalysisConfig,
    build_report,
    build_risk_items,
    calculate_indicators,
    classify_conclusion,
    fetch_history,
    fetch_realtime,
    format_number,
    format_percent,
    judge_macd,
    judge_trend,
    judge_volume,
    normalize_symbol,
)


def run_app() -> None:
    st.set_page_config(
        page_title="A 股技术观察",
        layout="wide",
    )

    st.title("A 股技术观察")
    st.caption("基于 AKShare 行情数据生成技术观察报告，不构成任何投资建议。")

    with st.sidebar:
        st.header("分析参数")
        symbol_input = st.text_input("股票代码", value="600519", max_chars=6)
        today = date.today()
        default_start = today - timedelta(days=420)
        start_date = st.date_input("开始日期", value=default_start)
        end_date = st.date_input("结束日期", value=today)
        adjust = st.selectbox(
            "复权方式",
            options=["qfq", "hfq", ""],
            format_func=lambda value: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[value],
        )
        language = st.selectbox(
            "报告语言",
            options=["zh", "en"],
            format_func=lambda value: {"zh": "中文", "en": "English"}[value],
        )
        submitted = st.button("生成分析", type="primary", use_container_width=True)

    if not submitted:
        st.info("输入股票代码和日期范围后，点击“生成分析”。")
        return

    try:
        symbol = normalize_symbol(symbol_input)
        config = AnalysisConfig(
            symbol=symbol,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjust,
        ).normalized()
        with st.spinner("正在获取行情并计算指标..."):
            history = fetch_history(config)
            analyzed = calculate_indicators(history)
            realtime = fetch_realtime(symbol)
            report = build_report(symbol, analyzed, realtime, language=language)
    except (RuntimeError, ValueError) as exc:
        st.error(f"错误：{exc}")
        return

    render_analysis(symbol, analyzed, report, language)


def render_analysis(
    symbol: str,
    analyzed: pd.DataFrame,
    report: str,
    language: str,
) -> None:
    latest = analyzed.iloc[-1]
    previous = analyzed.iloc[-2] if len(analyzed) >= 2 else latest
    trend_signal = judge_trend(latest, language)
    volume_signal = judge_volume(latest, language)
    macd_signal = judge_macd(latest, previous, language)
    stance = classify_conclusion(latest, previous, language)
    risk_items = build_risk_items(latest, analyzed, language)

    st.subheader("核心指标" if language == "zh" else "Key Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("最新价格" if language == "zh" else "Latest Close", format_number(latest["close"]))
    metric_cols[1].metric("MA20", format_number(latest["MA20"]))
    metric_cols[2].metric("RSI", format_number(latest["RSI"]))
    metric_cols[3].metric("结论分级" if language == "zh" else "Grade", stance)

    st.dataframe(
        latest_indicator_frame(latest, language),
        use_container_width=True,
        hide_index=True,
    )

    col_trend, col_volume = st.columns(2)
    with col_trend:
        st.subheader("趋势判断" if language == "zh" else "Trend")
        st.write(trend_signal)
        st.write(
            f"MA5 / MA20 / MA60: {format_number(latest['MA5'])} / "
            f"{format_number(latest['MA20'])} / {format_number(latest['MA60'])}"
        )
    with col_volume:
        st.subheader("成交量" if language == "zh" else "Volume")
        st.write(volume_signal)
        st.write(
            ("成交量日变化：" if language == "zh" else "Daily volume change: ")
            + format_percent(latest["VOL_CHANGE"])
        )
        st.write(
            ("相对 20 日均量：" if language == "zh" else "Volume vs. MA20: ")
            + f"{format_number(latest['VOL_RATIO_20'])}x"
        )

    col_momentum, col_risk = st.columns(2)
    with col_momentum:
        st.subheader("动量" if language == "zh" else "Momentum")
        st.write(macd_signal)
        st.write(
            f"MACD DIF / DEA / HIST: {format_number(latest['MACD_DIF'])} / "
            f"{format_number(latest['MACD_DEA'])} / {format_number(latest['MACD_HIST'])}"
        )
    with col_risk:
        st.subheader("风险提示" if language == "zh" else "Risk")
        for item in risk_items:
            st.write(f"- {item}")

    st.subheader("Markdown 报告" if language == "zh" else "Markdown Report")
    st.download_button(
        label="下载 Markdown 报告" if language == "zh" else "Download Markdown Report",
        data=report.encode("utf-8"),
        file_name=f"{symbol}_analysis.md",
        mime="text/markdown",
    )
    st.markdown(report)


def latest_indicator_frame(latest: pd.Series, language: str) -> pd.DataFrame:
    if language == "en":
        return pd.DataFrame(
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
    return pd.DataFrame(
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


if __name__ == "__main__":
    run_app()
