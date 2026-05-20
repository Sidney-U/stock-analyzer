from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ashare_analyzer.analyzer import (
    AnalysisConfig,
    build_dashboard_analysis,
    build_dashboard_report,
    calculate_indicators,
    fetch_history,
    format_number,
    format_percent,
    normalize_symbol,
    save_report,
)


def main() -> None:
    st.set_page_config(page_title="A股技术分析 Dashboard", layout="wide")
    st.title("A股技术分析 Dashboard")
    st.caption("交易观察与风险提示。所有分析基于历史行情和技术指标，不构成投资建议。")

    with st.sidebar:
        st.header("分析参数")
        today = date.today()
        symbol_input = st.text_input("股票代码", value="600519", max_chars=6)
        start_date = st.date_input("起始日期", value=today - timedelta(days=420))
        end_date = st.date_input("结束日期", value=today)
        adjust_label = st.selectbox("复权方式", ["前复权", "后复权", "不复权"])
        period = st.radio("分析周期", ["短线", "中线", "长线"], horizontal=True, index=1)
        submitted = st.button("生成分析", type="primary", use_container_width=True)

    if not submitted:
        st.info("请输入股票代码、日期范围、复权方式和分析周期，然后点击“生成分析”。")
        return

    adjust = {"前复权": "qfq", "后复权": "hfq", "不复权": ""}[adjust_label]
    try:
        symbol = normalize_symbol(symbol_input)
        config = AnalysisConfig(
            symbol=symbol,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjust,
        ).normalized()

        with st.spinner("正在获取历史行情并计算技术指标..."):
            history = fetch_history(config)
            analyzed = calculate_indicators(history)
            analysis = build_dashboard_analysis(analyzed, period=period)
            report = build_dashboard_report(symbol, analyzed, analysis)
            report_path = save_report(report, Path("reports") / f"{symbol}_dashboard.md")
    except (RuntimeError, ValueError) as exc:
        st.error(f"错误：{exc}")
        return

    render_dashboard(symbol, analyzed, analysis, report, report_path)


def render_dashboard(
    symbol: str,
    analyzed: pd.DataFrame,
    analysis: dict[str, object],
    report: str,
    report_path: Path,
) -> None:
    st.success(f"{symbol} 分析完成。报告已保存到：{report_path}")

    st.subheader("核心指标")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘价", format_number(analysis["latest_close"]), format_percent(analysis["pct_change"]))
    c2.metric("成交量变化", format_percent(analysis["volume_change"]))
    c3.metric("RSI", format_number(analysis["rsi"]))
    c4.metric("趋势判断", str(analysis["trend_grade"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("MA5", format_number(analysis["ma5"]))
    c6.metric("MA20", format_number(analysis["ma20"]))
    c7.metric("MA60", format_number(analysis["ma60"]))
    c8.metric("MACD Histogram", format_number(analysis["macd_hist"]))

    st.dataframe(build_metric_frame(analysis), use_container_width=True, hide_index=True)

    st.subheader("技术图表")
    st.plotly_chart(price_volume_figure(analyzed), use_container_width=True)
    col_macd, col_rsi = st.columns(2)
    with col_macd:
        st.plotly_chart(macd_figure(analyzed), use_container_width=True)
    with col_rsi:
        st.plotly_chart(rsi_figure(analyzed), use_container_width=True)

    st.subheader("深度解读")
    d1, d2, d3 = st.columns(3)
    d1.info(f"趋势判断：{analysis['trend_grade']}\n\n{analysis['trend_text']}")
    d2.info(f"动量判断：{analysis['momentum']}\n\nMACD DIF / DEA / Histogram：{format_number(analysis['macd_dif'])} / {format_number(analysis['macd_dea'])} / {format_number(analysis['macd_hist'])}")
    d3.info(f"量价关系：{analysis['volume_price']}\n\n成交量变化：{format_percent(analysis['volume_change'])}")

    support = analysis["support"]
    resistance = analysis["resistance"]
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("支撑位")
        st.write(f"- 近20日低点：{format_number(support['近20日低点'])}")
        st.write(f"- MA20：{format_number(support['MA20'])}")
        st.write(f"- MA60：{format_number(support['MA60'])}")
    with s2:
        st.subheader("压力位")
        st.write(f"- 近20日高点：{format_number(resistance['近20日高点'])}")
        st.write(f"- 近60日高点：{format_number(resistance['近60日高点'])}")
        st.write(f"- 风险位：{analysis['risk_text']}")

    st.subheader("情景分析")
    for name, text in analysis["scenarios"].items():
        st.write(f"- **{name}**：{text}")

    st.subheader("交易观察")
    buy_low, buy_high = analysis["observation_buy_range"]
    st.write(f"- 观察买入区间：{format_number(buy_low)} - {format_number(buy_high)}")
    st.write(f"- 风险控制位：{format_number(analysis['risk_control_level'])}")
    st.write(f"- 减仓观察位：{format_number(analysis['reduce_watch_level'])}")
    st.write(f"- 趋势确认位：{format_number(analysis['trend_confirm_level'])}")
    st.warning("以上为基于历史数据和技术指标的参考观察位，不构成投资建议或绝对买卖指令。")

    st.subheader("风险提示")
    for item in analysis["risk_items"]:
        st.write(f"- {item}")

    st.subheader("Markdown 报告")
    st.download_button(
        label="下载 Markdown 报告",
        data=report.encode("utf-8"),
        file_name=report_path.name,
        mime="text/markdown",
    )
    st.markdown(report)


def build_metric_frame(analysis: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
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


def price_volume_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    for ma in ["MA5", "MA20", "MA60"]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[ma], mode="lines", name=ma), row=1, col=1)
    colors = ["#d94f45" if close >= open_ else "#2e8b57" for close, open_ in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量", marker_color=colors), row=2, col=1)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def macd_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#d94f45" if value >= 0 else "#2e8b57" for value in df["MACD_HIST"]]
    fig.add_bar(x=df["date"], y=df["MACD_HIST"], name="Histogram", marker_color=colors)
    fig.add_scatter(x=df["date"], y=df["MACD_DIF"], mode="lines", name="DIF")
    fig.add_scatter(x=df["date"], y=df["MACD_DEA"], mode="lines", name="DEA")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def rsi_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=df["RSI"], mode="lines", name="RSI")
    fig.add_hline(y=70, line_dash="dash", line_color="#d94f45")
    fig.add_hline(y=30, line_dash="dash", line_color="#2e8b57")
    fig.update_layout(height=360, yaxis_range=[0, 100], margin=dict(l=20, r=20, t=30, b=20))
    return fig


if __name__ == "__main__":
    main()
