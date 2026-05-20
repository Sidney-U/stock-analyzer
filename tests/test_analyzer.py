from __future__ import annotations

import unittest
import http.client
from datetime import date
from unittest.mock import patch

import pandas as pd
import requests
import urllib3

from ashare_analyzer.analyzer import (
    AnalysisConfig,
    HISTORY_FETCH_ERROR,
    build_dashboard_analysis,
    build_dashboard_report,
    build_report,
    calculate_indicators,
    fetch_history,
    format_number,
    normalize_symbol,
    validate_trade_date,
)
from ashare_analyzer.cli import main
from ashare_analyzer.web import latest_indicator_frame


def eastmoney_history(rows: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "日期": dates.strftime("%Y-%m-%d"),
            "开盘": [10.0 + index for index in range(rows)],
            "收盘": [10.5 + index for index in range(rows)],
            "最高": [10.8 + index for index in range(rows)],
            "最低": [9.8 + index for index in range(rows)],
            "成交量": [1000 + index * 100 for index in range(rows)],
            "成交额": [100000 + index * 10000 for index in range(rows)],
            "振幅": [3.0] * rows,
            "涨跌幅": [1.0] * rows,
            "涨跌额": [0.1] * rows,
            "换手率": [1.0] * rows,
        }
    )


def sina_history(rows: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = [10.5 + index for index in range(rows)]
    return pd.DataFrame(
        {
            "date": dates.date,
            "open": [10.0 + index for index in range(rows)],
            "high": [10.8 + index for index in range(rows)],
            "low": [9.8 + index for index in range(rows)],
            "close": close,
            "volume": [1000 + index * 100 for index in range(rows)],
            "amount": [100000 + index * 10000 for index in range(rows)],
            "turnover": [0.01] * rows,
        }
    )


def sample_history(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = pd.Series([10 + index * 0.1 for index in range(rows)])
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "volume": [1000 + index * 10 for index in range(rows)],
            "amount": [100000 + index * 1000 for index in range(rows)],
            "amplitude": [2.0] * rows,
            "pct_change": [0.5] * rows,
            "change": [0.05] * rows,
            "turnover": [1.0] * rows,
        }
    )


class ValidationTests(unittest.TestCase):
    def test_normalize_symbol_accepts_six_digit_code(self) -> None:
        self.assertEqual(normalize_symbol(" 600519 "), "600519")

    def test_normalize_symbol_rejects_invalid_code(self) -> None:
        with self.assertRaisesRegex(ValueError, "6 位数字"):
            normalize_symbol("60051A")

    def test_validate_trade_date_accepts_valid_date(self) -> None:
        self.assertEqual(validate_trade_date("20240101", "start_date"), "20240101")

    def test_validate_trade_date_rejects_invalid_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "不是有效日期"):
            validate_trade_date("20240231", "start_date")

    def test_config_normalized_rejects_reversed_range(self) -> None:
        config = AnalysisConfig("600519", start_date="20240102", end_date="20240101")
        with self.assertRaisesRegex(ValueError, "开始日期不能晚于结束日期"):
            config.normalized()


class IndicatorTests(unittest.TestCase):
    def test_calculate_indicators_adds_expected_columns(self) -> None:
        analyzed = calculate_indicators(sample_history())
        latest = analyzed.iloc[-1]

        self.assertIn("MA5", analyzed.columns)
        self.assertIn("MACD_HIST", analyzed.columns)
        self.assertIn("VOL_RATIO_20", analyzed.columns)
        self.assertGreater(latest["MA5"], latest["MA20"])
        self.assertGreater(latest["VOL_RATIO_20"], 1)

    def test_build_report_contains_core_sections_without_realtime(self) -> None:
        analyzed = calculate_indicators(sample_history())
        report = build_report("600519", analyzed, {})

        self.assertIn("# A 股技术观察报告：600519", report)
        self.assertIn("## Executive Summary", report)
        self.assertIn("## 结论分级", report)
        self.assertIn("## 最新指标速览", report)
        self.assertIn("## 实时行情概览", report)
        self.assertIn("## 趋势", report)
        self.assertIn("## 动量", report)
        self.assertIn("## 成交量", report)
        self.assertIn("## 风险", report)
        self.assertIn("最新价格", report)
        self.assertRegex(report, r"当前分级：(偏强|中性|偏弱)")
        self.assertIn("## 最近 10 个交易日指标", report)
        self.assertIn(date.today().isoformat(), report)

    def test_build_report_supports_english(self) -> None:
        analyzed = calculate_indicators(sample_history())
        report = build_report("600519", analyzed, {}, language="en")

        self.assertIn("# A-Share Technical Observation Report: 600519", report)
        self.assertIn("## Executive Summary", report)
        self.assertIn("## Conclusion Grade", report)
        self.assertIn("## Latest Indicator Snapshot", report)
        self.assertIn("## Trend", report)
        self.assertIn("## Momentum", report)
        self.assertIn("## Volume", report)
        self.assertIn("## Risk", report)
        self.assertIn("Latest Close", report)
        self.assertRegex(report, r"Current grade: (Bullish|Neutral|Weak)")
        self.assertNotIn("买入", report)
        self.assertNotIn("卖出", report)

    def test_format_number_handles_missing_values(self) -> None:
        self.assertEqual(format_number(float("nan")), "N/A")

    def test_streamlit_indicator_frame_uses_report_fields(self) -> None:
        analyzed = calculate_indicators(sample_history())
        frame = latest_indicator_frame(analyzed.iloc[-1], "zh")

        self.assertEqual(
            ["最新价格", "MA5", "MA20", "MA60", "RSI", "MACD"],
            list(frame.columns),
        )
        self.assertEqual(frame.iloc[0]["最新价格"], format_number(analyzed.iloc[-1]["close"]))

    def test_dashboard_analysis_contains_deep_reading_fields(self) -> None:
        analyzed = calculate_indicators(sample_history())
        analysis = build_dashboard_analysis(analyzed, period="中线")

        self.assertIn(analysis["trend_grade"], {"偏强", "中性", "偏弱"})
        self.assertIn(analysis["momentum"], {"增强", "衰减", "背离", "中性"})
        self.assertIn(
            analysis["volume_price"],
            {"放量上涨", "缩量上涨", "放量下跌", "缩量下跌"},
        )
        self.assertIn("近20日低点", analysis["support"])
        self.assertIn("近60日高点", analysis["resistance"])
        self.assertIn("强势情景", analysis["scenarios"])
        self.assertIn("observation_buy_range", analysis)

    def test_dashboard_report_uses_required_sections_without_absolute_advice(self) -> None:
        analyzed = calculate_indicators(sample_history())
        analysis = build_dashboard_analysis(analyzed, period="短线")
        report = build_dashboard_report("600519", analyzed, analysis)

        for section in [
            "## Executive Summary",
            "## 核心指标表",
            "## 趋势分析",
            "## 动量分析",
            "## 量价分析",
            "## 支撑/压力位",
            "## 情景推演",
            "## 交易观察",
            "## 风险提示",
        ]:
            self.assertIn(section, report)
        self.assertIn("观察买入区间", report)
        self.assertIn("风险控制位", report)
        self.assertIn("减仓观察位", report)
        self.assertIn("趋势确认位", report)
        self.assertIn("不构成任何投资建议", report)
        self.assertNotIn("建议买入", report)
        self.assertNotIn("建议卖出", report)

    def test_dashboard_report_uses_conditional_scenario_language(self) -> None:
        analyzed = calculate_indicators(sample_history())
        analysis = build_dashboard_analysis(analyzed, period="中线")
        report = build_dashboard_report("600519", analyzed, analysis)

        self.assertIn("涉及后续走势的内容均采用条件情景表达", report)
        self.assertRegex(report, r"若价格有效突破近20日高点 .+，且成交量高于20日均量，则趋势确认信号增强。")
        self.assertRegex(report, r"若价格围绕 MA20（.+）上下震荡，则市场处于方向选择阶段。")
        self.assertRegex(report, r"若价格跌破 MA60（.+）或 MACD 出现死叉，则趋势转弱风险上升。")
        self.assertRegex(report, r"若有效突破近20日高点 .+ 并伴随放量，则趋势确认度提升。")
        self.assertNotIn("趋势确认信号会更清晰", report)
        self.assertNotIn("需要警惕趋势转弱", report)
        self.assertNotIn("市场可能", report)


class NetworkErrorTests(unittest.TestCase):
    def test_fetch_history_uses_primary_source_when_available(self) -> None:
        config = AnalysisConfig("600519", start_date="20240101", end_date="20240131")

        with patch(
            "ashare_analyzer.analyzer.ak.stock_zh_a_hist",
            return_value=eastmoney_history(),
        ) as primary:
            with patch("ashare_analyzer.analyzer.ak.stock_zh_a_daily") as fallback:
                df = fetch_history(config)

        primary.assert_called_once()
        fallback.assert_not_called()
        self.assertEqual(
            ["date", "open", "close", "high", "low", "volume"],
            list(df.columns[:6]),
        )

    def test_fetch_history_falls_back_to_sina_when_primary_fails(self) -> None:
        config = AnalysisConfig("600519", start_date="20240101", end_date="20240131")

        with patch(
            "ashare_analyzer.analyzer.ak.stock_zh_a_hist",
            side_effect=requests.exceptions.ConnectionError("network down"),
        ):
            with patch(
                "ashare_analyzer.analyzer.ak.stock_zh_a_daily",
                return_value=sina_history(),
            ) as fallback:
                df = fetch_history(config)

        fallback.assert_called_once_with(
            symbol="sh600519",
            start_date="20240101",
            end_date="20240131",
            adjust="qfq",
        )
        self.assertIn("pct_change", df.columns)
        self.assertIn("amplitude", df.columns)
        self.assertIn("change", df.columns)
        self.assertFalse(df["volume"].isna().any())

    def test_fetch_history_wraps_network_exceptions(self) -> None:
        config = AnalysisConfig("600519", start_date="20240101", end_date="20240131")
        network_errors = [
            requests.exceptions.ConnectionError("network down"),
            urllib3.exceptions.ProtocolError("connection broken"),
            http.client.RemoteDisconnected("remote end closed connection"),
        ]

        for error in network_errors:
            with self.subTest(error=type(error).__name__):
                with patch(
                    "ashare_analyzer.analyzer.ak.stock_zh_a_hist",
                    side_effect=error,
                ):
                    with patch(
                        "ashare_analyzer.analyzer.ak.stock_zh_a_daily",
                        side_effect=error,
                    ):
                        with self.assertRaisesRegex(RuntimeError, HISTORY_FETCH_ERROR):
                            fetch_history(config)

    def test_cli_reports_history_fetch_failure_without_traceback(self) -> None:
        with patch(
            "sys.argv",
            ["ashare-analyzer", "600519", "--start-date", "20240101"],
        ):
            with patch(
                "ashare_analyzer.cli.fetch_history",
                side_effect=RuntimeError(HISTORY_FETCH_ERROR),
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()

        message = str(raised.exception)
        self.assertIn(f"错误：{HISTORY_FETCH_ERROR}", message)
        self.assertNotIn("Traceback", message)


if __name__ == "__main__":
    unittest.main()
