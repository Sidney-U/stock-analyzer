from __future__ import annotations

import argparse
from pathlib import Path

from ashare_analyzer.analyzer import (
    AnalysisConfig,
    build_report,
    calculate_indicators,
    fetch_history,
    fetch_realtime,
    normalize_symbol,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 A 股股票技术观察 Markdown 报告")
    parser.add_argument("symbol", help="6 位 A 股股票代码，例如 600519")
    parser.add_argument("--start-date", help="历史行情开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", help="历史行情结束日期，格式 YYYYMMDD")
    parser.add_argument(
        "--adjust",
        default="qfq",
        choices=["", "qfq", "hfq"],
        help="复权方式：空字符串为不复权，qfq 为前复权，hfq 为后复权",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径，默认 reports/{symbol}_analysis.md",
    )
    parser.add_argument(
        "--language",
        default="zh",
        choices=["zh", "en"],
        help="报告语言：zh 为中文，en 为英文，默认 zh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        symbol = normalize_symbol(args.symbol)
        output = args.output or f"reports/{symbol}_analysis.md"
        config = AnalysisConfig(
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
        ).normalized()

        history = fetch_history(config)
        analyzed = calculate_indicators(history)
        realtime = fetch_realtime(symbol)
        report = build_report(symbol, analyzed, realtime, language=args.language)
        output_path = save_report(report, Path(output))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc

    print(f"报告已生成：{output_path}")


if __name__ == "__main__":
    main()
