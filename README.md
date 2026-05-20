# A 股股票技术观察报告

这是一个使用 AKShare 获取 A 股历史行情和实时行情的 Python 项目。输入股票代码后，程序会计算 MA5、MA20、MA60、RSI、MACD、成交量变化、支撑压力位和趋势情景，并生成 Markdown 分析报告。

报告只输出观察信号，包括趋势判断、量价关系、技术指标和风险提示，不提供绝对买入或卖出结论。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI 使用

生成贵州茅台 `600519` 的默认报告：

```bash
python -m ashare_analyzer.cli 600519
```

指定历史行情区间和输出文件：

```bash
python -m ashare_analyzer.cli 600519 --start-date 20240101 --end-date 20260520 --output reports/600519.md
```

输出英文报告：

```bash
python -m ashare_analyzer.cli 600519 --language en
```

复权方式：

```bash
python -m ashare_analyzer.cli 600519 --adjust qfq
python -m ashare_analyzer.cli 600519 --adjust hfq
python -m ashare_analyzer.cli 600519 --adjust ""
```

如果股票代码、日期格式或日期区间不合法，命令行会输出中文错误提示。实时行情接口不可用时，程序会继续基于历史行情生成报告。

## Dashboard 网页界面

启动 Streamlit Dashboard：

```bash
streamlit run app.py
```

Dashboard 支持：

- 输入股票代码、起始日期、结束日期。
- 选择前复权、后复权或不复权。
- 选择短线、中线或长线分析周期。
- 点击按钮生成分析。
- 页面展示核心指标、K 线图、均线、成交量、MACD、RSI、趋势判断、动量判断、量价关系、支撑压力位、情景推演、交易观察和风险提示。
- 下载 Markdown 报告。

报告定位为“交易观察与风险提示”。页面可以给出观察买入区间、风险控制位、减仓观察位和趋势确认位，但这些价位仅基于历史数据和技术指标，不构成投资建议或绝对买卖指令。

## 测试

安装依赖后运行：

```bash
python -m unittest discover -v
```

## 项目结构

```text
.
├── app.py
├── ashare_analyzer
│   ├── __init__.py
│   ├── analyzer.py
│   ├── cli.py
│   └── web.py
├── tests
│   ├── __init__.py
│   └── test_analyzer.py
├── requirements.txt
└── README.md
```

## 指标说明

- MA5、MA20、MA60：分别代表 5 日、20 日、60 日移动平均线，用于观察短中期趋势。
- RSI：默认使用 14 日周期，用于观察价格动能和短线过热或偏弱状态。
- MACD：使用 EMA12、EMA26 和 DEA9，输出 DIF、DEA 和 MACD 柱。
- 成交量变化：包含单日成交量变化和相对 20 日均量的倍数。

## 注意事项

- AKShare 行情接口依赖第三方数据源，可能因网络、行情源调整或访问频率导致失败。
- 实时行情可能存在延迟，历史行情和实时行情字段也可能随 AKShare 版本变化。
- 技术指标是历史数据的统计结果，只适合辅助观察，不能替代基本面研究、风险控制或独立判断。
