# AI 对冲基金

这是一个基于 AI 的对冲基金概念验证项目。目标是探索如何使用 AI 进行交易决策。本项目仅供**教育**用途，不用于实际交易或投资。

本系统由多个智能体协同工作：

1. 阿斯沃斯·达摩达兰智能体 — 估值学之父，专注于企业故事、数字和严谨的估值分析
2. 本杰明·格雷厄姆智能体 — 价值投资之父，只买入具有安全边际的隐藏宝石
3. 比尔·阿克曼智能体 — 激进投资者，大胆建仓并推动变革
4. 凯瑟琳·伍德智能体 — 成长投资女王，坚信创新与颠覆的力量
5. 查理·芒格智能体 — 巴菲特的黄金搭档，只以合理价格买入优秀企业
6. 迈克尔·布瑞智能体 — 大空头逆向投资者，猎取深度价值
7. 莫尼什·帕布莱智能体 — 达恩多投资者，以低风险追求翻倍回报
8. 纳西姆·塔勒布智能体 — 黑天鹅风险分析师，关注尾部风险、反脆弱性和不对称收益
9. 彼得·林奇智能体 — 实战投资者，在日常业务中寻找"十倍股"
10. 菲利普·费舍尔智能体 — 严谨的成长投资者，采用深度"闲聊"调研法
11. 拉凯什·琼君瓦拉智能体 — 印度大牛市
12. 斯坦利·德鲁肯米勒智能体 — 宏观投资传奇，寻找具有增长潜力的不对称机会
13. 沃伦·巴菲特智能体 — 奥马哈先知，以合理价格寻找优秀企业
14. 估值分析智能体 — 计算股票内在价值并生成交易信号
15. 情绪分析智能体 — 分析市场情绪并生成交易信号
16. 基本面分析智能体 — 分析基本面数据并生成交易信号
17. 技术分析智能体 — 分析技术指标并生成交易信号
18. 风险管理器 — 计算风险指标并设定仓位限制
19. 投资组合经理 — 做出最终交易决策并生成订单

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

注意：本系统不会实际执行任何交易。

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## 免责声明

本项目仅供**教育和研究用途**。

- 不用于实际交易或投资
- 不提供任何投资建议或收益保证
- 作者不对任何财务损失承担责任
- 投资决策请咨询专业理财顾问
- 过往表现不代表未来收益

使用本软件即表示您同意仅将其用于学习目的。

## 目录
- [安装方法](#安装方法)
- [运行方法](#运行方法)
  - [命令行界面](#命令行界面)
  - [Web 应用](#web-应用)
- [如何贡献](#如何贡献)
- [功能建议](#功能建议)
- [许可证](#许可证)

## 安装方法

在运行 AI 对冲基金之前，您需要安装依赖并配置 API 密钥。以下步骤适用于全栈 Web 应用和命令行界面。

### 1. 克隆仓库

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. 配置 API 密钥

创建 `.env` 文件来存放您的 API 密钥：
```bash
# 在项目根目录创建 .env 文件
cp .env.example .env
```

打开并编辑 `.env` 文件，填入您的 API 密钥：
```bash
# 用于运行 OpenAI 托管的 LLM（gpt-4o、gpt-4o-mini 等）
OPENAI_API_KEY=your-openai-api-key

# 用于获取金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key

# 数据源：使用 "akshare" 分析 A 股，使用 "financial_datasets" 分析美股
DATA_SOURCE=akshare
```

**重要提示**：您必须至少设置一个 LLM API 密钥（如 `OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY`）才能运行本系统。

### A 股支持（AKShare）

本项目支持通过 AKShare 获取中国 A 股数据。设置方法：

1. 在 `.env` 文件中设置 `DATA_SOURCE=akshare`
2. 使用 6 位数字的股票代码（如 `600418` 代表江淮汽车，`000001` 代表平安银行）
3. AKShare 无需额外 API 密钥

## 运行方法

### 命令行界面

您可以直接通过终端运行 AI 对冲基金。这种方式提供更精细的控制，适合自动化、脚本和集成用途。

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### 快速开始

1. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装依赖：
```bash
poetry install
```

#### 运行 AI 对冲基金
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

分析 A 股：
```bash
poetry run python src/main.py --ticker 600418,000001 --analysts-all --model glm-5.1
```

您也可以添加 `--ollama` 标志来使用本地 LLM。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

您可以指定开始和结束日期，在特定时间段内做出决策。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### 运行回测
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**示例输出：**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

注意：`--ollama`、`--start-date` 和 `--end-date` 标志同样适用于回测。

### Web 应用

运行 AI 对冲基金的新方式是通过我们的 Web 应用，它提供了用户友好的界面。推荐给偏好可视化界面的用户。

详细的 Web 应用安装和运行说明请参见[这里](https://github.com/virattt/ai-hedge-fund/tree/main/app)。

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />

## 如何贡献

1. Fork 本仓库
2. 创建功能分支
3. 提交您的更改
4. 推送到分支
5. 创建 Pull Request

**重要提示**：请保持您的 Pull Request 小而聚焦，以便于审查和合并。

## 功能建议

如果您有功能建议，请提交一个 [issue](https://github.com/virattt/ai-hedge-fund/issues)，并确保标记为 `enhancement`。

## 许可证

本项目基于 MIT 许可证 — 详情请参见 LICENSE 文件。
