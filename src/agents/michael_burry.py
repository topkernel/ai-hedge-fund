from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class MichaelBurrySignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def michael_burry_agent(state: AgentState, agent_id: str = "michael_burry_agent"):
    """Analyse stocks using Michael Burry's deep‑value, contrarian framework."""
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]  # YYYY‑MM‑DD
    tickers: list[str] = data["tickers"]

    # We look one year back for insider trades / news flow
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取财务科目")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取内部人交易")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "获取公司新闻")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "分析价值")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "分析资产负债表")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status(agent_id, ticker, "分析内部人活动")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "分析逆向情绪")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        # ------------------------------------------------------------------
        # Aggregate score & derive preliminary signal
        # ------------------------------------------------------------------
        total_score = (
            value_analysis["score"]
            + balance_sheet_analysis["score"]
            + insider_analysis["score"]
            + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"]
            + balance_sheet_analysis["max_score"]
            + insider_analysis["max_score"]
            + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect data for LLM reasoning & output
        # ------------------------------------------------------------------
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "生成LLM输出")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=burry_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to the graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(burry_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "迈克尔·布瑞智能体")

    state["data"]["analyst_signals"][agent_id] = burry_analysis

    progress.update_status(agent_id, None, "完成")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub‑analysis helpers
###############################################################################


def _latest_line_item(line_items: list):
    """Return the most recent line‑item object or *None*."""
    return line_items[0] if line_items else None


# ----- Value ----------------------------------------------------------------

def _analyze_value(metrics, line_items, market_cap):
    """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""

    max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
    score = 0
    details: list[str] = []

    # Free‑cash‑flow yield
    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"极佳的自由现金流收益率 {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"非常高的自由现金流收益率 {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"可观的自由现金流收益率 {fcf_yield:.1%}")
        else:
            details.append(f"较低的自由现金流收益率 {fcf_yield:.1%}")
    else:
        details.append("自由现金流数据不可用")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f}（<6）")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f}（<10）")
            else:
                details.append(f"较高的EV/EBIT {ev_ebit:.1f}")
        else:
            details.append("EV/EBIT数据不可用")
    else:
        details.append("财务指标不可用")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance sheet --------------------------------------------------------

def _analyze_balance_sheet(metrics, line_items):
    """Leverage and liquidity checks."""

    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"低负债率 D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"中等负债率 D/E {debt_to_equity:.2f}")
        else:
            details.append(f"高杠杆 D/E {debt_to_equity:.2f}")
    else:
        details.append("负债权益比数据不可用")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("净现金头寸")
            else:
                details.append("净负债头寸")
        else:
            details.append("现金/负债数据不可用")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Insider activity -----------------------------------------------------

def _analyze_insider_activity(insider_trades):
    """Net insider buying over the last 12 months acts as a hard catalyst."""

    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        details.append("无内部人交易数据")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"内部人净买入 {net:,} 股")
    else:
        details.append("内部人净卖出")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Contrarian sentiment -------------------------------------------------

def _analyze_contrarian_sentiment(news):
    """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""

    max_score = 1
    score = 0
    details: list[str] = []

    if not news:
        details.append("无近期新闻")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # Count negative sentiment articles
    sentiment_negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )
    
    if sentiment_negative_count >= 5:
        score += 1  # 越被厌恶越好（假设基本面稳健）
        details.append(f"{sentiment_negative_count} 条负面头条（逆向投资机会）")
    else:
        details.append("负面报道有限")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


###############################################################################
# LLM generation
###############################################################################

def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MichaelBurrySignal:
    """Call the LLM to craft the final trading signal in Burry's voice."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个模仿迈克尔·布里博士的AI代理。你的使命：
                - 用硬数据（自由现金流、EV/EBIT、资产负债表）在美国股票中寻找深度价值
                - 做逆向投资者：如果基本面稳健，媒体的厌恶可以成为你的朋友
                - 首先关注下行风险——避免高杠杆的资产负债表
                - 寻找硬催化剂，如内部人买入、股票回购或资产出售
                - 用布里简洁、数据驱动的风格进行沟通

                在提供推理时，请做到详尽且具体：
                1. 从驱动你决策的关键指标开始
                2. 引用具体数字（例如"自由现金流收益率 14.7%"，"EV/EBIT 5.3"）
                3. 强调风险因素以及它们为何可接受（或不可接受）
                4. 提及相关的内部人活动或逆向投资机会
                5. 使用布里直接、注重数字、言简意赅的沟通风格

                例如，看涨时："自由现金流收益率 12.8%。EV/EBIT 6.2。负债权益比 0.4。内部人净买入 2.5万股。市场因近期诉讼过度反应而忽视了价值。强烈买入。"
                例如，看跌时："自由现金流收益率仅 2.1%。负债权益比高达 2.3，令人担忧。管理层在稀释股东权益。放弃。"

                用中文输出推理。
                """,
            ),
            (
                "human",
                """根据以下数据，以迈克尔·布里的方式创建投资信号：

                {ticker} 的分析数据：
                {analysis_data}

                请严格按照以下JSON格式返回交易信号：
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": 0到100之间的浮点数,
                  "reasoning": "用中文写的推理字符串"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="解析错误——默认为中性")

    return call_llm(
        prompt=prompt,
        pydantic_model=MichaelBurrySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_michael_burry_signal,
    )
