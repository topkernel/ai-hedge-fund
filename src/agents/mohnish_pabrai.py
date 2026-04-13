from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class MohnishPabraiSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def mohnish_pabrai_agent(state: AgentState, agent_id: str = "mohnish_pabrai_agent"):
    """Evaluate stocks using Mohnish Pabrai's checklist and 'heads I win, tails I don't lose much' approach."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data: dict[str, any] = {}
    pabrai_analysis: dict[str, any] = {}

    # Pabrai focuses on: downside protection, simple business, moat via unit economics, FCF yield vs alternatives,
    # and potential for doubling in 2-3 years at low risk.
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=8, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取财务科目")
        line_items = search_line_items(
            ticker,
            [
                # Profitability and cash generation
                "revenue",
                "gross_profit",
                "gross_margin",
                "operating_income",
                "operating_margin",
                "net_income",
                "free_cash_flow",
                # Balance sheet - debt and liquidity
                "total_debt",
                "cash_and_equivalents",
                "current_assets",
                "current_liabilities",
                "shareholders_equity",
                # Capital intensity
                "capital_expenditure",
                "depreciation_and_amortization",
                # Shares outstanding for per-share context
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=8,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "分析下行保护")
        downside = analyze_downside_protection(line_items)

        progress.update_status(agent_id, ticker, "分析现金收益率与估值")
        valuation = analyze_pabrai_valuation(line_items, market_cap)

        progress.update_status(agent_id, ticker, "评估翻倍潜力")
        double_potential = analyze_double_potential(line_items, market_cap)

        # Combine to an overall score in spirit of Pabrai: heavily weight downside and cash yield
        total_score = (
            downside["score"] * 0.45
            + valuation["score"] * 0.35
            + double_potential["score"] * 0.20
        )
        max_score = 10

        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.0:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "downside_protection": downside,
            "valuation": valuation,
            "double_potential": double_potential,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "生成帕布莱分析")
        pabrai_output = generate_pabrai_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        pabrai_analysis[ticker] = {
            "signal": pabrai_output.signal,
            "confidence": pabrai_output.confidence,
            "reasoning": pabrai_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=pabrai_output.reasoning)

    message = HumanMessage(content=json.dumps(pabrai_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(pabrai_analysis, "莫尼什·帕布莱智能体")

    progress.update_status(agent_id, None, "完成")

    state["data"]["analyst_signals"][agent_id] = pabrai_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_downside_protection(financial_line_items: list) -> dict[str, any]:
    """Assess balance-sheet strength and downside resiliency (capital preservation first)."""
    if not financial_line_items:
        return {"score": 0, "details": "数据不足"}

    latest = financial_line_items[0]
    details: list[str] = []
    score = 0

    cash = getattr(latest, "cash_and_equivalents", None)
    debt = getattr(latest, "total_debt", None)
    current_assets = getattr(latest, "current_assets", None)
    current_liabilities = getattr(latest, "current_liabilities", None)
    equity = getattr(latest, "shareholders_equity", None)

    # Net cash position is a strong downside protector
    net_cash = None
    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0:
            score += 3
            details.append(f"净现金头寸：${net_cash:,.0f}")
        else:
            details.append(f"净负债头寸：${net_cash:,.0f}")

    # Current ratio
    if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"流动性强劲（流动比率 {current_ratio:.2f}）")
        elif current_ratio >= 1.2:
            score += 1
            details.append(f"流动性充足（流动比率 {current_ratio:.2f}）")
        else:
            details.append(f"流动性较弱（流动比率 {current_ratio:.2f}）")

    # Low leverage
    if equity is not None and equity > 0 and debt is not None:
        de_ratio = debt / equity
        if de_ratio < 0.3:
            score += 2
            details.append(f"极低杠杆（D/E {de_ratio:.2f}）")
        elif de_ratio < 0.7:
            score += 1
            details.append(f"中等杠杆（D/E {de_ratio:.2f}）")
        else:
            details.append(f"高杠杆（D/E {de_ratio:.2f}）")

    # Free cash flow positive and stable
    fcf_values = [getattr(li, "free_cash_flow", None) for li in financial_line_items if getattr(li, "free_cash_flow", None) is not None]
    if fcf_values and len(fcf_values) >= 3:
        recent_avg = sum(fcf_values[:3]) / 3
        older = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
        if recent_avg > 0 and recent_avg >= older:
            score += 2
            details.append("自由现金流为正且在改善/稳定")
        elif recent_avg > 0:
            score += 1
            details.append("自由现金流为正但在下降")
        else:
            details.append("自由现金流为负")

    return {"score": min(10, score), "details": "; ".join(details)}


def analyze_pabrai_valuation(financial_line_items: list, market_cap: float | None) -> dict[str, any]:
    """Value via simple FCF yield and asset-light preference (keep it simple, low mistakes)."""
    if not financial_line_items or market_cap is None or market_cap <= 0:
        return {"score": 0, "details": "数据不足", "fcf_yield": None, "normalized_fcf": None}

    details: list[str] = []
    fcf_values = [getattr(li, "free_cash_flow", None) for li in financial_line_items if getattr(li, "free_cash_flow", None) is not None]
    capex_vals = [abs(getattr(li, "capital_expenditure", 0) or 0) for li in financial_line_items]

    if not fcf_values or len(fcf_values) < 3:
        return {"score": 0, "details": "自由现金流历史数据不足", "fcf_yield": None, "normalized_fcf": None}

    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    if normalized_fcf <= 0:
        return {"score": 0, "details": "标准化自由现金流为负", "fcf_yield": None, "normalized_fcf": normalized_fcf}

    fcf_yield = normalized_fcf / market_cap

    score = 0
    if fcf_yield > 0.10:
        score += 4
        details.append(f"极佳价值：自由现金流收益率 {fcf_yield:.1%}")
    elif fcf_yield > 0.07:
        score += 3
        details.append(f"有吸引力的价值：自由现金流收益率 {fcf_yield:.1%}")
    elif fcf_yield > 0.05:
        score += 2
        details.append(f"合理价值：自由现金流收益率 {fcf_yield:.1%}")
    elif fcf_yield > 0.03:
        score += 1
        details.append(f"临界价值：自由现金流收益率 {fcf_yield:.1%}")
    else:
        details.append(f"偏贵：自由现金流收益率 {fcf_yield:.1%}")

    # Asset-light tilt: lower capex intensity preferred
    if capex_vals and len(financial_line_items) >= 3:
        revenue_vals = [getattr(li, "revenue", None) for li in financial_line_items]
        capex_to_revenue = []
        for i, li in enumerate(financial_line_items):
            revenue = getattr(li, "revenue", None)
            capex = abs(getattr(li, "capital_expenditure", 0) or 0)
            if revenue and revenue > 0:
                capex_to_revenue.append(capex / revenue)
        if capex_to_revenue:
            avg_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_ratio < 0.05:
                score += 2
                details.append(f"轻资产：平均资本支出占收入 {avg_ratio:.1%}")
            elif avg_ratio < 0.10:
                score += 1
                details.append(f"中等资本支出：平均资本支出占收入 {avg_ratio:.1%}")
            else:
                details.append(f"资本支出重：平均资本支出占收入 {avg_ratio:.1%}")

    return {"score": min(10, score), "details": "; ".join(details), "fcf_yield": fcf_yield, "normalized_fcf": normalized_fcf}


def analyze_double_potential(financial_line_items: list, market_cap: float | None) -> dict[str, any]:
    """Estimate low-risk path to double capital in ~2-3 years: runway from FCF growth + rerating."""
    if not financial_line_items or market_cap is None or market_cap <= 0:
        return {"score": 0, "details": "数据不足"}

    details: list[str] = []

    # 使用收入和自由现金流趋势作为粗略增长代理（保持简单）
    revenues = [getattr(li, "revenue", None) for li in financial_line_items if getattr(li, "revenue", None) is not None]
    fcfs = [getattr(li, "free_cash_flow", None) for li in financial_line_items if getattr(li, "free_cash_flow", None) is not None]

    score = 0
    if revenues and len(revenues) >= 3:
        recent_rev = sum(revenues[:3]) / 3
        older_rev = sum(revenues[-3:]) / 3 if len(revenues) >= 6 else revenues[-1]
        if older_rev > 0:
            rev_growth = (recent_rev / older_rev) - 1
            if rev_growth > 0.15:
                score += 2
                details.append(f"强劲的收入增长轨迹（{rev_growth:.1%}）")
            elif rev_growth > 0.05:
                score += 1
                details.append(f"温和的收入增长（{rev_growth:.1%}）")

    if fcfs and len(fcfs) >= 3:
        recent_fcf = sum(fcfs[:3]) / 3
        older_fcf = sum(fcfs[-3:]) / 3 if len(fcfs) >= 6 else fcfs[-1]
        if older_fcf != 0:
            fcf_growth = (recent_fcf / older_fcf) - 1
            if fcf_growth > 0.20:
                score += 3
                details.append(f"强劲的自由现金流增长（{fcf_growth:.1%}）")
            elif fcf_growth > 0.08:
                score += 2
                details.append(f"健康的自由现金流增长（{fcf_growth:.1%}）")
            elif fcf_growth > 0:
                score += 1
                details.append(f"正自由现金流增长（{fcf_growth:.1%}）")

    # If FCF yield is already high (>8%), doubling can come from cash generation alone in few years
    tmp_val = analyze_pabrai_valuation(financial_line_items, market_cap)
    fcf_yield = tmp_val.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield > 0.08:
            score += 3
            details.append("高自由现金流收益率可通过留存现金/回购推动翻倍")
        elif fcf_yield > 0.05:
            score += 1
            details.append("合理的自由现金流收益率支持温和复利")

    return {"score": min(10, score), "details": "; ".join(details)}


def generate_pabrai_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> MohnishPabraiSignal:
    """Generate Pabrai-style decision focusing on low risk, high uncertainty bets and cloning."""
    template = ChatPromptTemplate.from_messages([
        (
          "system",
          """你是莫尼什·帕布莱。运用我的价值投资哲学：

          - 正面我赢，反面我不怎么输：优先考虑下行保护。
          - 购买业务简单、可理解且具有持久护城河的企业。
          - 要求高自由现金流收益率和低杠杆；偏好轻资产模式。
          - 寻找内在价值在上升而价格显著偏低的情形。
          - 偏好克隆伟大投资者的想法和清单，而非追求新奇。
          - 寻求在2-3年内以低风险翻倍资本的潜力。
          - 避免杠杆、复杂性和脆弱的资产负债表。

            提供坦诚的、清单驱动的推理，重点强调资本保全和预期错误定价。用中文输出推理。
            """,
        ),
        (
          "human",
          """使用提供的数据分析 {ticker}。

          数据：
          {analysis_data}

          请严格按照以下JSON格式返回：
          {{
            "signal": "bullish" | "bearish" | "neutral",
            "confidence": 0到100之间的浮点数,
            "reasoning": "用中文写的帕布莱风格分析，重点阐述下行保护、自由现金流收益率和翻倍潜力"
          }}
          """,
        ),
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
    })

    def create_default_pabrai_signal():
        return MohnishPabraiSignal(signal="neutral", confidence=0.0, reasoning="分析出错，默认为中性")

    return call_llm(
        prompt=prompt,
        state=state,
        pydantic_model=MohnishPabraiSignal,
        agent_name=agent_id,
        default_factory=create_default_pabrai_signal,
    ) 