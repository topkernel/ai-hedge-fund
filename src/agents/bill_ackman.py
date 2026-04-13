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


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState, agent_id: str = "bill_ackman_agent"):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    ackman_analysis = {}
    
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)
        
        progress.update_status(agent_id, ticker, "收集财务报表项目")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )
        
        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        
        progress.update_status(agent_id, ticker, "分析业务质量")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)
        
        progress.update_status(agent_id, ticker, "分析资产负债表和资本结构")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)
        
        progress.update_status(agent_id, ticker, "分析激进主义潜力")
        activism_analysis = analyze_activism_potential(financial_line_items)
        
        progress.update_status(agent_id, ticker, "计算内在价值和安全边际")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)
        
        # Combine partial scores or signals
        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
        )
        max_possible_score = 20  # Adjust weighting as desired (5 from each sub-analysis, for instance)
        
        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis
        }
        
        progress.update_status(agent_id, ticker, "生成比尔·阿克曼分析")
        ackman_output = generate_ackman_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )
        
        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning
        }
        
        progress.update_status(agent_id, ticker, "完成", analysis=ackman_output.reasoning)
    
    # Wrap results in a single message for the chain
    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name=agent_id
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "比尔·阿克曼智能体")
    
    # Add signals to the overall state
    state["data"]["analyst_signals"][agent_id] = ackman_analysis

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages (moats), and potential for long-term growth.
    Also tries to infer brand strength if intangible_assets data is present (optional).
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "数据不足以分析业务质量"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        initial, final = revenues[-1], revenues[0]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% cumulative growth
                score += 2
                details.append(f"收入在整个期间增长了 {(growth_rate*100):.1f}% (增长强劲)。")
            else:
                score += 1
                details.append(f"收入增长为正但累计不到50% ({(growth_rate*100):.1f}%)。")
        else:
            details.append("收入未显著增长或数据不足。")
    else:
        details.append("收入数据不足以进行多期趋势分析。")
    
    # 2. Operating margin and free cash flow consistency
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("营业利润率经常超过15% (盈利能力良好)。")
        else:
            details.append("营业利润率未持续高于15%。")
    else:
        details.append("各期无营业利润率数据。")
    
    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("大多数期间自由现金流为正。")
        else:
            details.append("自由现金流未持续为正。")
    else:
        details.append("各期无自由现金流数据。")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"ROE高达 {latest_metrics.return_on_equity:.1%}, 表明具有竞争优势。")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE为 {latest_metrics.return_on_equity:.1%}, 适中。")
    else:
        details.append("ROE数据不可用。")
    
    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [item.intangible_assets for item in financial_line_items if item.intangible_assets]
    # if intangible_vals and sum(intangible_vals) > 0:
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")
    #     score += 1
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "数据不足以分析财务纪律"
        }
    
    # 1. Multi-period debt ratio or debt_to_equity
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("大多数期间债务股本比 < 1.0 (杠杆合理)。")
        else:
            details.append("多个期间债务股本比 >= 1.0 (杠杆可能偏高)。")
    else:
        # Fallback to total_liabilities / total_assets
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("大多数期间负债资产比 < 50%。")
            else:
                details.append("多个期间负债资产比 >= 50%。")
        else:
            details.append("无一致的杠杆比率数据。")
    
    # 2. Capital allocation approach (dividends + share counts)
    dividends_list = [
        item.dividends_and_other_cash_distributions
        for item in financial_line_items
        if item.dividends_and_other_cash_distributions is not None
    ]
    if dividends_list:
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("公司有向股东返还资本的历史 (分红)。")
        else:
            details.append("分红不持续或无分配数据。")
    else:
        details.append("各期无分红数据。")
    
    # Check for decreasing share count (simple approach)
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        # For buybacks, the newest count should be less than the oldest count
        if shares[0] < shares[-1]:
            score += 1
            details.append("流通股随时间减少 (可能存在回购)。")
        else:
            details.append("流通股在可用期间内未减少。")
    else:
        details.append("无多期股份数据以评估回购。")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_activism_potential(financial_line_items: list) -> dict:
    """
    Bill Ackman often engages in activism if a company has a decent brand or moat
    but is underperforming operationally.
    
    We'll do a simplified approach:
    - Look for positive revenue trends but subpar margins
    - That may indicate 'activism upside' if operational improvements could unlock value.
    """
    if not financial_line_items:
        return {
            "score": 0,
            "details": "数据不足以分析激进主义潜力"
        }
    
    # Check revenue growth vs. operating margin
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if len(revenues) < 2 or not op_margins:
        return {
            "score": 0,
            "details": "数据不足以评估激进主义潜力 (需要多年收入和利润率数据)。"
        }
    
    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else 0
    avg_margin = sum(op_margins) / len(op_margins)
    
    score = 0
    details = []
    
    # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
    if revenue_growth > 0.15 and avg_margin < 0.10:
        score += 2
        details.append(
            f"收入增长健康 (~{revenue_growth*100:.1f}%), 但利润率偏低 (平均 {avg_margin*100:.1f}%)。"
            "激进主义行动可能释放利润率改善空间。"
        )
    else:
        details.append("无明显激进主义机会信号 (利润率已经不错或增长疲弱)。")
    
    return {"score": score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
    """
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "数据不足以进行估值"
        }
    
    # Since financial_line_items are in descending order (newest first),
    # the most recent period is the first element
    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    
    if fcf <= 0:
        return {
            "score": 0,
            "details": f"无正的自由现金流可用于估值; 自由现金流 = {fcf}",
            "intrinsic_value": None
        }
    
    # Basic DCF assumptions
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / ((1 + discount_rate) ** projection_years)
    
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    # Simple scoring
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"计算的内在价值: ~{intrinsic_value:,.2f}",
        f"市值: ~{market_cap:,.2f}",
        f"安全边际: {margin_of_safety:.2%}"
    ]
    
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety
    }


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    Includes more explicit references to brand strength, activism potential, 
    catalysts, and management changes in the system prompt.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是比尔·阿克曼AI智能体，使用他的投资原则进行决策:

            1. 寻找具有持久竞争优势 (护城河) 的优质企业，通常是知名消费或服务品牌。
            2. 优先关注持续的自由现金流和长期增长潜力。
            3. 倡导强有力的财务纪律 (合理杠杆、高效资本配置)。
            4. 估值很重要: 以安全边际为目标追求内在价值。
            5. 考虑在管理层或运营改善可以释放巨大上行空间时采取激进主义行动。
            6. 集中投资于少数高确信度标的。

            在你的推理中:
            - 强调品牌实力、护城河或独特的市场定位。
            - 审查自由现金流生成和利润率趋势作为关键信号。
            - 分析杠杆、股票回购和分红作为资本纪律指标。
            - 提供带有数据支持的估值评估 (DCF、倍数等)。
            - 识别任何激进主义或价值创造的催化剂 (如成本削减、更好的资本配置)。
            - 在讨论弱点或机会时使用自信、分析性、有时具有对抗性的语调。

            用中文输出推理。返回最终建议 (signal: bullish, neutral, or bearish)，附带0-100的置信度和详细的推理。
            """
        ),
        (
            "human",
            """根据以下分析，创建阿克曼风格的投资信号。用中文回答。

            {ticker} 的分析数据:
            {analysis_data}

            严格以有效的JSON格式返回:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="分析出错，默认为中性"
        )

    return call_llm(
        prompt=prompt, 
        pydantic_model=BillAckmanSignal, 
        agent_name=agent_id, 
        state=state,
        default_factory=create_default_bill_ackman_signal,
    )
