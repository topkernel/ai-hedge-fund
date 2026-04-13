from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import math
from src.utils.api_key import get_api_key_from_state


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState, agent_id: str = "ben_graham_agent"):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "收集财务报表项目")
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # Perform sub-analyses
        progress.update_status(agent_id, ticker, "分析盈利稳定性")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "分析财务实力")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status(agent_id, ticker, "分析格雷厄姆估值")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        # Aggregate scoring
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status(agent_id, ticker, "生成本杰明·格雷厄姆分析")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status(agent_id, ticker, "完成", analysis=graham_output.reasoning)

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name=agent_id)

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "本杰明·格雷厄姆智能体")

    # Store signals in the overall state
    state["data"]["analyst_signals"][agent_id] = graham_analysis

    progress.update_status(agent_id, None, "完成")

    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "数据不足以进行盈利稳定性分析"}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append("多年每股收益数据不足。")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("所有可用期间每股收益均为正。")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("大多数期间每股收益为正。")
    else:
        details.append("多个期间每股收益为负。")

    # 2. EPS growth from earliest to latest
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("每股收益从最早到最新期间有所增长。")
    else:
        details.append("每股收益从最早到最新期间未增长。")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "无数据用于财务实力分析"}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"流动比率 = {current_ratio:.2f} (>=2.0: 稳健)。")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"流动比率 = {current_ratio:.2f} (中等偏强)。")
        else:
            details.append(f"流动比率 = {current_ratio:.2f} (<1.5: 流动性较弱)。")
    else:
        details.append("无法计算流动比率 (缺少流动负债或为零)。")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"负债率 = {debt_ratio:.2f}, 低于0.50 (保守)。")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"负债率 = {debt_ratio:.2f}, 偏高但尚可接受。")
        else:
            details.append(f"负债率 = {debt_ratio:.2f}, 按格雷厄姆标准相当高。")
    else:
        details.append("无法计算负债率 (缺少总资产数据)。")

    # 3. Dividend track record
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("公司在大多数报告年份支付了分红。")
            else:
                details.append("公司有一些分红支付，但不是大多数年份。")
        else:
            details.append("公司在这些期间未支付分红。")
    else:
        details.append("无分红数据可用于评估派息一致性。")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "数据不足以进行估值"}

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"净流动资产价值 = {net_current_asset_value:,.2f}")
        details.append(f"每股净流动资产价值 = {net_current_asset_value_per_share:,.2f}")
        details.append(f"每股价格 = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: 净流动资产价值 > 市值 (经典格雷厄姆深度价值)。")
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("每股净流动资产价值 >= 每股价格的2/3 (中等净流动资产折价)。")
    else:
        details.append("净流动资产价值未超过市值或数据不足以进行净流动资产分析。")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"格雷厄姆数 = {graham_number:.2f}")
    else:
        details.append("无法计算格雷厄姆数 (每股收益或每股账面价值缺失/<=0)。")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"安全边际 (格雷厄姆数) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("价格远低于格雷厄姆数 (>=50%安全边际)。")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("相对格雷厄姆数有一定安全边际。")
            else:
                details.append("价格接近或高于格雷厄姆数，安全边际低。")
        else:
            details.append("当前价格为零或无效；无法计算安全边际。")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是本杰明·格雷厄姆AI智能体，使用他的投资原则进行决策:
            1. 坚持安全边际，以低于内在价值的价格买入 (如使用格雷厄姆数、净流动资产)。
            2. 重视公司的财务实力 (低杠杆、充足的流动资产)。
            3. 偏好多年稳定盈利。
            4. 考虑分红记录作为额外安全保障。
            5. 避免投机性或高增长假设；关注经过验证的指标。

            在提供推理时，要详尽具体:
            1. 解释对你决策影响最大的关键估值指标 (格雷厄姆数、净流动资产价值、市盈率等)
            2. 突出具体的财务实力指标 (流动比率、债务水平等)
            3. 引用一段时间内盈利的稳定性或不稳定性
            4. 提供带有精确数字的量化证据
            5. 将当前指标与格雷厄姆的具体门槛进行比较 (例如 "流动比率2.5超过了格雷厄姆最低标准2.0")
            6. 使用本杰明·格雷厄姆保守、分析性的语调和风格

            例如，如果看涨: "该股票以净流动资产价值35%的折价交易，提供了充足的安全边际。流动比率2.5和债务股本比0.3表明财务状况强劲..."
            例如，如果看跌: "尽管盈利持续，当前价格50美元超过了我们计算的格雷厄姆数35美元，没有提供安全边际。此外，仅1.2的流动比率低于格雷厄姆偏好的2.0门槛..."

            用中文输出推理。返回理性的建议: bullish, bearish, 或 neutral，附带置信度 (0-100) 和详尽的推理。
            """,
            ),
            (
                "human",
                """根据以下分析，创建格雷厄姆风格的投资信号。用中文回答。

            {ticker} 的分析数据:
            {analysis_data}

            严格按以下JSON格式返回:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="分析生成出错；默认中性。")

    return call_llm(
        prompt=prompt,
        pydantic_model=BenGrahamSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ben_graham_signal,
    )
