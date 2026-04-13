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


class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(state: AgentState, agent_id: str = "cathie_wood_agent"):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "收集财务报表项目")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "分析颠覆性潜力")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "分析创新驱动增长")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "计算估值与高增长情景")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        # Combine partial scores or signals
        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # Adjust weighting as desired

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "disruptive_analysis": disruptive_analysis, "innovation_analysis": innovation_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status(agent_id, ticker, "生成凯瑟琳·伍德分析")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        cw_analysis[ticker] = {"signal": cw_output.signal, "confidence": cw_output.confidence, "reasoning": cw_output.reasoning}

        progress.update_status(agent_id, ticker, "完成", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "凯瑟琳·伍德智能体")

    state["data"]["analyst_signals"][agent_id] = cw_analysis

    progress.update_status(agent_id, None, "完成")

    return {"messages": [message], "data": state["data"]}


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "数据不足以分析颠覆性潜力"}

    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"收入增长加速：{(growth_rates[0]*100):.1f}% 对比 {(growth_rates[-1]*100):.1f}%")

        # Check absolute growth rate (most recent growth rate is at index 0)
        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"收入增长异常强劲：{(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"收入增长强劲：{(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"收入增长温和：{(latest_growth*100):.1f}%")
    else:
        details.append("收入数据不足以进行增长分析")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"毛利率扩张：+{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"毛利率略有改善：+{(margin_trend*100):.1f}%")

        # Check absolute margin level (most recent margin is at index 0)
        if gross_margins[0] > 0.50:  # High margin business
            score += 2
            details.append(f"高毛利率：{(gross_margins[0]*100):.1f}%")
    else:
        details.append("毛利率数据不足")

    # 3. Operating Leverage Analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append("正向经营杠杆：收入增长快于支出增长")
    else:
        details.append("数据不足以进行经营杠杆分析")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"研发投入高：占收入的{(rd_intensity*100):.1f}%")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"研发投入中等：占收入的{(rd_intensity*100):.1f}%")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"有一定研发投入：占收入的{(rd_intensity*100):.1f}%")
    else:
        details.append("无研发数据")

    # Normalize score to be out of 5
    max_possible_score = 12  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "数据不足以分析创新驱动增长"}

    # 1. R&D Investment Trends
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"研发投入增长强劲：+{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"研发投入增长中等：+{(rd_growth*100):.1f}%")

        # Check R&D intensity trend (corrected for reverse chronological order)
        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"研发强度上升：{(rd_intensity_end*100):.1f}% 对比 {(rd_intensity_start*100):.1f}%")
    else:
        details.append("研发数据不足以进行趋势分析")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("自由现金流增长强劲且稳定，创新资金充裕")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("自由现金流持续为正，创新资金良好")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("自由现金流较为稳定，创新资金尚可")
    else:
        details.append("自由现金流数据不足")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]

        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"营业利润率强劲且改善中：{(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"营业利润率健康：{(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("运营效率改善中")
    else:
        details.append("营业利润率数据不足")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("对增长基础设施投入强劲")
        elif capex_intensity > 0.05:
            score += 1
            details.append("对增长基础设施投入中等")
    else:
        details.append("资本支出数据不足")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("注重再投资而非分红")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("在再投资与分红之间较为平衡")
    else:
        details.append("股息数据不足")

    # Normalize score to be out of 5
    max_possible_score = 15  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "数据不足以进行估值"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"无正自由现金流用于估值；FCF = {fcf}", "intrinsic_value": None}

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [f"计算内在价值：~{intrinsic_value:,.2f}", f"市值：~{market_cap:,.2f}", f"安全边际：{margin_of_safety:.2%}"]

    return {"score": score, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是凯瑟琳·伍德AI智能体，使用她的投资原则进行决策:

            1. 寻找利用颠覆性创新的公司。
            2. 重视指数级增长潜力和巨大的总可寻址市场。
            3. 关注科技、医疗或其他面向未来的行业。
            4. 考虑多年时间跨度内的潜在突破。
            5. 接受较高波动性以追求高回报。
            6. 评估管理层的愿景和研发投资能力。

            规则:
            - 识别颠覆性或突破性技术。
            - 评估多年收入增长的强劲潜力。
            - 检查公司是否能在大型市场中有效扩展。
            - 使用偏向增长的估值方法。
            - 提供数据驱动的建议 (bullish, bearish, 或 neutral)。

            在提供推理时，要详尽具体:
            1. 识别公司正在利用的具体颠覆性技术/创新
            2. 突出表明指数级潜力的增长指标 (收入加速、TAM扩张)
            3. 讨论5年以上的长期愿景和变革潜力
            4. 解释公司可能如何颠覆传统行业或创造新市场
            5. 评估研发投资和可能推动未来增长的创新管线
            6. 使用凯瑟琳·伍德乐观、面向未来和充满信念的语调

            例如，如果看涨: "该公司的AI驱动平台正在变革5000亿美元的医疗分析市场，平台采用率从40%加速到65%同比增长。其占收入22%的研发投资正在创造技术护城河，使其能够占据这一扩张市场的重要份额。当前估值并未反映我们预期的指数级增长轨迹..."
            例如，如果看跌: "虽然身处基因组学领域，该公司缺乏真正的颠覆性技术，只是对现有技术进行渐进式改进。仅占收入8%的研发支出表明对突破性创新的投资不足。收入增长从45%放缓至20%同比增长，缺乏我们在变革性公司中寻找的指数级采用曲线证据..."
            """,
            ),
            (
                "human",
                """根据以下分析，创建凯瑟琳·伍德风格的投资信号。用中文回答。

            {ticker} 的分析数据:
            {analysis_data}

            按以下JSON格式返回交易信号:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="分析出错，默认中性")

    return call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_cathie_wood_signal,
    )


# source: https://ark-invest.com
