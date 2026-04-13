"""Chinese-to-English financial statement field mapping for A-share data."""

# 利润表 (Income Statement)
INCOME_STATEMENT_MAP = {
    "营业收入": "revenue",
    "营业成本": "cost_of_revenue",
    "营业利润": "operating_income",
    "利润总额": "ebit",
    "净利润": "net_income",
    "归属于母公司所有者的净利润": "net_income_to_parent",
    "利息费用": "interest_expense",
    "研发费用": "research_and_development",
    "基本每股收益": "earnings_per_share",
    "稀释每股收益": "diluted_eps",
    "营业税金及附加": "tax_and_surcharge",
    "销售费用": "selling_expenses",
    "管理费用": "admin_expenses",
    "财务费用": "financial_expenses",
    "投资收益": "investment_income",
    "资产减值损失": "asset_impairment_loss",
    "营业外收入": "non_operating_income",
    "营业外支出": "non_operating_expenses",
    "所得税费用": "income_tax_expense",
}

# 资产负债表 (Balance Sheet)
BALANCE_SHEET_MAP = {
    "资产总计": "total_assets",
    "负债合计": "total_liabilities",
    "流动资产合计": "current_assets",
    "流动负债合计": "current_liabilities",
    "所有者权益(或股东权益)合计": "shareholders_equity",
    "归属于母公司股东权益合计": "shareholders_equity_parent",
    "货币资金": "cash_and_equivalents",
    "存货": "inventory",
    "固定资产净额": "fixed_assets_net",
    "在建工程": "construction_in_progress",
    "无形资产": "intangible_assets",
    "应收账款": "accounts_receivable",
    "预付款项": "prepayments",
    "短期借款": "short_term_borrowings",
    "长期借款": "long_term_borrowings",
    "应付账款": "accounts_payable",
    "实收资本(或股本)": "outstanding_shares",
    "资本公积": "capital_reserve",
    "盈余公积": "surplus_reserve",
    "未分配利润": "retained_earnings",
    "商誉": "goodwill",
    "长期股权投资": "long_term_equity_investments",
    "递延所得税资产": "deferred_tax_assets",
    "递延所得税负债": "deferred_tax_liabilities",
}

# 现金流量表 (Cash Flow Statement)
CASH_FLOW_MAP = {
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "投资活动产生的现金流量净额": "investing_cash_flow",
    "筹资活动产生的现金流量净额": "financing_cash_flow",
    "购建固定资产、无形资产和其他长期资产所支付的现金": "capital_expenditure",
    "收回投资所收到的现金": "investment_proceeds",
    "取得投资收益收到的现金": "investment_income_received",
    "分配股利、利润或偿付利息所支付的现金": "dividends_and_other_cash_distributions",
    "吸收投资收到的现金": "issuance_or_purchase_of_equity_shares",
    "现金的期末余额": "cash_end",
    "现金的期初余额": "cash_begin",
}

# 合并所有映射: 英文 -> 中文
ENGLISH_TO_CHINESE = {}
for cn_map in [INCOME_STATEMENT_MAP, BALANCE_SHEET_MAP, CASH_FLOW_MAP]:
    for cn, en in cn_map.items():
        ENGLISH_TO_CHINESE[en] = cn

# 按报表分组判断需要拉取哪些报表
INCOME_ITEMS = {
    "revenue", "net_income", "operating_income", "gross_profit",
    "interest_expense", "research_and_development", "earnings_per_share",
    "ebit", "ebitda", "free_cash_flow",
}
BALANCE_ITEMS = {
    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
    "shareholders_equity", "cash_and_equivalents", "total_debt", "outstanding_shares",
    "working_capital", "book_value_per_share",
}
CASHFLOW_ITEMS = {
    "capital_expenditure", "free_cash_flow", "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares", "operating_cash_flow",
}
