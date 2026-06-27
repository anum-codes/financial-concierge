from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import re
import json
import datetime
import sys
import os
from typing import Any
from google.adk.agents.llm_agent import Agent
from google.adk.workflow import Workflow, START, node
from pydantic import BaseModel, Field
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

# 1. Structured Output Schemas
class OrchestratorOutput(BaseModel):
    intent: str = Field(description="The detected user intent. Must be exactly one of: 'TransactionClassifier', 'BudgetAnalyst', 'AlertAgent', or 'ReportGenerator'.")

class StubOutput(BaseModel):
    agent: str = Field(description="The name of the subagent.")
    status: str = Field(description="The status of the execution.")

class TransactionClassificationResult(BaseModel):
    amount: float = Field(description="The transaction amount.")
    description: str = Field(description="The description of the purchase/expense.")
    category: str = Field(description="The category: Food, Transport, Shopping, Bills, Health, Entertainment, or Other.")
    date: str = Field(description="The date of the transaction in YYYY-MM-DD format.")

class TransactionClassifierOutput(BaseModel):
    agent: str = Field(default="TransactionClassifier", description="The name of the agent.")
    amount: float = Field(description="The transaction amount.")
    description: str = Field(description="The description of the expense.")
    category: str = Field(description="The category of the expense.")
    date: str = Field(description="The date of the transaction.")
    status: str = Field(default="success", description="The status of the classification.")

class BudgetAnalystCategoryResult(BaseModel):
    category: str = Field(description="The budget category the user is asking about. Must be exactly one of: Food, Transport, Shopping, Bills, Health, Entertainment, Other.")

class BudgetAnalystOutput(BaseModel):
    agent: str = Field(default="BudgetAnalyst", description="The name of the agent.")
    category: str = Field(description="The queried budget category.")
    budget: float = Field(description="The monthly budget for the category.")
    spent: float = Field(description="Amount spent so far this month.")
    remaining: float = Field(description="Budget remaining (budget - spent). Negative when over budget.")
    status: str = Field(description="'over_budget' if spent > budget, otherwise 'under_budget'.")
    percentage_used: float = Field(description="Percentage of budget used (spent / budget * 100).")

class AlertItem(BaseModel):
    category: str = Field(description="The budget category.")
    severity: str = Field(description="'over_budget', 'warning', or 'on_track'.")
    spent: float = Field(description="Amount spent so far this month.")
    budget: float = Field(description="The monthly budget for this category.")
    percentage_used: float = Field(description="Percentage of budget used.")

class AlertAgentOutput(BaseModel):
    agent: str = Field(default="AlertAgent", description="The name of the agent.")
    alerts: list[AlertItem] = Field(description="List of categories that are over_budget or warning.")
    total_alerts: int = Field(description="Total number of alerts (over_budget + warning categories).")

class ReportGeneratorOutput(BaseModel):
    agent: str = Field(default="ReportGenerator", description="The name of the agent.")
    period: str = Field(default="weekly", description="The report period.")
    total_budget: float = Field(description="Sum of budgets across all categories.")
    total_spent: float = Field(description="Sum of spending across all categories.")
    total_remaining: float = Field(description="total_budget - total_spent.")
    savings_rate: float = Field(description="Percentage of total budget not spent (total_remaining / total_budget * 100).")
    over_budget_categories: list[str] = Field(description="Categories where spent > budget, sorted by overspend descending.")
    under_budget_categories: list[str] = Field(description="Top 3 categories furthest under budget by absolute remaining amount.")
    status: str = Field(default="success", description="Status of the report generation.")

# 2. PII Redaction Node
@node
def redact_pii_node(ctx: Any, node_input: str) -> str:
    """Redacts PII from the user input query and stores the safe version in state."""
    # Redact email addresses
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', node_input)

    # Redact credit card numbers FIRST (4 groups of 4 digits separated by spaces or dashes)
    # Must run before phone so the broad phone regex doesn't consume 3 of the 4 groups first.
    cc_pattern = r'\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b'
    text = re.sub(cc_pattern, '[REDACTED_CARD]', text)

    # Redact phone numbers – handles US (3+3+4), UK (3+4+4), international (+44 20 …)
    # Requires at least one separator between digit groups to avoid eating bare numbers.
    phone_pattern = r'(?:\+?\d{1,3}[\s.-])?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}'
    text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)

    ctx.state['query'] = text
    return text


# 3. Intent Classification Orchestrator Agent
orchestrator_agent = Agent(
    name="OrchestratorAgent",
    model="gemini-3.5-flash",
    instruction="""
    You are the Orchestrator Agent for a personal finance concierge.
    Analyze the user's input query and classify their intent into one of the following exact strings:
    
    - 'TransactionClassifier': for queries about adding, logging, spending, buying, or paying (e.g. keywords: "add", "spent", "bought", "paid").
    - 'BudgetAnalyst': for queries about budgets, limits, remaining budget, or overspending (e.g. keywords: "budget", "how much left", "overspend").
    - 'AlertAgent': for queries about warnings, alerts, or unusual transactions/spending (e.g. keywords: "alert", "warning", "unusual").
    - 'ReportGenerator': for queries about summaries, reports, or weekly updates (e.g. keywords: "report", "summary", "weekly").
    
    Respond strictly with JSON matching the OrchestratorOutput schema.
    """,
    output_schema=OrchestratorOutput
)

# 4. Router Node
@node
def router(ctx: Any, node_input: OrchestratorOutput) -> OrchestratorOutput:
    """Sets the routing destination value on the workflow context."""
    ctx.route = node_input.intent
    return node_input

# 5. Subagent Stubs & Implementations
def get_classification_instruction(ctx: Any) -> str:
    today = datetime.date.today().isoformat()
    return f"""
    You are a Transaction Classifier Agent.
    Your task is to analyze the user's finance transaction query and extract the following:
    - amount: a float number representing the transaction amount.
    - description: a short string describing the expense.
    - date: the date of the transaction in YYYY-MM-DD format. If the date or year is not explicitly mentioned, use today's date: {today}.
    - category: categorize the transaction into exactly one of: Food, Transport, Shopping, Bills, Health, Entertainment, Other.

    Provide your response strictly in structured JSON matching the output schema.
    """

transaction_classifier_agent = Agent(
    name="TransactionClassifierAgent",
    model="gemini-3.5-flash",
    instruction=get_classification_instruction,
    output_schema=TransactionClassificationResult
)

@node(rerun_on_resume=True)
async def transaction_classifier(ctx: Any) -> TransactionClassifierOutput:
    query = ctx.state.get('query', '')
    result = await ctx.run_node(transaction_classifier_agent, node_input=query)
    # ctx.run_node may return a plain dict or a Pydantic model depending on the
    # ADK version; handle both to avoid AttributeError.
    if isinstance(result, dict):
        amount      = result["amount"]
        description = result["description"]
        category    = result["category"]
        date        = result["date"]
    else:
        amount      = result.amount
        description = result.description
        category    = result.category
        date        = result.date
    return TransactionClassifierOutput(
        agent="TransactionClassifier",
        amount=amount,
        description=description,
        category=category,
        date=date,
        status="success"
    )

# ── BudgetAnalyst ──────────────────────────────────────────────────────────

# Fallback values if MCP connection fails
FALLBACK_MONTHLY_BUDGETS: dict[str, float] = {
    "Food": 200, "Transport": 100, "Shopping": 150,
    "Bills": 300, "Health": 100, "Entertainment": 80, "Other": 50,
}

FALLBACK_CURRENT_SPENDING: dict[str, float] = {
    "Food": 120, "Transport": 45, "Shopping": 200,
    "Bills": 300, "Health": 20, "Entertainment": 60, "Other": 10,
}

# MCP Client initialization
mcp_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[mcp_server_path],
            env=os.environ.copy()
        ),
        timeout=10.0
    )
)

async def get_budgets_and_spending() -> tuple[dict[str, float], dict[str, float]]:
    """Fetches real budget and spending data from Google Sheets via MCP, falling back to hardcoded values if it fails."""
    try:
        res = await toolset._execute_with_session(
            lambda session: session.call_tool("get_budget_summary", arguments={}),
            "Failed to call get_budget_summary"
        )
        if not res or not res.content:
            raise ValueError("Empty response from get_budget_summary")
        data = json.loads(res.content[0].text)
        
        # Check if get_budget_summary returned success or is structured correctly
        if isinstance(data, dict) and data.get("status") == "success":
            monthly_budgets = {}
            current_spending = {}
            for cat_info in data.get("categories", []):
                cat = cat_info.get("category")
                if cat:
                    monthly_budgets[cat] = float(cat_info.get("budget", 0))
                    current_spending[cat] = float(cat_info.get("spent", 0))
            if monthly_budgets:
                return monthly_budgets, current_spending
        raise ValueError(f"Invalid data or status in get_budget_summary: {data}")
    except Exception as e:
        print(f"Warning: Failed to load data from MCP server: {e}. Falling back to hardcoded values.")
        return FALLBACK_MONTHLY_BUDGETS, FALLBACK_CURRENT_SPENDING

budget_analyst_agent = Agent(
    name="BudgetAnalystAgent",
    model="gemini-3.5-flash",
    instruction="""
    You are a budget category extractor.
    Given the user's finance query, identify which single budget category they are asking about.
    The category must be exactly one of: Food, Transport, Shopping, Bills, Health, Entertainment, Other.
    Respond strictly in JSON matching the output schema.
    """,
    output_schema=BudgetAnalystCategoryResult,
)

@node(rerun_on_resume=True)
async def budget_analyst(ctx: Any) -> BudgetAnalystOutput:
    query = ctx.state.get('query', '')
    result = await ctx.run_node(budget_analyst_agent, node_input=query)

    # Handle both dict and Pydantic model responses
    category = result["category"] if isinstance(result, dict) else result.category

    # Normalise to a known key; fall back to "Other"
    category = category.strip().title()
    
    monthly_budgets, current_spending = await get_budgets_and_spending()
    if category not in monthly_budgets:
        category = "Other"

    budget  = monthly_budgets.get(category, 0.0)
    spent   = current_spending.get(category, 0.0)
    remaining       = budget - spent
    percentage_used = round(spent / budget * 100, 2) if budget else 0.0
    status  = "over_budget" if spent > budget else "under_budget"

    output = BudgetAnalystOutput(
        agent="BudgetAnalyst",
        category=category,
        budget=budget,
        spent=spent,
        remaining=remaining,
        status=status,
        percentage_used=percentage_used,
    )
    # Explicitly print the final structured result so the CLI displays it,
    # matching the behaviour of TransactionClassifierAgent.
    print(f"[BudgetAnalyst]: {json.loads(output.model_dump_json())}")
    return output

# ── AlertAgent ────────────────────────────────────────────────────────────

@node
async def alert_agent(ctx: Any) -> AlertAgentOutput:
    """Pure-Python alert scanner: no LLM needed.
    Classifies every category and returns only over_budget / warning items.
    """
    alerts: list[AlertItem] = []
    
    monthly_budgets, current_spending = await get_budgets_and_spending()

    for category, budget in monthly_budgets.items():
        spent = current_spending.get(category, 0.0)
        percentage_used = round(spent / budget * 100, 2) if budget else 0.0

        if spent > budget:
            severity = "over_budget"
        elif spent >= budget * 0.80:
            severity = "warning"
        else:
            severity = "on_track"

        if severity in ("over_budget", "warning"):
            alerts.append(AlertItem(
                category=category,
                severity=severity,
                spent=spent,
                budget=budget,
                percentage_used=percentage_used,
            ))

    # Sort: over_budget first, then warning; within each group by percentage desc
    alerts.sort(key=lambda a: (0 if a.severity == "over_budget" else 1, -a.percentage_used))

    output = AlertAgentOutput(
        agent="AlertAgent",
        alerts=alerts,
        total_alerts=len(alerts),
    )
    print(f"[AlertAgent]: {json.loads(output.model_dump_json())}")
    return output

# ── ReportGenerator ───────────────────────────────────────────────────

@node
async def report_generator(ctx: Any) -> ReportGeneratorOutput:
    """Pure-Python weekly finance summary report. No LLM needed."""
    monthly_budgets, current_spending = await get_budgets_and_spending()

    total_budget    = sum(monthly_budgets.values())
    total_spent     = sum(current_spending.values())
    total_remaining = round(total_budget - total_spent, 2)
    savings_rate    = round(total_remaining / total_budget * 100, 2) if total_budget else 0.0

    # Categories where spent > budget, sorted by overspend (descending)
    over_budget_categories = sorted(
        [cat for cat, bgt in monthly_budgets.items()
         if current_spending.get(cat, 0) > bgt],
        key=lambda cat: current_spending.get(cat, 0) - monthly_budgets[cat],
        reverse=True,
    )

    # Top-3 categories furthest under budget (highest remaining), excluding over-budget ones
    under_budget_categories = sorted(
        [cat for cat, bgt in monthly_budgets.items()
         if current_spending.get(cat, 0) <= bgt],
        key=lambda cat: monthly_budgets[cat] - current_spending.get(cat, 0),
        reverse=True,
    )[:3]

    output = ReportGeneratorOutput(
        agent="ReportGenerator",
        period="weekly",
        total_budget=round(total_budget, 2),
        total_spent=round(total_spent, 2),
        total_remaining=total_remaining,
        savings_rate=savings_rate,
        over_budget_categories=over_budget_categories,
        under_budget_categories=under_budget_categories,
        status="success",
    )
    print(f"[ReportGenerator]: {json.loads(output.model_dump_json())}")
    return output

# 6. Workflow Orchestrator Graph Definition
root_agent = Workflow(
    name="finance_concierge_workflow",
    edges=[
        (START, redact_pii_node),
        (redact_pii_node, orchestrator_agent),
        (orchestrator_agent, router),
        (router, {
            "TransactionClassifier": transaction_classifier,
            "BudgetAnalyst": budget_analyst,
            "AlertAgent": alert_agent,
            "ReportGenerator": report_generator
        })
    ]
)
