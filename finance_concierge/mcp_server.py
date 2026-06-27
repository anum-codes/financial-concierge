from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

"""
mcp_server.py - MCP server providing Google Sheets integration tools
for the Finance Concierge Agent.

Tools exposed:
  - append_transaction(date, description, amount, category)
  - get_transactions(start_date, end_date)
  - update_budget(category, amount)
  - get_budget_summary()

Environment variables (loaded from .env):
  GOOGLE_OAUTH_CREDENTIALS_FILE   - path to the OAuth credentials JSON key file
  GOOGLE_SHEET_ID                 - the target Google Spreadsheet ID

Run with:
  python -m finance_concierge.mcp_server
  # or directly:
  python finance_concierge/mcp_server.py
"""

import json
import os
from datetime import datetime
from typing import Any

import gspread
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Environment & Google Sheets bootstrap
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

OAUTH_CREDENTIALS_FILE: str = os.environ.get("GOOGLE_OAUTH_CREDENTIALS_FILE", "")
SHEET_ID: str = os.environ.get("GOOGLE_SHEET_ID", "")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Sheet tab names (must exist inside the spreadsheet)
TRANSACTIONS_SHEET = "Transactions"
BUDGETS_SHEET = "Budgets"

# Expected column order in the Transactions sheet
# Row 1 is the header: Date | Description | Amount | Category
TRANSACTION_HEADERS = ["Date", "Description", "Amount", "Category"]

# Expected column order in the Budgets sheet
# Row 1 is the header: Category | Budget | Spent
BUDGET_HEADERS = ["Category", "Budget", "Spent"]


def _get_client() -> gspread.Client:
    """Return an authorised gspread client."""
    if not OAUTH_CREDENTIALS_FILE:
        raise EnvironmentError(
            "GOOGLE_OAUTH_CREDENTIALS_FILE is not set in the environment."
        )
    return gspread.oauth(
        scopes=SCOPES,
        credentials_filename=OAUTH_CREDENTIALS_FILE,
        authorized_user_filename="token.json",
    )


def _open_sheet(worksheet_name: str) -> gspread.Worksheet:
    """Open a worksheet by name from the configured spreadsheet."""
    if not SHEET_ID:
        raise EnvironmentError("GOOGLE_SHEET_ID is not set in the environment.")
    client = _get_client()
    spreadsheet = client.open_by_key(SHEET_ID)
    return spreadsheet.worksheet(worksheet_name)


def _ok(data: Any) -> list[TextContent]:
    """Wrap a result as a successful MCP TextContent response."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _err(message: str) -> list[TextContent]:
    """Wrap an error message as an MCP TextContent response."""
    return [TextContent(type="text", text=json.dumps({"error": message}, indent=2))]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def append_transaction(
    date_str: str,
    description: str,
    amount: float,
    category: str,
) -> list[TextContent]:
    """Appends a new transaction row to the Transactions sheet."""
    try:
        ws = _open_sheet(TRANSACTIONS_SHEET)

        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return _err(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")

        row = [str(parsed_date), description.strip(), float(amount), category.strip()]
        ws.append_row(row, value_input_option="USER_ENTERED")

        all_rows = ws.get_all_values()
        new_row_index = len(all_rows)  # 1-indexed (includes header)

        return _ok(
            {
                "status": "success",
                "message": "Transaction appended successfully.",
                "row_index": new_row_index,
                "data": {
                    "date": str(parsed_date),
                    "description": description.strip(),
                    "amount": float(amount),
                    "category": category.strip(),
                },
            }
        )
    except EnvironmentError as exc:
        return _err(str(exc))
    except gspread.exceptions.SpreadsheetNotFound:
        return _err(f"Spreadsheet with ID '{SHEET_ID}' not found.")
    except gspread.exceptions.WorksheetNotFound:
        return _err(f"Worksheet '{TRANSACTIONS_SHEET}' not found in spreadsheet.")
    except Exception as exc:
        return _err(f"Unexpected error: {exc}")


def get_transactions(
    start_date: str,
    end_date: str,
) -> list[TextContent]:
    """Reads all transactions between start_date and end_date (inclusive)."""
    try:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            return _err(f"Invalid date format: {exc}. Expected YYYY-MM-DD.")

        if start > end:
            return _err("start_date must be on or before end_date.")

        ws = _open_sheet(TRANSACTIONS_SHEET)
        rows = ws.get_all_records(expected_headers=TRANSACTION_HEADERS)

        filtered: list[dict] = []
        for row in rows:
            try:
                row_date = datetime.strptime(str(row["Date"]), "%Y-%m-%d").date()
            except ValueError:
                continue  # skip malformed rows
            if start <= row_date <= end:
                filtered.append(
                    {
                        "date": str(row_date),
                        "description": row.get("Description", ""),
                        "amount": float(row.get("Amount", 0)),
                        "category": row.get("Category", ""),
                    }
                )

        return _ok(
            {
                "status": "success",
                "start_date": str(start),
                "end_date": str(end),
                "count": len(filtered),
                "transactions": filtered,
            }
        )
    except EnvironmentError as exc:
        return _err(str(exc))
    except gspread.exceptions.SpreadsheetNotFound:
        return _err(f"Spreadsheet with ID '{SHEET_ID}' not found.")
    except gspread.exceptions.WorksheetNotFound:
        return _err(f"Worksheet '{TRANSACTIONS_SHEET}' not found in spreadsheet.")
    except Exception as exc:
        return _err(f"Unexpected error: {exc}")


def update_budget(category: str, amount: float) -> list[TextContent]:
    """Updates the Budget column for a given category in the Budgets sheet."""
    try:
        ws = _open_sheet(BUDGETS_SHEET)
        rows = ws.get_all_values()

        if not rows:
            return _err(f"Worksheet '{BUDGETS_SHEET}' is empty.")

        header = rows[0]
        try:
            cat_col = header.index("Category") + 1   # 1-indexed
            bgt_col = header.index("Budget") + 1
        except ValueError:
            return _err(
                f"Worksheet '{BUDGETS_SHEET}' is missing expected headers: "
                "Category, Budget."
            )

        category = category.strip().title()
        old_value: float | None = None

        for i, row in enumerate(rows[1:], start=2):  # row 1 is header
            if row[cat_col - 1].strip().title() == category:
                old_value = float(row[bgt_col - 1]) if row[bgt_col - 1] else 0.0
                ws.update_cell(i, bgt_col, float(amount))
                return _ok(
                    {
                        "status": "success",
                        "message": f"Budget for '{category}' updated.",
                        "category": category,
                        "old_budget": old_value,
                        "new_budget": float(amount),
                    }
                )

        # Category not found → append a new row (Spent defaults to 0)
        ws.append_row([category, float(amount), 0], value_input_option="USER_ENTERED")
        return _ok(
            {
                "status": "success",
                "message": f"Category '{category}' not found; new row appended.",
                "category": category,
                "old_budget": None,
                "new_budget": float(amount),
            }
        )
    except EnvironmentError as exc:
        return _err(str(exc))
    except gspread.exceptions.SpreadsheetNotFound:
        return _err(f"Spreadsheet with ID '{SHEET_ID}' not found.")
    except gspread.exceptions.WorksheetNotFound:
        return _err(f"Worksheet '{BUDGETS_SHEET}' not found in spreadsheet.")
    except Exception as exc:
        return _err(f"Unexpected error: {exc}")


def get_budget_summary() -> list[TextContent]:
    """Returns all rows from the Budgets sheet with computed metrics."""
    try:
        ws = _open_sheet(BUDGETS_SHEET)
        rows = ws.get_all_records(expected_headers=BUDGET_HEADERS)

        summary: list[dict] = []
        total_budget = 0.0
        total_spent = 0.0

        for row in rows:
            budget = float(row.get("Budget", 0))
            spent = float(row.get("Spent", 0))
            remaining = round(budget - spent, 2)
            percentage_used = round(spent / budget * 100, 2) if budget else 0.0

            if spent > budget:
                severity = "over_budget"
            elif spent >= budget * 0.80:
                severity = "warning"
            else:
                severity = "on_track"

            summary.append(
                {
                    "category": row.get("Category", ""),
                    "budget": budget,
                    "spent": spent,
                    "remaining": remaining,
                    "percentage_used": percentage_used,
                    "severity": severity,
                }
            )
            total_budget += budget
            total_spent += spent

        total_remaining = round(total_budget - total_spent, 2)
        savings_rate = (
            round(total_remaining / total_budget * 100, 2) if total_budget else 0.0
        )

        return _ok(
            {
                "status": "success",
                "categories": summary,
                "totals": {
                    "total_budget": round(total_budget, 2),
                    "total_spent": round(total_spent, 2),
                    "total_remaining": total_remaining,
                    "savings_rate": savings_rate,
                },
            }
        )
    except EnvironmentError as exc:
        return _err(str(exc))
    except gspread.exceptions.SpreadsheetNotFound:
        return _err(f"Spreadsheet with ID '{SHEET_ID}' not found.")
    except gspread.exceptions.WorksheetNotFound:
        return _err(f"Worksheet '{BUDGETS_SHEET}' not found in spreadsheet.")
    except Exception as exc:
        return _err(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

app = Server("finance-concierge-sheets")

TOOLS: list[Tool] = [
    Tool(
        name="append_transaction",
        description=(
            "Append a new financial transaction row to the Google Sheet. "
            "Requires date (YYYY-MM-DD), description, amount (float), and "
            "category (Food | Transport | Shopping | Bills | Health | "
            "Entertainment | Other)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Transaction date in YYYY-MM-DD format.",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the purchase or expense.",
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount as a positive float.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Budget category. One of: Food, Transport, Shopping, "
                        "Bills, Health, Entertainment, Other."
                    ),
                },
            },
            "required": ["date", "description", "amount", "category"],
        },
    ),
    Tool(
        name="get_transactions",
        description=(
            "Read all transactions from the Google Sheet that fall between "
            "start_date and end_date (inclusive, YYYY-MM-DD format)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Filter start date in YYYY-MM-DD format.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Filter end date in YYYY-MM-DD format.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    ),
    Tool(
        name="update_budget",
        description=(
            "Update the monthly budget amount for a specific category in the "
            "Google Sheet. Creates the row if the category does not yet exist."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Budget category to update (e.g. 'Shopping').",
                },
                "amount": {
                    "type": "number",
                    "description": "New monthly budget amount.",
                },
            },
            "required": ["category", "amount"],
        },
    ),
    Tool(
        name="get_budget_summary",
        description=(
            "Return all budget categories from the Google Sheet with their "
            "spent amounts, remaining balances, percentage used, severity "
            "status, and aggregate totals."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "append_transaction":
        return append_transaction(
            date_str=arguments["date"],
            description=arguments["description"],
            amount=float(arguments["amount"]),
            category=arguments["category"],
        )
    if name == "get_transactions":
        return get_transactions(
            start_date=arguments["start_date"],
            end_date=arguments["end_date"],
        )
    if name == "update_budget":
        return update_budget(
            category=arguments["category"],
            amount=float(arguments["amount"]),
        )
    if name == "get_budget_summary":
        return get_budget_summary()

    return _err(f"Unknown tool: '{name}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
