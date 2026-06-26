# Finance Concierge Agent

## Project Overview
A multi-agent personal finance manager built with ADK 2.0.

## Architecture
- Orchestrator Agent routes user intent to specialized subagents
- TransactionClassifier Agent categorizes expenses
- BudgetAnalyst Agent tracks spending vs budget
- AlertAgent fires warnings on unusual spending
- ReportGenerator Agent produces weekly summaries

## Technical Rules
- Use ADK 2.0 graph workflow API only (NOT 1.x SequentialAgent style)
- All secrets and API keys must go in .env file — never hardcoded
- MCP server connects to Google Sheets for data persistence
- PII redaction must happen before any LLM invocation
- All agents must return structured JSON output

## Stack
- Python, ADK 2.0, agents-cli, MCP, Google Sheets API