---
name: finance-analyzer
description: Analyzes spending patterns and provides actionable financial advice.
---

# Finance Analyzer Skill
Analyzes spending patterns and provides actionable financial advice.

## When to use
When user asks "how am I doing", "analyze my spending", "give me advice",
or "what should I cut back on"

## Actions
- Calculate savings rate from total budget vs total spent
- Identify top overspending categories
- Suggest one specific actionable cut per overspending category
- Return advice in structured JSON format

## Output Format
{"skill": "finance-analyzer", "savings_rate": float, 
 "advice": [{"category": str, "suggestion": str}]}
