import unittest
import json
from unittest.mock import MagicMock
from finance_concierge.agent import redact_pii_node, router, transaction_classifier, budget_analyst, alert_agent, report_generator, OrchestratorOutput, StubOutput

class TestFinanceConcierge(unittest.IsolatedAsyncioTestCase):
    def test_pii_redaction_email(self):
        ctx = MagicMock()
        ctx.state = {}
        input_text = "My email is test.user+123@example.co.uk. Please email me."
        redacted = redact_pii_node._func(ctx, input_text)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertNotIn("test.user", redacted)
        self.assertEqual(ctx.state['query'], redacted)

    def test_pii_redaction_phone(self):
        ctx = MagicMock()
        ctx.state = {}
        input_text = "Call me at +1 (555) 123-4567 or 555-456-7890."
        redacted = redact_pii_node._func(ctx, input_text)
        self.assertEqual(redacted.count("[REDACTED_PHONE]"), 2)
        self.assertNotIn("555", redacted)
        self.assertEqual(ctx.state['query'], redacted)

    def test_pii_redaction_credit_card(self):
        ctx = MagicMock()
        ctx.state = {}
        input_text = "My card number is 1234-5678-9012-3456."
        redacted = redact_pii_node._func(ctx, input_text)
        self.assertIn("[REDACTED_CARD]", redacted)
        self.assertNotIn("1234", redacted)
        self.assertEqual(ctx.state['query'], redacted)

    def test_pii_redaction_no_pii(self):
        ctx = MagicMock()
        ctx.state = {}
        input_text = "spent 50 dollars on lunch yesterday."
        redacted = redact_pii_node._func(ctx, input_text)
        self.assertEqual(redacted, "spent 50 dollars on lunch yesterday.")
        self.assertEqual(ctx.state['query'], redacted)

    def test_router(self):
        ctx = MagicMock()
        orchestrator_output = OrchestratorOutput(intent="TransactionClassifier")
        res = router._func(ctx, orchestrator_output)
        self.assertEqual(ctx.route, "TransactionClassifier")
        self.assertEqual(res.intent, "TransactionClassifier")

    async def test_transaction_classifier(self):
        from unittest.mock import AsyncMock
        ctx = MagicMock()
        ctx.state = {'query': "spent 50 dollars on lunch"}
        
        from finance_concierge.agent import TransactionClassificationResult
        mock_result = TransactionClassificationResult(
            amount=50.0,
            description="lunch",
            category="Food",
            date="2026-06-26"
        )
        
        ctx.run_node = AsyncMock(return_value=mock_result)
        
        tc_res = await transaction_classifier._func(ctx)
        self.assertEqual(tc_res.agent, "TransactionClassifier")
        self.assertEqual(tc_res.amount, 50.0)
        self.assertEqual(tc_res.description, "lunch")
        self.assertEqual(tc_res.category, "Food")
        self.assertEqual(tc_res.date, "2026-06-26")
        self.assertEqual(tc_res.status, "success")

    async def test_budget_analyst(self):
        from unittest.mock import AsyncMock, patch
        ctx = MagicMock()
        ctx.state = {'query': "how much is my food budget"}
        
        from finance_concierge.agent import BudgetAnalystCategoryResult
        mock_category = BudgetAnalystCategoryResult(category="Food")
        ctx.run_node = AsyncMock(return_value=mock_category)
        
        with patch('finance_concierge.agent.get_budgets_and_spending') as mock_get:
            mock_get.return_value = ({"Food": 200.0, "Other": 50.0}, {"Food": 120.0, "Other": 10.0})
            
            res = await budget_analyst._func(ctx)
            self.assertEqual(res.agent, "BudgetAnalyst")
            self.assertEqual(res.category, "Food")
            self.assertEqual(res.budget, 200.0)
            self.assertEqual(res.spent, 120.0)
            self.assertEqual(res.remaining, 80.0)
            self.assertEqual(res.percentage_used, 60.0)
            self.assertEqual(res.status, "under_budget")

    async def test_alert_agent(self):
        from unittest.mock import patch
        ctx = MagicMock()
        
        with patch('finance_concierge.agent.get_budgets_and_spending') as mock_get:
            # Food spent 170/200 = 85% (warning), Shopping spent 200/150 = 133.3% (over_budget)
            mock_get.return_value = (
                {"Food": 200.0, "Shopping": 150.0, "Bills": 100.0},
                {"Food": 170.0, "Shopping": 200.0, "Bills": 50.0}
            )
            
            res = await alert_agent._func(ctx)
            self.assertEqual(res.agent, "AlertAgent")
            self.assertEqual(res.total_alerts, 2)
            self.assertEqual(res.alerts[0].category, "Shopping")
            self.assertEqual(res.alerts[0].severity, "over_budget")
            self.assertEqual(res.alerts[1].category, "Food")
            self.assertEqual(res.alerts[1].severity, "warning")

    async def test_report_generator(self):
        from unittest.mock import patch
        ctx = MagicMock()
        
        with patch('finance_concierge.agent.get_budgets_and_spending') as mock_get:
            mock_get.return_value = (
                {"Food": 200.0, "Shopping": 150.0},
                {"Food": 120.0, "Shopping": 200.0}
            )
            
            res = await report_generator._func(ctx)
            self.assertEqual(res.agent, "ReportGenerator")
            self.assertEqual(res.total_budget, 350.0)
            self.assertEqual(res.total_spent, 320.0)
            self.assertEqual(res.total_remaining, 30.0)
            self.assertEqual(res.savings_rate, 8.57)
            self.assertEqual(res.over_budget_categories, ["Shopping"])
            self.assertEqual(res.under_budget_categories, ["Food"])

    async def test_get_budgets_and_spending_success(self):
        from unittest.mock import AsyncMock, patch
        from mcp.types import CallToolResult, TextContent
        
        mock_result = CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "categories": [
                            {"category": "Food", "budget": 500.0, "spent": 100.0},
                            {"category": "Bills", "budget": 300.0, "spent": 300.0}
                        ]
                    })
                )
            ]
        )
        
        with patch('finance_concierge.agent.toolset._execute_with_session', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            from finance_concierge.agent import get_budgets_and_spending
            budgets, spending = await get_budgets_and_spending()
            self.assertEqual(budgets["Food"], 500.0)
            self.assertEqual(spending["Food"], 100.0)
            self.assertEqual(budgets["Bills"], 300.0)
            self.assertEqual(spending["Bills"], 300.0)

    async def test_get_budgets_and_spending_fallback(self):
        from unittest.mock import AsyncMock, patch
        with patch('finance_concierge.agent.toolset._execute_with_session', new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Connection refused")
            from finance_concierge.agent import get_budgets_and_spending, FALLBACK_MONTHLY_BUDGETS
            budgets, spending = await get_budgets_and_spending()
            self.assertEqual(budgets, FALLBACK_MONTHLY_BUDGETS)

if __name__ == '__main__':
    unittest.main()
