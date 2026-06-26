import unittest
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
        ctx = MagicMock()
        ctx.state = {'query': "spent 50 dollars on lunch"}
        
        from finance_concierge.agent import TransactionClassificationResult
        mock_result = TransactionClassificationResult(
            amount=50.0,
            description="lunch",
            category="Food",
            date="2026-06-26"
        )
        
        async def mock_run_node(agent, node_input):
            return mock_result
            
        ctx.run_node = mock_run_node
        
        tc_res = await transaction_classifier._func(ctx)
        self.assertEqual(tc_res.agent, "TransactionClassifier")
        self.assertEqual(tc_res.amount, 50.0)
        self.assertEqual(tc_res.description, "lunch")
        self.assertEqual(tc_res.category, "Food")
        self.assertEqual(tc_res.date, "2026-06-26")
        self.assertEqual(tc_res.status, "success")

    def test_subagent_stubs(self):
        ctx = MagicMock()
        
        ba_res = budget_analyst._func(ctx)
        self.assertEqual(ba_res.agent, "BudgetAnalyst")
        self.assertEqual(ba_res.status, "stub")
        
        aa_res = alert_agent._func(ctx)
        self.assertEqual(aa_res.agent, "AlertAgent")
        self.assertEqual(aa_res.status, "stub")
        
        rg_res = report_generator._func(ctx)
        self.assertEqual(rg_res.agent, "ReportGenerator")
        self.assertEqual(rg_res.status, "stub")

if __name__ == '__main__':
    unittest.main()
