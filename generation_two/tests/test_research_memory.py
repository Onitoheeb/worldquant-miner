import json
import os
import sqlite3
import tempfile
import unittest

from generation_two.research_memory import ResearchMemory


class ResearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "research.db")
        self.memory = ResearchMemory(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_records_catalogue_and_experiment_evidence(self):
        self.assertTrue(self.memory.record_catalogue(
            "USA",
            operators=[{
                "name": "ts_rank",
                "definition": "ts_rank(input, lookback)",
                "scope": ["REGULAR"],
            }],
            fields=[{
                "id": "close",
                "type": "MATRIX",
                "dataset_id": "price",
                "coverage": 0.99,
            }],
        ))
        self.assertTrue(self.memory.record_result({
            "alpha_id": "alpha-1",
            "template": "ts_rank(close, 20)",
            "region": "USA",
            "success": True,
            "sharpe": 1.5,
            "fitness": 1.2,
            "turnover": 0.3,
            "returns": 0.12,
            "drawdown": 0.08,
            "power_pool_corr": json.dumps({"max": 0.21}),
            "prod_corr": json.dumps({"max": 0.18}),
            "checks": json.dumps([{"name": "SELF_CORRELATION", "result": "PASS"}]),
            "timestamp": 100.0,
        }))
        context = self.memory.build_research_context("USA")
        self.assertIn("ts_rank(close, 20)", context)
        self.assertIn("Maximum absolute reported correlation: 0.210", context)
        self.assertIn("SELF_CORRELATION", context)

    def test_failed_experiment_becomes_a_lesson_not_a_winner(self):
        self.memory.record_result({
            "template": "unknown_operator(close)",
            "region": "USA",
            "success": False,
            "error_message": "Unknown operator: unknown_operator",
            "timestamp": 101.0,
        })
        self.assertEqual(self.memory.get_diverse_top_results("USA"), [])
        context = self.memory.build_research_context("USA")
        self.assertIn("unknown_operator(close)", context)
        self.assertIn("Unknown operator", context)

    def test_same_alpha_is_updated_instead_of_duplicated(self):
        result = {
            "alpha_id": "alpha-2",
            "template": "rank(close)",
            "region": "USA",
            "success": True,
            "sharpe": 1.0,
            "fitness": 1.0,
            "timestamp": 200.0,
        }
        self.memory.record_result(result)
        result["sharpe"] = 1.4
        result["timestamp"] = 201.0
        self.memory.record_result(result)

        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM research_experiments WHERE alpha_id = ?",
                ("alpha-2",),
            ).fetchone()[0]
            stored_sharpe = conn.execute(
                "SELECT sharpe FROM research_experiments WHERE alpha_id = ?",
                ("alpha-2",),
            ).fetchone()[0]

        self.assertEqual(count, 1)
        self.assertEqual(stored_sharpe, 1.4)


if __name__ == "__main__":
    unittest.main()
