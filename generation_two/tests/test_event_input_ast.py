import unittest

from generation_two.core.fast_expr_ast import FASTEXPRParser
from generation_two.core.template_validator import TemplateValidator
from generation_two.core.template_generator import _cleanup_legacy_event_input_flags


class EventInputASTTests(unittest.TestCase):
    def setUp(self):
        operators = [
            {"name": name, "scope": ["REGULAR"]}
            for name in ("add", "subtract", "multiply", "divide", "rank")
        ] + [
            {"name": "is_nan", "scope": ["REGULAR"], "definition": "is_nan(input)"},
            {"name": "reverse", "scope": ["REGULAR"], "definition": "reverse(x)"},
            {"name": "sign", "scope": ["REGULAR"], "definition": "sign(x)"},
            {"name": "sqrt", "scope": ["REGULAR"], "definition": "sqrt(x)"},
        ]
        data_fields = [
            {
                "id": "event_field",
                "type": "MATRIX",
                "name": "Event metric",
                "is_event_input": True,
            },
            {
                "id": "event_description_field",
                "type": "MATRIX",
                "name": "Metric",
                "description": "A field derived from event data",
                "is_event_input": False,
            },
            {
                "id": "regular_field",
                "type": "MATRIX",
                "name": "Revenue",
                "description": "Standard financial metric",
                "is_event_input": False,
            },
        ]
        self.parser = FASTEXPRParser(
            operators=operators,
            data_fields=data_fields,
            alpha_type="REGULAR",
        )
        self.parser.event_input_incompatible_operators.update(
            {"add", "subtract", "multiply", "divide", "rank"}
        )

    def _assert_event_error(self, expression, operator_name):
        _, errors = self.parser.parse(expression)
        matching = [
            error for error in errors
            if error.error_type == "event_input_incompatible"
            and error.operator_name == operator_name
        ]
        self.assertTrue(
            matching,
            msg=f"Expected {operator_name} event-input error for: {expression}; got {errors}",
        )

    def test_binary_add_rejects_event_field(self):
        self._assert_event_error("event_field + regular_field", "add")

    def test_unary_negation_is_checked_as_multiply(self):
        self._assert_event_error("-event_field", "multiply")

    def test_description_alone_does_not_classify_field_as_event(self):
        _, errors = self.parser.parse(
            "event_description_field + regular_field"
        )
        self.assertFalse(any(
            error.error_type == "event_input_incompatible"
            for error in errors
        ))

    def test_nested_operator_checks_event_field(self):
        self._assert_event_error("rank(event_field)", "rank")

    def test_rejects_extra_argument_to_is_nan(self):
        _, errors = self.parser.parse("is_nan(regular_field, 30)")
        self.assertTrue(any(
            error.error_type == "invalid_operator_arguments"
            and error.operator_name == "is_nan"
            for error in errors
        ))

    def test_rejects_extra_argument_to_reverse(self):
        _, errors = self.parser.parse("reverse(sign(regular_field), 10)")
        self.assertTrue(any(
            error.error_type == "invalid_operator_arguments"
            and error.operator_name == "reverse"
            for error in errors
        ))

    def test_accepts_catalogue_valid_unary_calls(self):
        _, errors = self.parser.parse("sqrt(reverse(sign(regular_field)))")
        self.assertFalse(
            [error for error in errors if error.error_type == "invalid_operator_arguments"],
            msg=f"Unexpected arity errors: {errors}",
        )


    def test_validator_uses_field_metadata_loaded_after_initialization(self):
        validator = TemplateValidator(
            operators=[
                {"name": "is_nan", "scope": ["REGULAR"], "definition": "is_nan(input)"}
            ],
            data_fields=[],
            use_ast=True,
        )
        validator.parser.event_input_incompatible_operators.add("is_nan")

        # This mirrors TemplateGenerator fetching regional fields after the
        # validator itself has already been constructed.
        validator.update_data_fields([
            {
                "id": "event_field",
                "type": "MATRIX",
                "name": "Event metric",
                "description": "Event-derived data",
                "is_event_input": True,
            }
        ])

        valid, error, _ = validator.validate_template("is_nan(event_field)")
        self.assertFalse(valid)
        self.assertIn("does not support event inputs", error)

    def test_unknown_event_classification_is_not_inferred_from_text(self):
        parser = FASTEXPRParser(
            operators=[{"name": "add", "scope": ["REGULAR"]}],
            data_fields=[{
                "id": "event_named_unknown",
                "type": "VECTOR",
                "name": "event sentiment",
                "description": "News event sentiment value",
            }],
        )
        self.assertFalse(parser._is_event_input_field("event_named_unknown"))

    def test_ambiguous_error_does_not_guess_first_expression_operator(self):
        validator = TemplateValidator(
            operators=[{"name": "rank", "scope": ["REGULAR"]}],
            data_fields=[],
            use_ast=True,
        )
        validator._learn_event_input_compatibility(
            "rank(event_field)",
            "Invalid input data for this expression",
        )
        self.assertNotIn("rank", validator.parser.event_input_incompatible_operators)

    def test_database_loader_ignores_guessed_legacy_operator_rows(self):
        import os
        import tempfile
        from generation_two.storage.backtest_storage import BacktestStorage

        handle, db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        try:
            validator = TemplateValidator(
                operators=[],
                data_fields=[],
                db_path=db_path,
                use_ast=True,
            )
            storage = BacktestStorage(db_path)
            storage.store_compiler_knowledge(
                knowledge_type="event_input_incompatible",
                operator_name="rank",
                compatibility_status="incompatible",
                error_message="Invalid input data for this expression",
                learned_from_template="rank(event_field)",
                metadata={"extracted_from_template": True},
            )
            storage.store_compiler_knowledge(
                knowledge_type="event_input_incompatible",
                operator_name="add",
                compatibility_status="incompatible",
                error_message="Operator add does not support event inputs",
                learned_from_template="add(event_field, price)",
                metadata={"evidence_source": "explicit_brain_error"},
            )

            validator.parser.event_input_incompatible_operators.clear()
            loaded = validator.parser.load_event_input_knowledge_from_db(db_path)

            self.assertEqual(loaded, 1)
            self.assertIn("add", validator.parser.event_input_incompatible_operators)
            self.assertNotIn("rank", validator.parser.event_input_incompatible_operators)
        finally:
            os.unlink(db_path)

    def test_startup_cleanup_backs_up_and_clears_only_heuristic_flags(self):
        import json
        import os
        import sqlite3
        import tempfile
        from contextlib import closing

        handle, db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        backup_path = os.path.abspath(db_path) + ".event_flags_pre_cleanup.bak"
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                with conn:
                    conn.execute("""
                    CREATE TABLE field_types (
                        field_id TEXT NOT NULL,
                        region TEXT NOT NULL,
                        delay INTEGER NOT NULL,
                        is_event_input INTEGER,
                        field_data TEXT,
                        updated_at TEXT,
                        UNIQUE(field_id, region, delay)
                    )
                """)
                    rows = [
                        ("legacy_false", "USA", 1, 0, json.dumps({"id": "legacy_false"})),
                        ("legacy_true", "USA", 1, 1, json.dumps({"id": "legacy_true"})),
                        ("explicit_true", "USA", 1, 1, json.dumps({"is_event_input": True})),
                        ("explicit_false", "USA", 1, 0, json.dumps({"is_event_input": False})),
                    ]
                    conn.executemany(
                        "INSERT INTO field_types VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                        rows,
                    )

            cleaned = _cleanup_legacy_event_input_flags(db_path)
            self.assertEqual(cleaned, 2)
            self.assertTrue(os.path.isfile(backup_path))

            with closing(sqlite3.connect(db_path)) as conn:
                flags = dict(conn.execute(
                    "SELECT field_id, is_event_input FROM field_types"
                ).fetchall())
            self.assertIsNone(flags["legacy_false"])
            self.assertIsNone(flags["legacy_true"])
            self.assertEqual(flags["explicit_true"], 1)
            self.assertEqual(flags["explicit_false"], 0)
        finally:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_event_repair_does_not_guess_ts_operator_replacement(self):
        validator = TemplateValidator(
            operators=[
                {"name": "add", "scope": ["REGULAR"], "definition": "add(x, y)"}
            ],
            data_fields=[],
            use_ast=True,
        )
        expression = "add(event_field, regular_field)"
        fixed, fixes = validator._fix_event_input_error(
            expression,
            "Operator add does not support event inputs",
            region="USA",
        )
        self.assertEqual(fixed, expression)
        self.assertEqual(fixes, [])


if __name__ == "__main__":
    unittest.main()
