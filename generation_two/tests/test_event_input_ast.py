import unittest

from generation_two.core.fast_expr_ast import FASTEXPRParser
from generation_two.core.template_validator import TemplateValidator


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
            },
            {
                "id": "event_description_field",
                "type": "MATRIX",
                "name": "Metric",
                "description": "A field derived from event data",
            },
            {
                "id": "regular_field",
                "type": "MATRIX",
                "name": "Revenue",
                "description": "Standard financial metric",
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

    def test_event_metadata_in_description_is_recognized(self):
        self._assert_event_error("event_description_field + regular_field", "add")

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
            }
        ])

        valid, error, _ = validator.validate_template("is_nan(event_field)")
        self.assertFalse(valid)
        self.assertIn("does not support event inputs", error)

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
