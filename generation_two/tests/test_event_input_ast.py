import unittest

from generation_two.core.fast_expr_ast import FASTEXPRParser


class EventInputASTTests(unittest.TestCase):
    def setUp(self):
        operators = [
            {"name": name, "scope": ["REGULAR"]}
            for name in ("add", "subtract", "multiply", "divide", "rank")
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


if __name__ == "__main__":
    unittest.main()
