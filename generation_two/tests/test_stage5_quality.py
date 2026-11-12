from types import SimpleNamespace

from generation_two.gui.components.workflow_steps.step5_simulation import (
    _meets_research_metric_gates,
)


def test_stage5_rejects_completed_but_low_quality_alpha():
    result = SimpleNamespace(
        success=True, alpha_id="weak", sharpe=1.1, fitness=0.8, turnover=12.0
    )
    assert not _meets_research_metric_gates(result)


def test_stage5_accepts_minimum_metric_gates_and_fractional_turnover():
    result = SimpleNamespace(
        success=True, alpha_id="candidate", sharpe=1.5, fitness=1.0, turnover=0.30
    )
    assert _meets_research_metric_gates(result)


def test_stage5_rejects_missing_alpha_id():
    result = SimpleNamespace(
        success=True, alpha_id="", sharpe=2.5, fitness=1.4, turnover=10.0
    )
    assert not _meets_research_metric_gates(result)
