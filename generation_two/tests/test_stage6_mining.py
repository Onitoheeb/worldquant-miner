from types import SimpleNamespace
from unittest.mock import Mock, patch

from generation_two.core.slot_manager import SlotStatus
from generation_two.gui.components.workflow_steps.step6_mining_modules.mining_engine import MiningEngine


def test_glb_expression_starts_one_worker_and_reserves_two_slots():
    slot_manager = Mock()
    slot_manager.get_slot_status.return_value = SimpleNamespace(status=SlotStatus.IDLE)
    slot_manager.assign_slot.side_effect = [[0, 1], [2, 3], [4, 5], [6, 7], None]

    correlation_tracker = Mock()
    correlation_tracker.get_low_correlation_templates.return_value = []

    engine = MiningEngine(
        generator=None,
        simulator_tester=None,
        backtest_storage=None,
        slot_manager=slot_manager,
        correlation_tracker=correlation_tracker,
        duplicate_detector=None,
        search_strategy_manager=None,
        sim_counter=SimpleNamespace(can_simulate=lambda: True),
    )
    templates = iter((f"alpha_{i}", "GLB") for i in range(8))
    engine._pick_unsimulated_template_from_db = lambda: next(templates, (None, None))

    with patch("random.random", return_value=0.9), patch(
        "generation_two.gui.components.workflow_steps.step6_mining_modules.mining_engine.threading.Thread"
    ) as thread_class:
        engine._process_simulations()

    # Four GLB expressions fit into eight visual slots. Each expression gets
    # exactly one worker, not two workers for its two reserved slots.
    assert thread_class.call_count == 4
    assert slot_manager.assign_slot.call_count == 5
    assert all(call.kwargs["args"][0] in ([0, 1], [2, 3], [4, 5], [6, 7]) for call in thread_class.call_args_list)


def test_stage6_metric_gate_rejects_completed_but_weak_alphas():
    weak = SimpleNamespace(
        success=True, alpha_id="weak-alpha", sharpe=0.9, fitness=0.7, turnover=12.0
    )
    assert not MiningEngine._meets_metric_gates(weak)


def test_stage6_metric_gate_accepts_fractional_turnover_and_strong_metrics():
    strong = SimpleNamespace(
        success=True, alpha_id="strong-alpha", sharpe=1.8, fitness=1.2, turnover=0.25
    )
    assert MiningEngine._meets_metric_gates(strong)


def test_stage6_metric_gate_rejects_missing_alpha_id():
    no_id = SimpleNamespace(
        success=True, alpha_id="", sharpe=2.4, fitness=1.5, turnover=10.0
    )
    assert not MiningEngine._meets_metric_gates(no_id)
