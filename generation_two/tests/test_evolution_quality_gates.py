from generation_two.evolution.alpha_evolution_engine import AlphaEvolutionEngine, AlphaResult


def test_evolution_population_uses_only_metric_qualified_seeds():
    engine = AlphaEvolutionEngine()
    engine.initialize_population([
        AlphaResult("rank(close)", 2.0, 1.2, 10.0, "USA", True),
        AlphaResult("rank(volume)", 2.8, 0.7, 8.0, "USA", True),
        AlphaResult("rank(open)", 2.5, 1.4, 35.0, "USA", True),
        AlphaResult("rank(high)", 3.0, 1.5, 12.0, "USA", False),
    ])

    assert [item.template for item in engine.population] == ["rank(close)"]


def test_evolution_does_not_emit_unvalidated_offspring():
    engine = AlphaEvolutionEngine(mutation_rate=1.0, crossover_rate=1.0)
    engine.initialize_population([
        AlphaResult("rank(close)", 2.0, 1.2, 10.0, "USA", True),
        AlphaResult("rank(volume)", 1.8, 1.1, 12.0, "USA", True),
    ])

    offspring = engine.evolve_generation()

    known_good = {item.template for item in engine.population}
    assert offspring
    assert set(offspring).issubset(known_good)


def test_evolution_accepts_offspring_only_after_validator_approves():
    class Accepted:
        success = True

    engine = AlphaEvolutionEngine(
        mutation_rate=1.0,
        crossover_rate=1.0,
        candidate_validator=lambda expression: Accepted(),
    )
    engine.initialize_population([
        AlphaResult("rank(close)", 2.0, 1.2, 10.0, "USA", True),
        AlphaResult("rank(volume)", 1.8, 1.1, 12.0, "USA", True),
    ])

    offspring = engine.evolve_generation()

    assert len(offspring) == len(engine.population)
