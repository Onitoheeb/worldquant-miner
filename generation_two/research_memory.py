"""
Persistent research memory for the Generation Two BRAIN alpha researcher.

This module records observed experiment outcomes and turns them into grounded
context for future LLM research. It does not infer operator compatibility from
names, does not label untested expressions as valid, and does not submit alphas.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_OPERATOR_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_RESERVED = {
    "and", "or", "not", "true", "false", "nan", "inf",
    "if", "else", "for", "in",
}


class ResearchMemory:
    """SQLite-backed experiment memory shared by generation and simulation."""

    def __init__(self, db_path: str = "generation_two_backtests.db"):
        self.db_path = db_path
        self._create_tables()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_tables(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alpha_id TEXT,
                    template TEXT NOT NULL,
                    region TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    sharpe REAL DEFAULT 0,
                    fitness REAL DEFAULT 0,
                    turnover REAL DEFAULT 0,
                    returns REAL DEFAULT 0,
                    drawdown REAL DEFAULT 0,
                    correlations TEXT DEFAULT '',
                    power_pool_corr TEXT DEFAULT '',
                    prod_corr TEXT DEFAULT '',
                    checks_json TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    operators_json TEXT DEFAULT '[]',
                    fields_json TEXT DEFAULT '[]',
                    timestamp REAL NOT NULL,
                    UNIQUE(alpha_id, template, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_region_success
                ON research_experiments(region, success, fitness, sharpe)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis TEXT NOT NULL,
                    region TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    supporting_evidence TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(research_experiments)")
            }
            for column_name in ("power_pool_corr", "prod_corr", "checks_json"):
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE research_experiments ADD COLUMN {column_name} TEXT DEFAULT ''"
                    )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_catalogue_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    delay INTEGER NOT NULL,
                    alpha_type TEXT NOT NULL,
                    operators_json TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    captured_at REAL NOT NULL
                )
            """)

    @staticmethod
    def _value(result, key, default=None):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    @staticmethod
    def _extract_expression_components(template: str):
        operators = sorted(set(_OPERATOR_CALL.findall(template or "")))
        identifiers = sorted(set(_IDENTIFIER.findall(template or "")))
        fields = [
            name for name in identifiers
            if name.lower() not in _RESERVED
            and name not in operators
            and not name.isupper()
            and not name.isdigit()
        ]
        return operators, fields

    def record_result(self, result) -> bool:
        """Persist an observed simulation result without inventing compatibility."""
        try:
            template = str(self._value(result, "template", "") or "").strip()
            region = str(self._value(result, "region", "") or "").strip()
            if not template or not region:
                return False

            timestamp = float(self._value(result, "timestamp", 0) or time.time())
            alpha_id = str(self._value(result, "alpha_id", "") or "")
            operators, fields = self._extract_expression_components(template)
            correlations = self._value(result, "correlations", "") or ""
            power_pool_corr = self._value(result, "power_pool_corr", "") or ""
            prod_corr = self._value(result, "prod_corr", "") or ""
            checks = self._value(result, "checks", "") or ""
            def serialize(value):
                return value if isinstance(value, str) else json.dumps(value, default=str)
            correlations = serialize(correlations)
            power_pool_corr = serialize(power_pool_corr)
            prod_corr = serialize(prod_corr)
            checks = serialize(checks)

            values = (
                alpha_id, template, region,
                int(bool(self._value(result, "success", False))),
                float(self._value(result, "sharpe", 0) or 0),
                float(self._value(result, "fitness", 0) or 0),
                float(self._value(result, "turnover", 0) or 0),
                float(self._value(result, "returns", 0) or 0),
                float(self._value(result, "drawdown", 0) or 0),
                correlations, power_pool_corr, prod_corr, checks,
                str(self._value(result, "error_message", "") or "")[:2000],
                json.dumps(operators), json.dumps(fields), timestamp,
            )
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO research_experiments
                    (alpha_id, template, region, success, sharpe, fitness,
                     turnover, returns, drawdown, correlations, power_pool_corr,
                     prod_corr, checks_json, error_message, operators_json,
                     fields_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, values)
            return True
        except Exception:
            logger.exception("Could not record research experiment")
            return False

    def record_catalogue(
        self, region: str, operators: List[Dict], fields: List[Dict],
        delay: int = 1, alpha_type: str = "REGULAR"
    ) -> bool:
        """Persist the operator and field metadata available for a research run."""
        if not region:
            return False
        try:
            operator_keys = ("name", "definition", "description", "category", "scope")
            field_keys = (
                "id", "type", "field_type", "description", "category",
                "dataset", "dataset_id", "coverage", "delay", "region"
            )
            operator_snapshot = [
                {key: item.get(key) for key in operator_keys if item.get(key) is not None}
                for item in (operators or [])[:1500]
            ]
            field_snapshot = [
                {key: item.get(key) for key in field_keys if item.get(key) is not None}
                for item in (fields or [])[:6000]
            ]
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO research_catalogue_snapshots
                    (region, delay, alpha_type, operators_json, fields_json, captured_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    region, int(delay), (alpha_type or "REGULAR").upper(),
                    json.dumps(operator_snapshot, default=str),
                    json.dumps(field_snapshot, default=str), time.time()
                ))
            return True
        except Exception:
            logger.exception("Could not persist BRAIN research catalogue")
            return False

    def record_hypothesis(
        self, hypothesis: str, region: str,
        status: str = "proposed", supporting_evidence: str = ""
    ) -> bool:
        if not hypothesis or not region:
            return False
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO research_hypotheses
                    (hypothesis, region, status, supporting_evidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (hypothesis[:2000], region, status,
                      supporting_evidence[:4000], time.time()))
            return True
        except Exception:
            logger.exception("Could not record research hypothesis")
            return False

    @staticmethod
    def _operator_set(template: str):
        return set(_OPERATOR_CALL.findall(template or ""))

    def get_diverse_top_results(
        self, region: str, limit: int = 5, candidate_pool: int = 60
    ) -> List[Dict]:
        """Return high-performing historical experiments with operator diversity."""
        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT alpha_id, template, sharpe, fitness, turnover,
                           returns, drawdown, correlations, timestamp
                    FROM research_experiments
                    WHERE region = ? AND success = 1
                    ORDER BY fitness DESC, sharpe DESC, timestamp DESC
                    LIMIT ?
                """, (region, candidate_pool)).fetchall()

            selected = []
            operator_sets = []
            for row in rows:
                item = {
                    "alpha_id": row[0], "template": row[1],
                    "sharpe": row[2] or 0, "fitness": row[3] or 0,
                    "turnover": row[4] or 0, "returns": row[5] or 0,
                    "drawdown": row[6] or 0, "correlations": row[7] or "",
                    "timestamp": row[8] or 0,
                }
                current = self._operator_set(item["template"])
                # Avoid filling the context with near-identical operator families.
                too_similar = False
                for previous in operator_sets:
                    union = current | previous
                    similarity = len(current & previous) / len(union) if union else 1.0
                    if similarity >= 0.8:
                        too_similar = True
                        break
                if too_similar:
                    continue
                selected.append(item)
                operator_sets.append(current)
                if len(selected) >= limit:
                    break
            return selected
        except Exception:
            logger.exception("Could not retrieve diverse research seeds")
            return []

    def get_recent_failures(self, region: str, limit: int = 8) -> List[Dict]:
        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT template, error_message, timestamp
                    FROM research_experiments
                    WHERE region = ? AND success = 0 AND error_message != ''
                    ORDER BY timestamp DESC LIMIT ?
                """, (region, limit)).fetchall()
            return [
                {"template": row[0], "error": row[1], "timestamp": row[2]}
                for row in rows
            ]
        except Exception:
            logger.exception("Could not retrieve failed research experiments")
            return []

    def build_research_context(self, region: str, max_chars: int = 6500) -> str:
        """Build concise, evidence-grounded context for the next LLM generation."""
        seeds = self.get_diverse_top_results(region, limit=5)
        failures = self.get_recent_failures(region, limit=6)
        if not seeds and not failures:
            return ""

        sections = [
            "PERSISTENT RESEARCH MEMORY (observed BRAIN experiments only):",
            "Use these observations as evidence, not guarantees. Do not invent "
            "operator support, field compatibility, or performance.",
        ]
        if seeds:
            sections.append("STRONGER HISTORICAL EXPERIMENTS:")
            for index, item in enumerate(seeds, 1):
                sections.append(
                    f"{index}. Expression: {item['template']} | "
                    f"Sharpe={item['sharpe']:.3f}, Fitness={item['fitness']:.3f}, "
                    f"Turnover={item['turnover']:.3f}, Returns={item['returns']:.3f}, "
                    f"Drawdown={item['drawdown']:.3f}"
                )
        if failures:
            sections.append("RECENT FAILED EXPERIMENTS (avoid repeating blindly):")
            for item in failures:
                sections.append(
                    f"- {item['template']} | observed error: {item['error'][:350]}"
                )
        sections.append(
            "RESEARCH INSTRUCTIONS: Form a distinct financial hypothesis. "
            "Prefer a different data relationship or operator structure when "
            "historical candidates are similar. Preserve the economic meaning "
            "of the hypothesis. Treat failures as configuration-specific evidence. "
            "Only the BRAIN simulation can establish actual validity and performance."
        )
        return "\n".join(sections)[:max_chars]

    def build_evolution_prompt(self, region: str, count: int = 3) -> str:
        """Create an LLM prompt to explore measured winners without cloning them."""
        seeds = self.get_diverse_top_results(region, limit=5)
        if not seeds:
            return (
                f"Develop {count} distinct, testable quantitative alpha hypotheses "
                f"for region {region}. Use only the currently supplied BRAIN "
                "operators and data fields. Explain the financial intuition briefly "
                "and generate different FASTEXPR expressions."
            )
        seed_lines = [
            f"- {s['template']} (Sharpe {s['sharpe']:.3f}, "
            f"Fitness {s['fitness']:.3f}, Turnover {s['turnover']:.3f})"
            for s in seeds
        ]
        return (
            f"Act as a quantitative research scientist. Develop {count} new alpha "
            f"experiments for region {region}. Start from the measured candidates "
            "below, identify the economic signal behind them, then explore "
            "meaningfully different hypotheses and transformations. Do not merely "
            "change constants or wrap the same expression in another operator. "
            "Avoid near-duplicates and explain the hypothesis each expression tests. "
            "Use only supplied fields and operators; never assume unsupported "
            "operators are valid. Return FASTEXPR expressions only when requested.\n"
            "MEASURED RESEARCH SEEDS:\n" + "\n".join(seed_lines)
        )
