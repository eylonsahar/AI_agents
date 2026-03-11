#!/usr/bin/env python3
"""
AItzik Automated Test Suite — pre-deployment validation.

Tiers:
  0 — Pure unit tests (no server, no LLM)
  1 — API contract tests (server needed, no LLM)
  2 — Intent classifier tests (adversarial prompts)
  3 — Full pipeline tests (budget-guarded)

Usage:
  python -m validation.test_auto_suite --tiers 0 1 --url http://localhost:8001
  python -m validation.test_auto_suite --tiers 0 1 2 3 --random --budget 3.0
  python -m validation.test_auto_suite --tiers 0 1 2 3 --random 8 --budget 3.0
  python -m validation.test_auto_suite --tiers 2 3 --replay validation/samples/saved_prompts_sample.json

  With --random [N]: N is the number of prompts to generate per agent (default 6).
  Tier 3 generates N total: floor(N*3/4) normal good prompts + floor(N/4) inexact-model prompts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .models import COST_PER_RUN_ESTIMATE, ROOT, TestResult


# ---------------------------------------------------------------------------
# AItzikTestSuite
# ---------------------------------------------------------------------------


class AItzikTestSuite:
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        budget_limit_usd: float = 3.0,
    ):
        self.base_url = base_url
        self.budget_limit_usd = budget_limit_usd
        self.max_pipeline_runs = max(1, int(budget_limit_usd / COST_PER_RUN_ESTIMATE))
        self.pipeline_runs_used = 0
        self.results: List[TestResult] = []
        self.saved_prompts: Dict[str, Any] = {}
        self._declared_modules: set = set()
        self._module_consistency: Dict[str, Any] = {}
        self.efficiency: List[Dict[str, Any]] = []

    def _add_result(self, r: TestResult) -> None:
        self.results.append(r)

    def _can_run_pipeline(self) -> bool:
        return self.pipeline_runs_used < self.max_pipeline_runs

    def _advance_progress(self) -> None:
        """Update tqdm bar and budget postfix (no-op if no bar)."""
        if getattr(self, "_pbar", None) is not None:
            self._pbar.update(1)
            budget_used = self.pipeline_runs_used * COST_PER_RUN_ESTIMATE
            pct = 100 * budget_used / self.budget_limit_usd if self.budget_limit_usd > 0 else 0
            self._pbar.set_postfix_str(f"${budget_used:.2f}/${self.budget_limit_usd:.2f} ({pct:.0f}%)")

    def run(
        self,
        tiers: List[int],
        random_mode: bool = False,
        replay_file: Optional[str] = None,
        num_examples: int = 6,
    ) -> None:
        good_prompts: List[str] = []
        inexact_prompts: List[str] = []
        bad_prompts: List[str] = []

        if replay_file and os.path.isfile(replay_file):
            with open(replay_file, encoding="utf-8") as f:
                data = json.load(f)
                good_prompts = data.get("good_prompts", [])
                bad_prompts = data.get("bad_prompts", [])
                inexact_prompts = data.get("inexact_prompts", [])
        elif random_mode and (2 in tiers or 3 in tiers):
            if 2 in tiers:
                from .bad_user_agent import BadUserAgent

                bad_agent = BadUserAgent()
                bad_prompts = bad_agent.generate(num_examples)
            if 3 in tiers:
                from .good_user_agent import GoodUserAgent

                good_agent = GoodUserAgent()
                n_inexact = num_examples // 4
                n_normal = num_examples - n_inexact
                good_prompts = good_agent.generate(n_normal)
                inexact_prompts = good_agent.generate_inexact(n_inexact)
            self.saved_prompts = {
                "generated_at": datetime.now().isoformat(),
                "good_prompts": good_prompts,
                "bad_prompts": bad_prompts,
                "inexact_prompts": inexact_prompts,
            }
        else:
            if 2 in tiers:
                bad_prompts = [
                    "book a flight to Paris under $5000, from 2022",
                    "find me a house under $200000 from 2018",
                    "recommend a laptop under $2000, 2022 or newer",
                ]
            if 3 in tiers:
                good_prompts = [
                    "Economy sedan, max $15,000, 2019 or newer",
                    "Pickup truck under $30000 from 2017",
                ]

        # Compute total steps for progress bar (ETA + budget) — from tier modules
        total_steps = 0
        if 0 in tiers:
            from .tier0 import get_step_count as get_tier0_step_count
            total_steps += get_tier0_step_count()
        if 1 in tiers:
            from .tier1 import get_step_count as get_tier1_step_count
            total_steps += get_tier1_step_count()
        if 2 in tiers:
            from .tier2 import get_step_count as get_tier2_step_count
            total_steps += get_tier2_step_count(bad_prompts)
        if 3 in tiers:
            from .tier3 import get_fixed_step_count
            total_steps += get_fixed_step_count() + len(good_prompts) + len(inexact_prompts)

        from tqdm import tqdm

        with tqdm(total=total_steps, desc="Validation", unit="test") as pbar:
            self._pbar = pbar

            if 0 in tiers:
                try:
                    from .tier0 import run_tier0

                    run_tier0(self)
                except Exception as e:
                    import traceback

                    self._add_result(
                        TestResult(
                            name="tier0_crash",
                            tier=0,
                            passed=False,
                            detail=f"Tier 0 failed to run: {e}",
                            exception_traceback=traceback.format_exc(),
                        )
                    )
            if 1 in tiers:
                try:
                    from .tier1 import run_tier1

                    run_tier1(self)
                except Exception as e:
                    import traceback

                    self._add_result(
                        TestResult(
                            name="tier1_crash",
                            tier=1,
                            passed=False,
                            detail=f"Tier 1 failed to run: {e}",
                            exception_traceback=traceback.format_exc(),
                        )
                    )
            if 2 in tiers:
                try:
                    from .tier2 import run_tier2

                    run_tier2(self, bad_prompts)
                except Exception as e:
                    import traceback

                    self._add_result(
                        TestResult(
                            name="tier2_crash",
                            tier=2,
                            passed=False,
                            detail=f"Tier 2 failed to run: {e}",
                            exception_traceback=traceback.format_exc(),
                        )
                    )
            if 3 in tiers:
                try:
                    from .tier3 import run_tier3

                    run_tier3(self, good_prompts, inexact_prompts)
                except Exception as e:
                    import traceback

                    self._add_result(
                        TestResult(
                            name="tier3_crash",
                            tier=3,
                            passed=False,
                            detail=f"Tier 3 failed to run: {e}",
                            exception_traceback=traceback.format_exc(),
                        )
                    )

    def generate_report(
        self,
        tiers_run: List[int],
        random_mode: bool,
        replay_file: Optional[str],
        num_examples: int = 6,
    ) -> Tuple[str, str]:
        """Write test_report_<ts>.json and .txt; return (json_path, txt_path)."""
        reports_dir = ROOT / "validation" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = reports_dir / f"test_report_{ts}.json"
        txt_path = reports_dir / f"test_report_{ts}.txt"

        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed and "SKIPPED" not in r.detail),
            "skipped": sum(1 for r in self.results if "SKIPPED" in r.detail),
        }

        report_json = {
            "run_at": datetime.now().isoformat(),
            "target_url": self.base_url,
            "tiers_run": tiers_run,
            "random_mode": random_mode,
            "num_examples": num_examples,
            "replay_file": replay_file,
            "summary": summary,
            "pipeline_runs_used": self.pipeline_runs_used,
            "efficiency": self.efficiency,
            "module_consistency": getattr(self, "_module_consistency", {}),
            "tests": [r.to_dict() for r in self.results],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, ensure_ascii=False)

        # Human-readable report
        lines = [
            "=" * 60,
            "AItzik Test Report",
            "=" * 60,
            f"Run at: {report_json['run_at']}",
            f"Target URL: {self.base_url}",
            f"Tiers run: {tiers_run}",
            f"Random mode: {random_mode}",
            f"Num examples: {num_examples}",
            f"Replay file: {replay_file or 'none'}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total: {summary['total']}  Passed: {summary['passed']}  Failed: {summary['failed']}  Skipped: {summary['skipped']}",
            f"Pipeline runs used: {self.pipeline_runs_used}",
            "",
        ]

        # Graded requirements
        lines.append("GRADED REQUIREMENTS")
        lines.append("-" * 40)
        for name in ["test_team_info", "test_agent_info_completeness", "test_model_architecture", "test_frontend_smoke"]:
            r = next((x for x in self.results if x.name == name), None)
            if r:
                lines.append(f"  {name}: {'PASS' if r.passed else 'FAIL'}")
        mc = getattr(self, "_module_consistency", {})
        if mc:
            mismatches = mc.get("mismatches", [])
            lines.append(f"  module_name_consistency: {'PASS' if not mismatches else 'FAIL'}")
        lines.append("")

        # Failed tests
        failed = [r for r in self.results if not r.passed and "SKIPPED" not in r.detail]
        if failed:
            lines.append("FAILED TESTS")
            lines.append("-" * 40)
            for r in failed:
                lines.append(f"  [{r.tier}] {r.name}: {r.detail}")
                if r.prompt_used:
                    lines.append(f"    prompt: {r.prompt_used[:80]}...")
            lines.append("")

        # AItzik response snapshots (for presentation — what AItzik returned on each end-to-end run)
        with_responses = [r for r in self.results if getattr(r, "response_snapshot", None)]
        if with_responses:
            lines.append("AITZIK RESPONSE SNAPSHOTS")
            lines.append("-" * 40)
            lines.append("(For manual review — not auto-tested.)")
            for r in with_responses:
                lines.append(f"  [{r.tier}] {r.name}")
                if r.prompt_used:
                    lines.append(f"    Prompt: {r.prompt_used[:80]}{'...' if len(r.prompt_used) > 80 else ''}")
                snap = (r.response_snapshot or "")[:500]
                lines.append(f"    Response: {snap}{'...' if len(r.response_snapshot or '') > 500 else ''}")
                lines.append("")
            lines.append("")

        # Unexpected exceptions (code failures — traceback + request for replay)
        with_traceback = [r for r in self.results if getattr(r, "exception_traceback", None)]
        if with_traceback:
            lines.append("UNEXPECTED EXCEPTIONS (code failures)")
            lines.append("-" * 40)
            for r in with_traceback:
                lines.append(f"  [{r.tier}] {r.name}: {r.detail}")
                if getattr(r, "request_snapshot", None):
                    lines.append(f"    Request (for replay): {json.dumps(r.request_snapshot, indent=6)}")
                lines.append("    Traceback:")
                for tb_line in (r.exception_traceback or "").split("\n"):
                    lines.append(f"      {tb_line}")
                lines.append("")
            lines.append("")

        # Efficiency
        if self.efficiency:
            lines.append("EFFICIENCY")
            lines.append("-" * 40)
            for e in self.efficiency:
                flag = " [FLAGGED]" if e.get("flagged") else ""
                lines.append(f"  {e.get('prompt', '')[:50]}... steps={e.get('step_count', 0)}{flag}")
            lines.append("")

        # Adversarial
        adv = [r for r in self.results if r.tier == 2 and not r.passed and "bypass" in r.detail.lower()]
        if adv:
            lines.append("ADVERSARIAL RESULTS (classifier bypasses)")
            lines.append("-" * 40)
            for r in adv:
                lines.append(f"  {r.prompt_used}")
            lines.append("")

        # Skipped
        skipped = [r for r in self.results if "SKIPPED" in r.detail]
        if skipped:
            lines.append("SKIPPED")
            lines.append("-" * 40)
            for r in skipped[:10]:
                lines.append(f"  {r.name}: {r.detail}")
            if len(skipped) > 10:
                lines.append(f"  ... and {len(skipped) - 10} more")
            lines.append("")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Save prompts if generated
        if self.saved_prompts:
            save_path = reports_dir / f"saved_prompts_{ts}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_prompts, f, indent=2, ensure_ascii=False)

        return str(json_path), str(txt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AItzik automated test suite")
    parser.add_argument("--tiers", type=int, nargs="+", default=[0,1], help="Tiers to run (0,1,2,3)")
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=6,
        default=None,
        metavar="N",
        help="Use LLM to generate N random prompts per agent (default 6). Tier 3 uses floor(N*3/4) normal + floor(N/4) inexact-model prompts.",
    )
    parser.add_argument("--replay", type=str, default=None, help="Replay prompts from saved JSON file")
    parser.add_argument("--budget", type=float, default=3.0, help="Budget limit in USD for pipeline runs")
    parser.add_argument("--url", type=str, default="http://localhost:8001", help="Base URL of the API server")
    args = parser.parse_args()

    random_mode = args.random is not None
    num_examples = args.random if args.random is not None else 6

    suite = AItzikTestSuite(base_url=args.url, budget_limit_usd=args.budget)
    suite.run(
        tiers=args.tiers,
        random_mode=random_mode,
        replay_file=args.replay,
        num_examples=num_examples,
    )
    json_path, txt_path = suite.generate_report(
        tiers_run=args.tiers,
        random_mode=random_mode,
        replay_file=args.replay,
        num_examples=num_examples,
    )
    print(f"Report written: {json_path}")
    print(f"Report written: {txt_path}")
    passed = sum(1 for r in suite.results if r.passed)
    failed = sum(1 for r in suite.results if not r.passed and "SKIPPED" not in r.detail)
    print(f"Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
