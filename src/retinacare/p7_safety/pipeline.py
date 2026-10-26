"""P7 Safety Pipeline orchestrator."""
import logging
from typing import Dict, Any, List

from retinacare.common.case_record import CaseRecord
from retinacare.p7_safety.rules import evaluate_safety_rules


class P7_Pipeline:
    """Pipeline 7: Safety Checks and Review Prioritization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("retinacare.p7")

    def run_case(self, case: CaseRecord) -> CaseRecord:
        """Evaluate safety rules and assign clinician review priority."""
        try:
            flags, priority = evaluate_safety_rules(case, self.config)

            case.p7["status"] = "ok"
            case.p7["flags"] = flags
            case.p7["priority"] = priority
            # Non-negotiable constraint: review is ALWAYS required
            case.p7["review_required"] = True

        except Exception as e:
            case.p7["status"] = "failed"
            case.p7["error"] = str(e)
            case.p7["priority"] = "REVIEW_PRIORITY_HIGH"
            case.p7["review_required"] = True

        return case

    def run_batch(self, cases: List[CaseRecord]) -> List[CaseRecord]:
        """Run safety checks on a batch of cases."""
        results = []
        for c in cases:
            res = self.run_case(c)
            results.append(res)
        return results
