"""P8 Report Builder Pipeline orchestrator."""
import logging
from pathlib import Path
from typing import Dict, Any, List

from retinacare.common.case_record import CaseRecord
from retinacare.p8_report.case_pdf import generate_case_pdf
from retinacare.p8_report.scorecard import generate_scorecard


class P8_Pipeline:
    """Pipeline 8: Report Builder (Clinician PDFs and System Scorecard)."""

    def __init__(self, config: Dict[str, Any], manifest_hash: str = "LOCKED"):
        self.config = config
        self.manifest_hash = manifest_hash
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.results_dir = repo_root / config["paths"]["results"]
        self.reports_dir = self.artifacts_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p8")

    def run_case(self, case: CaseRecord) -> CaseRecord:
        """Generate clinician PDF report for a single case."""
        case_id = case.case_id
        out_pdf_path = self.reports_dir / f"{case_id}.pdf"
        repo_root = Path(self.config.get("_repo_root", "."))

        try:
            success = generate_case_pdf(
                case,
                out_pdf_path=out_pdf_path,
                repo_root=repo_root,
                manifest_hash=self.manifest_hash
            )

            case.p8["status"] = "ok" if success else "failed"
            case.p8["error"] = None
            case.p8["pdf_path"] = str(out_pdf_path.relative_to(self.artifacts_dir.parent).as_posix())

        except Exception as e:
            case.p8["status"] = "failed"
            case.p8["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord]) -> List[CaseRecord]:
        """Generate clinician PDFs for a batch of cases."""
        results = []
        for c in cases:
            res = self.run_case(c)
            res.save(self.artifacts_dir)
            results.append(res)
        return results

    def build_global_reports(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build scorecard and run reports."""
        return generate_scorecard(metrics_dict, self.results_dir)
