"""Configuration loader and validator."""
from pathlib import Path
from typing import Any, Dict
import yaml


def find_repo_root() -> Path:
    """Find the retinacare repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs" / "config.yaml").exists() or (current / "PROGRESS.md").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load and validate the YAML configuration."""
    if config_path is None:
        repo_root = find_repo_root()
        config_path = str(repo_root / "configs" / "config.yaml")

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure required sections exist
    required_keys = ["seed", "paths", "sample", "preprocess", "anatomy", "lesion", "grading", "safety"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config section: {k}")

    # Resolve repo root into cfg for convenience
    repo_root = find_repo_root()
    cfg["_repo_root"] = str(repo_root)

    # Resolve data root path dynamically for cross-environment portability
    import os
    env_data_root = os.environ.get("RETINACARE_DATA_ROOT")
    if env_data_root and Path(env_data_root).exists():
        cfg["paths"]["data_root"] = env_data_root
    else:
        configured_path = Path(cfg["paths"].get("data_root", ""))
        if not configured_path.exists():
            if (repo_root.parent / "aptos2019-blindness-detection").exists():
                cfg["paths"]["data_root"] = str(repo_root.parent)
            elif (repo_root / "datasets").exists():
                cfg["paths"]["data_root"] = str(repo_root / "datasets")
            elif (repo_root / "data" / "raw").exists():
                cfg["paths"]["data_root"] = str(repo_root / "data" / "raw")

    return cfg
