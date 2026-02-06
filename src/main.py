import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def _repo_root() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:
        return Path(__file__).resolve().parents[1]


def _apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    return cfg


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg = _apply_mode_overrides(cfg)
    run_id = str(getattr(cfg.run, "run_id", ""))
    if not run_id:
        raise ValueError("run_id missing in cfg.run. Ensure run configuration is loaded.")

    repo_root = _repo_root()

    overrides = [
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"wandb.mode={cfg.wandb.mode}",
        f"seed={cfg.seed}",
    ]
    if cfg.mode == "trial":
        overrides.append("run.optuna.n_trials=0")

    cmd = [sys.executable, "-m", "src.train"] + overrides
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))


if __name__ == "__main__":
    main()
