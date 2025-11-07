import subprocess
import sys
from pathlib import Path

import hydra

################################################################################
# Launcher                                                                      #
################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if cfg.mode not in ("trial", "full"):
        raise ValueError("mode must be trial/full")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("[main] executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()