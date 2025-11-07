import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

################################################################################
# Utilities                                                                     #
################################################################################

def _save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)

################################################################################
# Per-run processing                                                            #
################################################################################

def _plot_learning_curves(history: pd.DataFrame, rid: str, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 4))
    for col in [c for c in ["train_loss", "val_acc", "hv"] if c in history.columns]:
        sns.lineplot(x=history.index, y=history[col], label=col)
    plt.title(f"Learning Curves – {rid}")
    plt.tight_layout()
    fp = out_dir / f"{rid}_learning_curve.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


def _plot_confusion_matrix(cm: np.ndarray, rid: str, out_dir: Path) -> Path:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title(f"Confusion – {rid}")
    plt.tight_layout()
    fp = out_dir / f"{rid}_confusion_matrix.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


def _process_run(run, out_root: Path):
    """Download history & summary, generate per-run plots and save metrics."""
    rid = run.id
    run_dir = out_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    history: pd.DataFrame = run.history()
    summary: Dict = dict(run.summary)
    cfg: Dict = dict(run.config)

    # CSV + JSON --------------------------------------------------------------
    history.to_csv(run_dir / "history.csv", index=False)
    _save_json({"summary": summary, "config": cfg}, run_dir / "metrics.json")

    # Figures -----------------------------------------------------------------
    paths: List[Path] = []
    paths.append(_plot_learning_curves(history, rid, run_dir))

    if "confusion_matrix" in summary:
        cm = np.asarray(summary["confusion_matrix"], dtype=int)
        paths.append(_plot_confusion_matrix(cm, rid, run_dir))

    for p in paths:
        print(p)
    return summary

################################################################################
# Aggregation / comparison                                                      #
################################################################################

def _aggregate(summaries: Dict[str, Dict], out_root: Path):
    comp_dir = out_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- Metrics
    # Collect every numeric scalar from summaries
    metrics: Dict[str, Dict[str, float]] = {}
    for rid, summ in summaries.items():
        for k, v in summ.items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, {})[rid] = float(v)

    # Primary metric ----------------------------------------------------------
    hv_map = metrics.get("hv", {})

    best_prop = max(
        ((rid, v) for rid, v in hv_map.items() if "proposed" in rid),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    best_base = max(
        ((rid, v) for rid, v in hv_map.items() if ("baseline" in rid or "comparative" in rid)),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    gap = (
        (best_prop[1] - best_base[1]) / best_base[1] * 100 if best_base[1] else 0.0
    )

    # ---------------------------------------------------------------- Figures
    # Bar chart
    if hv_map:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(hv_map.keys()), y=list(hv_map.values()))
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("HV")
        plt.tight_layout()
        fp_bar = comp_dir / "comparison_hv_bar.pdf"
        plt.savefig(fp_bar)
        plt.close()
        print(fp_bar)

        # Box plot grouped by method -----------------------------------------
        data = pd.DataFrame(
            {
                "run_id": list(hv_map.keys()),
                "hv": list(hv_map.values()),
                "method": [
                    "proposed" if "proposed" in rid else "baseline"
                    for rid in hv_map.keys()
                ],
            }
        )
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, x="method", y="hv")
        plt.tight_layout()
        fp_box = comp_dir / "comparison_hv_boxplot.pdf"
        plt.savefig(fp_box)
        plt.close()
        print(fp_box)

        # t-test --------------------------------------------------------------
        hv_prop = data[data.method == "proposed"]["hv"].values
        hv_base = data[data.method == "baseline"]["hv"].values
        t_pval = None
        if len(hv_prop) > 0 and len(hv_base) > 0:
            _, t_pval = ttest_ind(hv_prop, hv_base, equal_var=False)
    else:
        t_pval = None

    # Table CSV ---------------------------------------------------------------
    df_metrics = pd.DataFrame(metrics).T  # metric × run_id
    fp_table = comp_dir / "aggregated_metrics_table.csv"
    df_metrics.to_csv(fp_table)
    print(fp_table)

    # JSON dump ---------------------------------------------------------------
    out_json = {
        "primary_metric": "4-D hyper-volume (HV) of accuracy vs (energy, latency, power) on the unseen Pixel-8.",
        "metrics": metrics,
        "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
        "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
        "gap": gap,
        "stat_tests": {"hv_ttest_p": t_pval},
    }
    fp_json = comp_dir / "aggregated_metrics.json"
    _save_json(out_json, fp_json)
    print(fp_json)

################################################################################
# Main                                                                          #
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    out_root = Path(args.results_dir)
    run_ids_list: List[str] = json.loads(args.run_ids)

    # Load global WandB config -------------------------------------------------
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    hydra_cfg = OmegaConf.load(cfg_path)
    entity, project = hydra_cfg.wandb.entity, hydra_cfg.wandb.project

    api = wandb.Api()
    summaries: Dict[str, Dict] = {}

    for rid in run_ids_list:
        try:
            run = api.run(f"{entity}/{project}/{rid}")
        except wandb.errors.CommError:
            print(f"[warning] run {rid} not found – skipping")
            continue
        summaries[rid] = _process_run(run, out_root)

    _aggregate(summaries, out_root)
    print("[evaluation] complete")

if __name__ == "__main__":
    main()