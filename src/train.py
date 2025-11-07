import json
import os
import random
from pathlib import Path
from typing import Tuple, List

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from .model import (
    apply_calibration,
    build_backbone_with_peft,
    compute_hyper_volume,
    count_kept_gates,
    inject_tri_granular_gates,
    load_and_prepare_surrogate,
)
from .preprocess import build_dataloaders, build_tokenizer

################################################################################
# Reproducibility                                                               #
################################################################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

################################################################################
# WandB helper                                                                  #
################################################################################

def _init_wandb(cfg):
    """Initialise WandB respecting cfg.wandb.mode."""
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[wandb] Run URL: {run.get_url()}")
    return run

################################################################################
# Optuna objective                                                              #
################################################################################

def _build_trial_objects(cfg, device):
    model = build_backbone_with_peft(cfg)
    gates = inject_tri_granular_gates(model, cfg.model.gating.hard_concrete_temperature)
    model.to(device)
    return model, gates


def _optuna_objective(trial, base_cfg, device, loaders, surrogate, calib):
    """Light-weight objective – validation accuracy."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

    # Hyper-parameters ---------------------------------------------------------
    cfg.training.learning_rate = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    cfg.objective_weights.lambda_energy = trial.suggest_float("lambda_energy", 1e-1, 5.0, log=True)
    cfg.objective_weights.mu_latency = trial.suggest_float("mu_latency", 1e-1, 5.0, log=True)
    cfg.objective_weights.nu_power = trial.suggest_float("nu_power", 1e-1, 5.0, log=True)

    model, gates = _build_trial_objects(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    train_loader, val_loader = loaders
    model.train()

    # One mini-batch update ----------------------------------------------------
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()

    outputs = model(**batch)
    task_loss = outputs["loss"]

    stats_vec = count_kept_gates(gates, batch, cfg).to(device)
    hw_id = torch.tensor([cfg.surrogate.target_hw_id], device=device)
    e_pred, l_pred, p_pred = apply_calibration(surrogate(stats_vec, hw_id), calib)

    loss = (
        task_loss
        + cfg.objective_weights.lambda_energy * e_pred
        + cfg.objective_weights.mu_latency * l_pred
        + cfg.objective_weights.nu_power * p_pred
    )
    loss.backward()
    optimizer.step()

    # quick val ---------------------------------------------------------------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for vb in val_loader:
            vb = {k: v.to(device) for k, v in vb.items()}
            logits = model(**vb)["logits"]
            preds = logits.argmax(-1)
            correct += (preds == vb["labels"]).sum().item()
            total += preds.size(0)
            break  # only first batch
    acc = correct / max(total, 1)
    return acc

################################################################################
# Training loop                                                                 #
################################################################################

def _train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- Data ----
    tokenizer = build_tokenizer(cfg)
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    # --------------------------------------------------------- Surrogate + Cal #
    surrogate, calib = load_and_prepare_surrogate(cfg, device)

    # ------------------------------------------------------------ Optuna ----
    if cfg.optuna.n_trials > 0:
        print(f"[optuna] running {cfg.optuna.n_trials} trials …")
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(
            lambda t: _optuna_objective(
                t, cfg, device, (train_loader, val_loader), surrogate, calib
            ),
            n_trials=cfg.optuna.n_trials,
        )
        print("[optuna] best params:", study.best_params)
        # update cfg with best params
        for k, v in study.best_params.items():
            if k == "lr":
                cfg.training.learning_rate = v
            elif hasattr(cfg.objective_weights, k):
                setattr(cfg.objective_weights, k, v)

    # -------------------------------------------------------------- Model ----
    model = build_backbone_with_peft(cfg)
    gates = inject_tri_granular_gates(model, cfg.model.gating.hard_concrete_temperature)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    wandb_run = _init_wandb(cfg)

    global_step, best_val_acc = 0, 0.0
    final_cm: List[List[int]] = []

    for epoch in range(cfg.training.num_epochs):
        # ===================================================== Train ==========
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            task_loss = outputs["loss"]

            # hardware predictions -------------------------------------------
            stats_vec = count_kept_gates(gates, batch, cfg).to(device)
            hw_id = torch.tensor([cfg.surrogate.target_hw_id], device=device)
            e_pred, l_pred, p_pred = apply_calibration(surrogate(stats_vec, hw_id), calib)

            loss = (
                task_loss
                + cfg.objective_weights.lambda_energy * e_pred
                + cfg.objective_weights.mu_latency * l_pred
                + cfg.objective_weights.nu_power * p_pred
            )
            loss.backward()
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "task_loss": task_loss.item(),
                        "pred_energy": e_pred.item(),
                        "pred_latency": l_pred.item(),
                        "pred_power": p_pred.item(),
                    },
                    step=global_step,
                )
            global_step += 1
            if cfg.mode == "trial" and step >= 1:
                break

        # ===================================================== Val ==========
        model.eval()
        correct = total = 0
        val_loss_accum = 0.0
        val_preds: List[int] = []
        val_labels: List[int] = []
        with torch.no_grad():
            for vb in val_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                out = model(**vb)
                logits = out["logits"]
                val_loss_accum += F.cross_entropy(logits, vb["labels"]).item()
                preds = logits.argmax(-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(vb["labels"].cpu().tolist())
                correct += (preds == vb["labels"]).sum().item()
                total += preds.size(0)
                if cfg.mode == "trial":
                    break

        val_acc = correct / max(total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        # Confusion matrix ----------------------------------------------------
        num_labels = getattr(getattr(model, "config", None), "num_labels", len(set(val_labels)))
        cm = confusion_matrix(val_labels, val_preds, labels=list(range(num_labels)))
        final_cm = cm.tolist()

        # HV ------------------------------------------------------------------
        hv = compute_hyper_volume(
            val_acc,
            e_pred.item(),
            l_pred.item(),
            p_pred.item(),
            cfg.budgets,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "val_loss": val_loss_accum / max(total, 1),
                    "val_acc": val_acc,
                    "hv": hv,
                    "confusion_matrix": final_cm,
                },
                step=global_step,
            )
        print(f"[epoch {epoch}] val_acc={val_acc:.4f}  hv={hv:.4f}")

    # ===================================================== Finish ===========
    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["hv"] = hv
        wandb_run.summary["confusion_matrix"] = final_cm
        wandb_run.finish()

################################################################################
# Hydra entry-point                                                             #
################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # Merge run-specific YAML --------------------------------------------------
    run_cfg_path = Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(run_cfg_path)
    run_cfg = OmegaConf.load(run_cfg_path)
    cfg = OmegaConf.merge(cfg, run_cfg)

    # Mode handling -----------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.num_epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    set_seed(cfg.training.seed)
    _train(cfg)

if __name__ == "__main__":
    main()