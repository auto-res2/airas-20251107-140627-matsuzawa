import json
import math
import re
import types
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForSequenceClassification

################################################################################
# Hard-Concrete gate                                                            #
################################################################################

class HardConcreteGate(nn.Module):
    """Scalar / vector differentiable gate following Louizos et al."""

    def __init__(self, size: Tuple[int, ...] = (1,), temperature: float = 2.0, init_mean: float = 0.5):
        super().__init__()
        self.temperature = temperature
        logit = math.log(init_mean) - math.log(1 - init_mean)
        self.log_alpha = nn.Parameter(torch.full(size, logit))
        self.register_buffer("eps", torch.tensor(1e-6))

    def _sample(self):
        u = torch.rand_like(self.log_alpha)
        s = torch.sigmoid((torch.log(u + self.eps) - torch.log(1 - u + self.eps) + self.log_alpha) / self.temperature)
        return torch.clamp(s, 0.0, 1.0)

    def prob(self):
        return torch.sigmoid(self.log_alpha)

    def forward(self):
        return self._sample() if self.training else self.prob()

################################################################################
# Backbone + PEFT                                                              #
################################################################################

def build_backbone_with_peft(cfg):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name, num_labels=2, cache_dir=".cache/"
        )
    except ValueError:
        base = AutoModel.from_pretrained(cfg.model.name, cache_dir=".cache/")
        model = ClassificationWrapper(base, base.config.hidden_size, 2)

    # PEFT --------------------------------------------------------------------
    if cfg.model.get("peft"):
        p = cfg.model.peft
        lora_cfg = LoraConfig(
            r=p.r,
            lora_alpha=p.alpha,
            lora_dropout=p.dropout,
            task_type="SEQ_CLS",
            target_modules=list(p.target_modules),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    return model

################################################################################
# Gating injection                                                             #
################################################################################

def inject_tri_granular_gates(model: nn.Module, temperature: float):
    """Inject layer-, head- and LoRA-rank gates; return dict of gate lists."""
    layer_gates, head_gates, rank_gates = [], [], []

    # Layer gates --------------------------------------------------------------
    pattern = re.compile(r"\.layer\.[0-9]+|layers\.[0-9]+|\.h\.[0-9]+")
    for n, m in model.named_modules():
        if pattern.search(n) and not hasattr(m, "hc_layer_gate"):
            gate = HardConcreteGate((1,), temperature)
            setattr(m, "hc_layer_gate", gate)

            def hook(mod, _inp, output, gate=gate):
                return output * gate().view(1, 1, 1)

            m.register_forward_hook(hook)
            layer_gates.append(gate)

    # Head gates ---------------------------------------------------------------
    for m in model.modules():
        if hasattr(m, "num_heads") and not hasattr(m, "hc_head_gate"):
            n_heads = int(m.num_heads)
            gate = HardConcreteGate((n_heads,), temperature)
            setattr(m, "hc_head_gate", gate)

            def head_hook(mod, _inp, output, gate=gate):
                bs, sl, dim = output.shape
                head_dim = dim // n_heads
                out = output.view(bs, sl, n_heads, head_dim)
                gated = out * gate().view(1, 1, n_heads, 1)
                return gated.view(bs, sl, dim)

            m.register_forward_hook(head_hook)
            head_gates.append(gate)

    # Rank (LoRA) gates --------------------------------------------------------
    for m in model.modules():
        if hasattr(m, "lora_A") and not hasattr(m, "hc_rank_gate"):
            r = m.lora_A[0].weight.size(0)
            gate = HardConcreteGate((r,), temperature)
            setattr(m, "hc_rank_gate", gate)

            def lora_hook(mod, _inp, output, gate=gate):
                return output * gate().mean()

            m.register_forward_hook(lora_hook)
            rank_gates.append(gate)

    return {"layer": layer_gates, "head": head_gates, "rank": rank_gates}

################################################################################
# Gate stats helpers                                                           #
################################################################################

def _gate_kept(gates: List[HardConcreteGate], thresh: float = 0.5) -> int:
    if not gates:
        return 0
    keep = [g.prob().gt(thresh).float().sum().item() for g in gates]
    return int(sum(keep) / len(keep))


def count_kept_gates(gate_dict: Dict[str, List[HardConcreteGate]], batch, cfg):
    seq_len = batch["attention_mask"].sum(dim=1).float().mean().unsqueeze(0)
    kept_layers = torch.tensor([_gate_kept(gate_dict["layer"])]).float()
    kept_heads = torch.tensor([_gate_kept(gate_dict["head"])]).float()
    kept_ranks = torch.tensor([_gate_kept(gate_dict["rank"])]).float()
    precision_bits = 16 if "16" in str(getattr(cfg.model, "precision", "32")).lower() else 32
    prec = torch.tensor([precision_bits]).float()
    return torch.stack([seq_len, kept_layers, kept_heads, kept_ranks, prec], dim=1)

################################################################################
# Classification wrapper                                                       #
################################################################################

class ClassificationWrapper(nn.Module):
    def __init__(self, backbone: AutoModel, hidden: int, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden, num_labels)
        # Provide a minimal config attribute so that external code can query num_labels
        self.config = types.SimpleNamespace(num_labels=num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, -1]
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

################################################################################
# Hyper-volume computation                                                     #
################################################################################

def compute_hyper_volume(acc: float, energy: float, latency: float, power: float, budgets) -> float:
    if energy > budgets.energy_j or latency > budgets.latency_ms or power > budgets.power_w:
        return 0.0
    return max(acc, 0.0) * (budgets.energy_j - energy) * (budgets.latency_ms - latency) * (budgets.power_w - power)

################################################################################
# Meta-surrogate                                                               #
################################################################################

class MetaSurrogate(nn.Module):
    """ψ_φ predicting (energy, latency, power) from statistics + hardware id."""

    def __init__(self, n_hw: int, d_stat: int = 5):
        super().__init__()
        self.hw_embed = nn.Embedding(n_hw, 8)
        self.net = nn.Sequential(
            nn.Linear(d_stat + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # E, L, P
        )

    def forward(self, stats: torch.Tensor, hw_id: torch.Tensor):
        h = self.hw_embed(hw_id)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        x = torch.cat([stats, h], dim=-1)
        return self.net(x)

################################################################################
# Surrogate utilities                                                          #
################################################################################

def load_and_prepare_surrogate(cfg, device):
    """Load ψ_φ and calibration (α,β)."""
    model = MetaSurrogate(cfg.surrogate.n_hardware_tags).to(device)
    ckpt = Path(cfg.surrogate.checkpoint_path)
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location=device)
            state = state["state_dict"] if "state_dict" in state else state
            model.load_state_dict(state)
            print(f"[surrogate] loaded checkpoint {ckpt}")
        except Exception as e:
            print(f"[surrogate] failed to load checkpoint: {e}")
    else:
        print(f"[surrogate] checkpoint {ckpt} not found – using random weights")
    model.eval()

    # Calibration --------------------------------------------------------------
    alpha = torch.ones(1, 3, device=device)
    beta = torch.zeros(1, 3, device=device)
    calib_file = Path(getattr(cfg.surrogate, "calibration_file", ""))
    if calib_file.exists():
        with open(calib_file) as fp:
            d = json.load(fp)
        alpha = torch.tensor(d.get("alpha", [1.0, 1.0, 1.0]), device=device).view(1, 3)
        beta = torch.tensor(d.get("beta", [0.0, 0.0, 0.0]), device=device).view(1, 3)
        print(f"[surrogate] loaded calibration {calib_file}")
    else:
        print("[surrogate] no calibration file – identity calibration used")

    calib = {"alpha": alpha, "beta": beta}
    return model, calib


def apply_calibration(preds: torch.Tensor, calib: Dict[str, torch.Tensor]):
    """Apply α⊙pred + β and return tuple of (E,L,P) scalars."""
    corrected = calib["alpha"] * preds + calib["beta"]
    e_pred, l_pred, p_pred = corrected[0, 0], corrected[0, 1], corrected[0, 2]
    return e_pred, l_pred, p_pred