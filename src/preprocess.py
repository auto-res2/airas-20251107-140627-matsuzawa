from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

################################################################################
# Tokeniser                                                                     #
################################################################################

def build_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache/")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer

################################################################################
# Dataset                                                                       #
################################################################################

def _tokenise(batch, tokenizer, cfg):
    text_col = "sentence" if "sentence" in batch else list(batch.keys())[0]
    return tokenizer(
        batch[text_col],
        truncation=True,
        max_length=cfg.dataset.max_length,
        padding=cfg.dataset.padding,
    )


def build_dataloaders(cfg, tokenizer) -> Tuple[DataLoader, DataLoader]:
    name = cfg.dataset.source.split("huggingface:")[-1]
    raw = load_dataset(name, cache_dir=".cache/")

    if "label" in raw["train"].column_names:
        raw = raw.rename_column("label", "labels")

    tokenised = raw.map(lambda b: _tokenise(b, tokenizer, cfg), batched=True)
    tokenised.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    val_split = "validation" if "validation" in tokenised else "test"
    train_ds, val_ds = tokenised["train"], tokenised[val_split]

    # Trial mode – keep only two batches --------------------------------------
    if cfg.mode == "trial":
        train_ds = train_ds.select(range(min(2 * cfg.dataset.batch_size, len(train_ds))))
        val_ds = val_ds.select(range(min(2 * cfg.dataset.batch_size, len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader