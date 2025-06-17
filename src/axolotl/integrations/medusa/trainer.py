import logging
from types import MethodType
from typing import List, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import LabelSmoother
from transformers import PretrainedConfig
from axolotl.core.trainers.base import AxolotlTrainer

LOG = logging.getLogger("axolotl.medusa")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # typically -100

class ResBlock(nn.Module):
    """A simple residual block: linear layer (init to zero) + SiLU, added to input."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)   # initialize to zero for identity mapping:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))

def add_medusa_heads(model: nn.Module, medusa_num_heads: int):
    """
    Dynamically attach Medusa heads to the model.
    Adds `medusa_head` ModuleList to predict multiple tokens ahead.
    """
    # Determine hidden and vocab size from model's output layer
    if hasattr(model, "lm_head"):
        base_output_weight = model.lm_head.weight
    elif hasattr(model, "embed_out"):
        base_output_weight = model.embed_out.weight  # e.g. for models using embed_out
    else:
        raise RuntimeError("Cannot locate output head (lm_head or embed_out) in model for Medusa.")
    hidden_size = base_output_weight.shape[-1]
    vocab_size = base_output_weight.shape[0]
    model.medusa_num_heads = medusa_num_heads

    # Create new decoding heads: each is ResBlock + Linear to vocab:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    model.medusa_head = nn.ModuleList([
        nn.Sequential(
            ResBlock(hidden_size),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )
        for _ in range(medusa_num_heads)
    ])
    # Initialize each Medusa head's output weights from the base LM head for a good starting point:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
    for i in range(medusa_num_heads):
        model.medusa_head[i][-1].weight.data.copy_(base_output_weight.data)

    # Define a patched forward method that returns stacked logits from base and Medusa heads
    def medusa_forward(self, 
                       input_ids: Optional[torch.LongTensor] = None,
                       attention_mask: Optional[torch.Tensor] = None,
                       **kwargs):
        # Determine if we should freeze base model grads during forward
        train_only_heads: bool = bool(kwargs.pop("train_only_medusa_heads", False))
        # Run the original model forward under no-grad if base is frozen (Medusa-1):contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
        context_mgr = torch.no_grad() if train_only_heads else torch.enable_grad()
        with context_mgr:
            base_outputs = self(**{**kwargs, "input_ids": input_ids, "attention_mask": attention_mask}) \
                           if hasattr(self, "model") and callable(self.model) \
                           else super(type(self), self).forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            # ^ The base model forward. For PeftModel, self.model is the underlying base model.
            # If called on the base model itself, use super(...).forward(...).
            hidden_states = base_outputs[0] if isinstance(base_outputs, tuple) else base_outputs.logits
        # Compute logits for base head and each Medusa head
        base_logits = self.lm_head(hidden_states)
        medusa_logits = [base_logits]
        for h in self.medusa_head:
            medusa_logits.append(h(hidden_states))
        # Stack logits: shape = (num_heads+1, batch, seq_len, vocab)
        return torch.stack(medusa_logits, dim=0)
    # Monkey-patch the model's forward method
    if hasattr(model, "model") and callable(model.model):
        # If model is a Peft wrapper, patch the base model inside it
        model.model.forward = MethodType(medusa_forward, model.model)
    else:
        model.forward = MethodType(medusa_forward, model)
    LOG.info(f"Added {medusa_num_heads} Medusa heads to model.")
    return model

def freeze_base_model(model: nn.Module):
    """Freeze all original model parameters, leaving only Medusa heads trainable."""
    for name, param in model.named_parameters():
        # Freeze everything except Medusa head modules
        if not name.startswith("medusa_head"):
            param.requires_grad = False
    LOG.info("Base model layers frozen; only Medusa heads will be trained.")


class AxolotlMedusaTrainer(AxolotlTrainer):
    """
    Custom Trainer to handle Medusa multi-head loss computation.
    """
    def __init__(self, *args, medusa_num_heads=None, medusa_heads_coefficient=0.1, 
                 medusa_decay_coefficient=1.0, medusa_scheduler="constant",
                 medusa_lr_multiplier=1.0, train_only_medusa_heads=True, **kwargs):
        # Store Medusa parameters
        self.medusa_num_heads = medusa_num_heads or 0
        self.medusa_heads_coefficient = medusa_heads_coefficient or 1.0
        self.medusa_decay_coefficient = medusa_decay_coefficient or 1.0
        self.medusa_scheduler = medusa_scheduler or "constant"
        self.medusa_lr_multiplier = medusa_lr_multiplier or 1.0
        self.train_only_medusa_heads = train_only_medusa_heads
        # Workaround for quantized models without LoRA: mark as quantized to bypass checks
        model = kwargs.get("model")
        if self.train_only_medusa_heads and model is not None and hasattr(model, "is_quantized"):
            model.is_quantized = True
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute custom loss combining base and Medusa head predictions."""
        labels = inputs["labels"]
        # Forward pass – will return stacked logits tensor:contentReference[oaicite:21]{index=21}
        logits = model(
            input_ids=inputs.get("input_ids"), 
            attention_mask=inputs.get("attention_mask"),
            train_only_medusa_heads=self.train_only_medusa_heads
        )
        num_heads_total = logits.shape[0]  # = medusa_num_heads + 1 (including base)
        loss_fct = CrossEntropyLoss()
        total_loss = 0.0
        log_metrics = {}
        # Calculate loss for each head’s predictions:contentReference[oaicite:22]{index=22}:contentReference[oaicite:23]{index=23}
        for i in range(num_heads_total):
            # Align logits and labels: head i predicts tokens shifted by i positions ahead:contentReference[oaicite:24]{index=24}:contentReference[oaicite:25]{index=25}
            pred_logits = logits[i, :, : -(1 + i)]  # drop the last i+1 predictions
            target_labels = labels[..., (1 + i) :]   # drop the first i+1 labels
            pred_logits = pred_logits.reshape(-1, pred_logits.size(-1))
            target_labels = target_labels.reshape(-1).to(pred_logits.device)
            if target_labels.numel() == 0:
                continue  # skip if sequence is too short for this head
            loss_i = loss_fct(pred_logits, target_labels)
            if i == 0:
                # Base head loss (only include if training base model):contentReference[oaicite:26]{index=26}
                if not self.train_only_medusa_heads:
                    total_loss += loss_i
            else:
                # Weighted Medusa head loss:contentReference[oaicite:27]{index=27}
                coeff = 1.0
                if self.medusa_scheduler == "constant":
                    coeff = 1.0  # (could extend for other schedulers)
                # Decay factor: medusa_decay_coefficient^i
                total_loss += loss_i * (self.medusa_decay_coefficient ** i) * self.medusa_heads_coefficient * coeff
            # (Optional) Compute accuracy metrics for logging
            not_ignore = target_labels.ne(IGNORE_TOKEN_ID)
            correct = None
            with torch.no_grad():
                # Example: top-1 accuracy for head i
                _, top1 = pred_logits.max(dim=-1)
                correct = top1.eq(target_labels)
            if correct is not None:
                log_metrics[f"medusa{i}_acc"] = (correct.masked_select(not_ignore).float().mean().item())
            log_metrics[f"medusa{i}_loss"] = loss_i.item()
        LOG.debug(f"Medusa loss breakdown: {log_metrics}")
        return (total_loss, logits) if return_outputs else total_loss

    def create_optimizer(self):
        """
        Customize optimizer to apply a different LR to Medusa head params:contentReference[oaicite:28]{index=28}:contentReference[oaicite:29]{index=29}.
        """
        opt_model = self.model
        # Get default parameter groups from parent class (weight decay handling)
        decay_params = self.get_decay_parameter_names(opt_model)
        base_lr = self.args.learning_rate
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() 
                           if n in decay_params and p.requires_grad and "medusa_head" not in n],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() 
                           if n in decay_params and p.requires_grad and "medusa_head" in n],
                "weight_decay": self.args.weight_decay,
                "lr": base_lr * self.medusa_lr_multiplier,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() 
                           if n not in decay_params and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = AxolotlTrainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        self.optimizer = optimizer
        return optimizer


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path