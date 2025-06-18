import logging
from types import MethodType
from typing import List, Optional
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
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
    def __init__(self, *args,
                 medusa_num_heads=None, medusa_heads_coefficient=0.1,
                 medusa_decay_coefficient=1.0, medusa_scheduler="constant",
                 medusa_lr_multiplier=1.0, train_only_medusa_heads=True,
                 medusa_self_distillation=False, medusa_distillation_regularization=False,
                 medusa_logging=False, medusa_num_unfreeze_layers=0,
                 **kwargs):
        # Store Medusa parameters
        self.medusa_num_heads = medusa_num_heads or 0
        self.medusa_heads_coefficient = medusa_heads_coefficient or 1.0
        self.medusa_decay_coefficient = medusa_decay_coefficient or 1.0
        self.medusa_scheduler = medusa_scheduler or "constant"
        self.medusa_lr_multiplier = medusa_lr_multiplier or 1.0
        self.train_only_medusa_heads = train_only_medusa_heads
        # Store advanced Medusa features
        self.medusa_self_distillation = bool(medusa_self_distillation)
        self.medusa_distillation_regularization = bool(medusa_distillation_regularization)
        self.medusa_logging = bool(medusa_logging)
        self.medusa_num_unfreeze_layers = medusa_num_unfreeze_layers or 0
        # Workaround for quantized models without LoRA: mark as quantized to bypass checks
        model = kwargs.get("model")
        if self.train_only_medusa_heads and model is not None and hasattr(model, "is_quantized"):
            model.is_quantized = True
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute custom loss combining base and Medusa head predictions, with optional distillation regularization."""
        labels = inputs["labels"]
        # Forward pass – will return stacked logits tensor (shape: [num_heads+1, batch, seq_len, vocab])
        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            train_only_medusa_heads=self.train_only_medusa_heads
        )
        num_heads_total = logits.shape[0]  # = medusa_num_heads + 1 (including base)
        total_loss = 0.0
        log_metrics = {}

        # If self-distillation is enabled, prepare base (teacher) logits distribution (detached)
        teacher_logits_full = None
        if self.medusa_self_distillation:
            # Use base logits as teacher (detach to avoid gradient to base)
            teacher_logits_full = logits[0].detach()  # shape: (batch, seq_len, vocab)
        loss_fct = CrossEntropyLoss()
        for i in range(num_heads_total):
            # Align logits and labels for head i: head i predicts tokens shifted by i+1 positions ahead
            if i == 0:
                # Base head predictions (next token)
                pred_logits = logits[0][:, :-1, :]  # drop the last token prediction
                target_labels = labels[..., 1:]      # drop the first token label
            else:
                pred_logits = logits[i][:, :-(1 + i), :]  # drop the last i+1 predictions
                target_labels = labels[..., (1 + i):]     # drop the first i+1 labels
            # Flatten for loss computation
            pred_logits_flat = pred_logits.reshape(-1, pred_logits.size(-1))
            target_flat = target_labels.reshape(-1).to(pred_logits.device)
            # Skip if no labels for this head (sequence too short)
            if target_flat.numel() == 0:
                continue

            # Compute loss for this head
            if i == 0:
                # Base head loss (only include if training base model)
                if not self.train_only_medusa_heads:
                    if self.medusa_distillation_regularization:
                        # Use label smoothing via KL (label smoothing with epsilon=0.1)
                        # PyTorch CrossEntropyLoss supports label_smoothing in recent versions
                        base_loss = F.cross_entropy(pred_logits_flat, target_flat, ignore_index=IGNORE_TOKEN_ID, label_smoothing=0.1)
                    else:
                        base_loss = loss_fct(pred_logits_flat, target_flat)
                    total_loss += base_loss
                    log_metrics[f"medusa0_loss"] = base_loss.item()
            else:
                # Medusa head loss (with weighting)
                loss_i = loss_fct(pred_logits_flat, target_flat)
                # Scheduler coefficient based on medusa_scheduler type
                if self.medusa_scheduler == "constant":
                    coeff_schedule = 1.0
                elif self.medusa_scheduler == "linear":
                    # Linearly increase weight over training progress
                    if self.state and self.state.global_step is not None and self.state.max_steps is not None:
                        progress = float(self.state.global_step) / float(self.state.max_steps) if self.state.max_steps > 0 else 1.0
                    else:
                        progress = 1.0
                    coeff_schedule = progress
                elif self.medusa_scheduler == "sine":
                    if self.state and self.state.global_step is not None and self.state.max_steps is not None:
                        progress = float(self.state.global_step) / float(self.state.max_steps) if self.state.max_steps > 0 else 1.0
                    else:
                        progress = 1.0
                    coeff_schedule = math.sin(progress * math.pi / 2)  # Sine scheduler (0->1)
                else:
                    # Default to constant if unknown scheduler
                    coeff_schedule = 1.0
                # Decay factor for successive heads
                head_weight = (self.medusa_decay_coefficient ** i) * self.medusa_heads_coefficient * coeff_schedule
                total_loss += loss_i * head_weight
                log_metrics[f"medusa{i}_loss"] = loss_i.item()
            # Compute accuracy metrics for this head (top-1 to top-9 accuracy)
            not_ignore = target_flat.ne(IGNORE_TOKEN_ID)
            with torch.no_grad():
                # Compute top-9 predictions for accuracy metrics
                topk = min(9, pred_logits_flat.size(-1))
                topk_vals, topk_idx = pred_logits_flat.topk(topk, dim=-1)
                # Top-1 accuracy (same as original medusa{i}_acc)
                correct_top1 = topk_idx[:, 0].eq(target_flat)
                if correct_top1.numel() > 0:
                    acc1 = correct_top1.masked_select(not_ignore).float().mean().item()
                    log_metrics[f"medusa{i}_acc"] = acc1
                # Top-k (2 through 9) accuracies
                for k in range(2, 10):
                    if topk_idx.size(1) < k:
                        break  # if vocab smaller than k (unlikely)
                    hits_k = (topk_idx[:, :k].eq(target_flat.unsqueeze(-1))).any(dim=-1)
                    if hits_k.numel() > 0:
                        acc_k = hits_k.masked_select(not_ignore).float().mean().item()
                        log_metrics[f"medusa{i}_acc@{k}"] = acc_k
            # If self-distillation is enabled, compute KL divergence between base (teacher) and medusa head i
            if i > 0 and self.medusa_self_distillation and teacher_logits_full is not None:
                # Align teacher and student predictions for this head
                teacher_slice = teacher_logits_full[:, i:-1, :]  # base logits from index i to second-last
                student_slice = logits[i][:, :-(1 + i), :].detach()  # medusa head i logits (detach student for teacher calc)
                # Compute distribution for teacher and student
                B, T, V = teacher_slice.shape
                teacher_flat = teacher_slice.reshape(-1, V)
                student_log_flat = F.log_softmax(student_slice.reshape(-1, V), dim=-1)
                teacher_prob_flat = F.softmax(teacher_flat, dim=-1)
                # Mask out ignore tokens positions
                mask_flat = target_flat.ne(IGNORE_TOKEN_ID)
                # (Mask needs to align dimensions: target_flat corresponds to labels for these predictions)
                # Build mask for current head positions: target_flat already corresponds to head i labels after dropping initial tokens
                if mask_flat.any():
                    teacher_prob_masked = teacher_prob_flat[mask_flat]
                    student_log_masked = student_log_flat[mask_flat]
                    # KL divergence (teacher || student)
                    kl_div = F.kl_div(student_log_masked, teacher_prob_masked, reduction="batchmean")
                    # Weight KL similar to medusa head loss
                    kl_weighted = kl_div * (self.medusa_decay_coefficient ** i) * self.medusa_heads_coefficient
                    total_loss += kl_weighted
                    log_metrics[f"medusa{i}_kl"] = kl_div.item()
        LOG.debug(f"Medusa loss breakdown: {log_metrics}")
        # If enabled, log metrics to Weights & Biases (or via Trainer logging)
        if self.medusa_logging:
            try:
                import wandb
                # Include current training step in logging if available
                if self.state and self.state.global_step is not None:
                    wandb.log(log_metrics, step=self.state.global_step)
                else:
                    wandb.log(log_metrics)
            except ImportError:
                LOG.warning("Weights & Biases not installed; skipping medusa_logging.")
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
        base_model_name_or_path (str, optional): The name or path of the base model.
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """
    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        num_unfreezed_layers=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.num_unfreezed_layers = num_unfreezed_layers