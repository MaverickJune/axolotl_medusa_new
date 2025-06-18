from axolotl.integrations.base import BasePlugin
from pathlib import Path
import torch
import json
from .trainer import AxolotlMedusaTrainer, add_medusa_heads, freeze_base_model, MedusaConfig

from axolotl.utils.logging import get_logger
LOG = get_logger(__name__, use_environ=True)

class MedusaPlugin(BasePlugin):
    """
    Plugin to integrate Medusa training into Axolotl.
    """
    def get_input_args(self):
        # Provide the MedusaArgs schema so config can include Medusa fields
        return "axolotl.integrations.medusa.MedusaArgs"

    def post_model_build(self, cfg, model):
        # After the base model is loaded (before LoRA applied), attach Medusa heads if enabled
        if getattr(cfg, "medusa", False) or cfg.get("medusa_num_heads"):
            num_heads = cfg.medusa_num_heads or 0
            if num_heads and num_heads > 0:
                add_medusa_heads(model, num_heads)
                # Set medusa_num_layers default if not provided
                model.config.medusa_num_heads = num_heads
                model.config.medusa_num_layers = getattr(cfg, "medusa_num_layers", 1) or 1
                # If training only heads (Medusa-1), freeze base model now (will unfreeze partial layers later if configured)
                if getattr(cfg, "medusa_train_only_heads", True):
                    freeze_base_model(model)
                    # Partially unfreeze top transformer layers if configured
                    if getattr(cfg, "medusa_num_unfreeze_layers", 0):
                        base = model
                        # If model is a Peft wrapper, get the underlying base model
                        if hasattr(model, "get_base_model"):
                            base = model.get_base_model()
                        # Determine transformer layers list
                        layers_list = None
                        total_layers = None
                        if hasattr(base, "model") and hasattr(base.model, "layers"):
                            layers_list = base.model.layers
                            total_layers = len(layers_list)
                        elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
                            layers_list = base.transformer.h
                            total_layers = len(layers_list)
                        elif hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
                            layers_list = base.gpt_neox.layers
                            total_layers = len(layers_list)
                        if layers_list and total_layers:
                            num_unfreeze = int(getattr(cfg, "medusa_num_unfreeze_layers", 0) or 0)
                            num_unfreeze = min(num_unfreeze, total_layers)
                            # Unfreeze the top `num_unfreeze` layers
                            for layer_idx in range(total_layers - num_unfreeze, total_layers):
                                for param in layers_list[layer_idx].parameters():
                                    param.requires_grad = True
                            LOG.info(f"Partially unfreezing top {num_unfreeze} transformer layers (out of {total_layers}) for Medusa training.")
                        else:
                            LOG.warning("Could not identify transformer layers for partial unfreeze; skipping medusa_num_unfreeze_layers.")
    
    def pre_lora_load(self, cfg, model):
        # If using LoRA/QLoRA and Medusa-1 mode, we might prevent LoRA from updating (optional)
        if getattr(cfg, "medusa", False) and getattr(cfg, "medusa_train_only_heads", True):
            # Optionally, do nothing here – we'll freeze LoRA params after they're loaded
            pass

    def post_lora_load(self, cfg, model):
        # After LoRA weights are loaded, freeze them if base is not being trained
        if getattr(cfg, "medusa", False) and getattr(cfg, "medusa_train_only_heads", True):
            # Freeze all LoRA adapter parameters
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = False
            base = model
            if hasattr(model, "get_base_model"):
                base = model.get_base_model()
            # Freeze base model parameters, with optional partial unfreeze for top layers
            if getattr(cfg, "medusa_num_unfreeze_layers", 0):
                # Partially freeze base: freeze all except top N layers
                layers_list = None
                total_layers = None
                if hasattr(base, "model") and hasattr(base.model, "layers"):
                    layers_list = base.model.layers
                    total_layers = len(layers_list)
                elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
                    layers_list = base.transformer.h
                    total_layers = len(layers_list)
                elif hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
                    layers_list = base.gpt_neox.layers
                    total_layers = len(layers_list)
                num_unfreeze = int(getattr(cfg, "medusa_num_unfreeze_layers", 0) or 0)
                if layers_list and total_layers:
                    num_unfreeze = min(num_unfreeze, total_layers)
                    # Freeze all layers below the top N
                    freeze_until = total_layers - num_unfreeze
                    for idx in range(total_layers):
                        for param in layers_list[idx].parameters():
                            if idx < freeze_until:
                                param.requires_grad = False
                            else:
                                # top layers: ensure requires_grad True
                                param.requires_grad = True
                    LOG.info(f"Frozen base model layers except for top {num_unfreeze} layers (partial unfreeze applied).")
                else:
                    # Fallback: freeze all non-medusa parameters except maybe final layers by name pattern
                    for name, p in base.named_parameters():
                        if "lora_" in name:
                            p.requires_grad = False
                        elif name.startswith("model.layers"):
                            # Attempt to parse layer index from name
                            try:
                                layer_idx = int(name.split("model.layers.")[1].split(".")[0])
                            except Exception:
                                layer_idx = None
                            if layer_idx is not None and layer_idx >= (0 if total_layers is None else total_layers - num_unfreeze):
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        elif name.startswith("transformer.h"):
                            try:
                                layer_idx = int(name.split("transformer.h.")[1].split(".")[0])
                            except Exception:
                                layer_idx = None
                            if layer_idx is not None and layer_idx >= (0 if total_layers is None else total_layers - num_unfreeze):
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        else:
                            # Freeze all other base params (embeddings, ln_f, lm_head, etc.)
                            p.requires_grad = False
                    LOG.info(f"Applied name-based partial unfreeze for top {num_unfreeze} layers.")
            else:
                # No partial unfreeze specified: freeze all base params (default Medusa-1 behavior)
                for n, p in base.named_parameters():
                    if "lora_" in n or not n.startswith("medusa_head"):
                        p.requires_grad = False

    def post_model_load(self, cfg, model):
        # Ensure Medusa heads are on the correct device/dtype after all loading is done
        if getattr(cfg, "medusa", False) and hasattr(model, "medusa_head"):
            # Align medusa_head modules to model's dtype/device (especially if moved to GPU or bf16)
            try:
                model.medusa_head.to(model.dtype)
            except Exception:
                # If model.dtype not set, fall back to parameter device
                for p in model.medusa_head.parameters():
                    p.data = p.data.to(next(model.parameters()).device)
            # Log final setup
            LOG.info(f"Medusa heads attached: train_only_heads={getattr(cfg, 'medusa_train_only_heads', True)}; dtype={next(model.medusa_head.parameters()).dtype}")

    def post_train(self, cfg, model):
        """
        After Trainer.train() finishes but before the model is unloaded.
        Save the Medusa heads + a tiny MedusaConfig for inference.
        """
        if not hasattr(model, "medusa_head"):
            LOG.warning("No medusa_head found – nothing to export.")
            return

        out_dir = Path(cfg.output_dir) / "medusa"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1️⃣ save the heads
        torch.save(model.medusa_head.state_dict(), out_dir / "medusa_lm_head.pt")

        # 2️⃣ save a MedusaConfig
        medusa_cfg = MedusaConfig(
            medusa_num_heads=model.medusa_num_heads,
            medusa_num_layers=model.medusa_num_layers,
            base_model_name_or_path=cfg.base_model,
            num_unfreezed_layers=int(getattr(cfg, "medusa_num_unfreeze_layers", 0) or 0),
        )
        medusa_cfg.save_pretrained(out_dir)

        LOG.info("Exported Medusa heads to %s", out_dir)
