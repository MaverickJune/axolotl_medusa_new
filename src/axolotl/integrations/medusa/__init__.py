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
                # If training only heads (Medusa-1), freeze base model now
                if getattr(cfg, "medusa_train_only_heads", True):
                    freeze_base_model(model)

    def pre_lora_load(self, cfg, model):
        # If using LoRA/QLoRA and Medusa-1 mode, we might prevent LoRA from updating (optional)
        if getattr(cfg, "medusa", False) and getattr(cfg, "medusa_train_only_heads", True):
            # Optionally, do nothing here – we'll freeze LoRA params after they're loaded
            pass

    def post_lora_load(self, cfg, model):
        # After LoRA weights are loaded, freeze them if base is not being trained
        if getattr(cfg, "medusa", False) and getattr(cfg, "medusa_train_only_heads", True):
            # Freeze any LoRA adapter parameters so they won't train
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = False
            # If model is a PeftModel wrapper, also mark underlying base as frozen just in case
            if hasattr(model, "get_base_model"):
                base = model.get_base_model()
                for n, p in base.named_parameters():
                    if "lora_" in n or not n.startswith("medusa_head"):
                        p.requires_grad = False

    def post_model_load(self, cfg, model):
        # Ensure Medusa heads are on the correct device/dtype after all loading is done
        if getattr(cfg, "medusa", False) and hasattr(model, "medusa_head"):
            # Align medusa_head modules to model's dtype/device (especially if moved to GPU or bf16)
            try:
                model.medusa_head.to(model.dtype)  # set dtype (device should already match if model moved)
            except Exception:
                pass  # model.dtype may not exist, alternatively iterate submodules:
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
        torch.save(model.medusa_head.state_dict(),
                out_dir / "medusa_lm_head.pt")

        # 2️⃣ save a MedusaConfig
        medusa_cfg = MedusaConfig(
            medusa_num_heads=model.medusa_num_heads,
            medusa_num_layers=model.medusa_num_layers,
            base_model_name_or_path=cfg.base_model,
        )
        medusa_cfg.save_pretrained(out_dir)

        LOG.info("Exported Medusa heads to %s", out_dir)
    
    def get_trainer_cls(self, cfg):
        # If Medusa training is enabled, use the custom trainer
        if getattr(cfg, "medusa", False) or getattr(cfg, "medusa_num_heads", 0) > 0:
            return AxolotlMedusaTrainer
        return None
