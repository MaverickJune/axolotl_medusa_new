from typing import Optional
from pydantic import BaseModel

from typing import Optional
from pydantic import BaseModel

class MedusaArgs(BaseModel):
    """
    Configuration arguments for Medusa training.
    """
    medusa: Optional[bool] = None  # Enable Medusa training if True
    medusa_num_heads: Optional[int] = None  # Number of Medusa decoding heads to add
    medusa_num_layers: Optional[int] = None  # (Reserved) Number of transformer layers to attach heads (default 1)
    medusa_train_only_heads: Optional[bool] = None  # If True, train only Medusa heads (Medusa-1); base model is frozen
    medusa_heads_coefficient: Optional[float] = None  # Loss coefficient for Medusa heads
    medusa_decay_coefficient: Optional[float] = None  # Decay factor for loss of successive Medusa heads
    medusa_scheduler: Optional[str] = None  # Medusa loss scheduler type (e.g. "constant")
    medusa_lr_multiplier: Optional[float] = None  # LR multiplier for Medusa head parameters
    # New advanced Medusa features (backward compatible)
    medusa_self_distillation: Optional[bool] = None  # If True, enable self-distillation (KL alignment to base outputs)
    medusa_distillation_regularization: Optional[bool] = None  # If True, use KL loss with smoothed target (label smoothing) for base head
    medusa_logging: Optional[bool] = None  # If True, enable additional Medusa metrics logging (e.g. W&B logging)
    medusa_num_unfreeze_layers: Optional[int] = None  # Number of top transformer layers to unfreeze (partial base training)

