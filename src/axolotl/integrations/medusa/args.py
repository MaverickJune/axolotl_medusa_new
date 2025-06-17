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
