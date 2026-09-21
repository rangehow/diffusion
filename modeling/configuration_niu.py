from transformers import ModernBertConfig


class NiuConfig(ModernBertConfig):
    model_type = "niu"

    def __init__(
        self,
        # ... other parameters
        use_token_change_task: bool = False,  # <-- 新增的开关
        residual_dropout=0.0,
        **kwargs
    ):
        # ...
        self.use_token_change_task = use_token_change_task
        self.mask_token_id = 50284
        self.residual_dropout = residual_dropout
        super().__init__(**kwargs)

    # Note: In transformers 4.57.x, global_rope_theta and local_rope_theta
    # are still normal instance attributes in ModernBertConfig.__init__.
    # Do NOT add @property shims — they block the parent's attribute assignment.