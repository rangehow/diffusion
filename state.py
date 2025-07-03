from transformers.trainer_callback import TrainerState


class ExtraLogState(TrainerState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extra_data = {}
    
    def set_extra(self, key, value):
        """设置额外数据"""
        self.extra_data[key] = value
    
    def get_extra(self, key, default=None):
        """获取额外数据"""
        return self.extra_data.get(key, default)
    
    def update_extra(self, **kwargs):
        """批量更新额外数据"""
        self.extra_data.update(kwargs)