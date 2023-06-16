from .trainer import HookBase, DefaultTrainer

class UncertaintyTraining(HookBase):
    
    
    def __init__(self) -> None:
        super().__init__()
    
    def before_train(self, trainer: DefaultTrainer):
        return super().before_train(trainer)