class HookBase:
    
    def __init__(self) -> None:
        pass
    
    def before_train(self, trainer):
        raise NotImplementedError
    
    def after_train(self, trainer):
        raise NotImplementedError
    
    def before_epoch(self, trainer):
        raise NotImplementedError
    
    def after_epoch(self, trainer):
        raise NotImplementedError

    def before_step(self, trainer):
        raise NotImplementedError
    
    def after_step(self, trainer):
        raise NotImplementedError
    
    