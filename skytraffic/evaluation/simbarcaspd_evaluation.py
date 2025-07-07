from ..data.datasets import SimBarcaSpeed
from .metr_evaluation import MetrEvaluator
from .metr_gmm_evaluation import MetrGMMEvaluator

class SimBarcaSpeedEvaluator(MetrEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def common_metrics_by_horizon(self, pred, label, ignore_value=SimBarcaSpeed.invalid_value, mape_threshold=1.0, verbose: bool = False):
        return super().common_metrics_by_horizon(pred, label, ignore_value, mape_threshold, verbose)

class SimBarcaSpeedGMMEvaluator(MetrGMMEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def common_metrics_by_horizon(self, pred, label, ignore_value=SimBarcaSpeed.invalid_value, mape_threshold=1.0, verbose: bool = False):
        return super().common_metrics_by_horizon(pred, label, ignore_value, mape_threshold, verbose)