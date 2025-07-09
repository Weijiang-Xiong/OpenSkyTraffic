from ..data.datasets import SimBarcaSpeed
from .metr_evaluation import MetrEvaluator
from .metr_gmm_evaluation import MetrGMMEvaluator

class SimBarcaSpeedEvaluator(MetrEvaluator):

    ignore_value = SimBarcaSpeed.data_null_value
    mape_threshold = 1.0


class SimBarcaSpeedGMMEvaluator(MetrGMMEvaluator):
    
    ignore_value = SimBarcaSpeed.data_null_value
    mape_threshold = 1.0