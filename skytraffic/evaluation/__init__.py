from .metr_evaluation import MetrEvaluator
from .metr_gmm_evaluation import MetrGMMEvaluator
from .simbarca_evaluation import SimBarcaEvaluator
from .simbarca_gmm_evaluation import SimBarcaGMMEvaluator
from .simbarcaspd_evaluation import SimBarcaSpeedEvaluator, SimBarcaSpeedGMMEvaluator

def build_evaluator(evaluator_type, **kwargs):
    
    match evaluator_type:
        case 'simbarca':
            return SimBarcaEvaluator(**kwargs)
        case 'simbarcagmm':
            return SimBarcaGMMEvaluator(**kwargs)
        case 'metrla' | 'pemsbay':
            return MetrEvaluator(**kwargs)
        case 'metrlagmm':
            return MetrGMMEvaluator(**kwargs)
        case 'simbarcaspd':
            return SimBarcaSpeedEvaluator(**kwargs)
        case 'simbarcaspdgmm':
            return SimBarcaSpeedGMMEvaluator(**kwargs)
        case _:
            raise ValueError('No evaluator is specified')