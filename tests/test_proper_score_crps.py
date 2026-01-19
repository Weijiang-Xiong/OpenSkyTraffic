import numpy as np
from properscoring import crps_ensemble, crps_gaussian
from einops import repeat

def normal_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

reference = crps_gaussian(x=0.0, mu=0.0, sig=1.0)
print("CRPS for standard normal distribution:", reference)

ensemble = np.random.RandomState(0).randn(100_000).reshape(10, -1)
crps_samples = crps_ensemble(np.zeros(10), ensemble)
print("CRPS from samples:", crps_samples)
print("Mean CRPS from samples:", np.mean(crps_samples))

x = repeat(np.linspace(-100, 100, num=1000), 'n -> m n', m=10)
crps_from_weight = crps_ensemble(np.zeros(10), x, weights=normal_pdf(x))
print("CRPS from weighted values:", crps_from_weight)
print("Mean CRPS from weighted values:", np.mean(crps_from_weight))