from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.data.datasets import SimBarcaMSMT, SimBarcaRandomObservation, SimBarcaSpeed, MetrDataset, PEMSBayDataset

metrla = OmegaConf.create()

metrla.train = L(MetrDataset)(
    split="train",
    adj_type="doubletransition"
)

metrla.val = L(MetrDataset)(
    split="val",
    adj_type="doubletransition"
)

metrla.test = L(MetrDataset)(
    split="test",
    adj_type="doubletransition"
)


pemsbay = OmegaConf.create()

pemsbay.train = L(PEMSBayDataset)(
    split="train",
    adj_type="doubletransition"
)

pemsbay.val = L(PEMSBayDataset)(
    split="val",
    adj_type="doubletransition"
)

pemsbay.test = L(PEMSBayDataset)(
    split="test",
    adj_type="doubletransition"
)

simbarcaspd = OmegaConf.create()
simbarcaspd.train = L(SimBarcaSpeed)(split="train", input_nan_to_global_avg=True)
simbarcaspd.test = L(SimBarcaSpeed)(split="test", input_nan_to_global_avg=True)

simbarca_msmt = OmegaConf.create()

simbarca_msmt.train = L(SimBarcaMSMT)(
    split="train", 
    input_window=30, 
    pred_window=30, 
    step_size=3, 
    sample_per_session=20
)

simbarca_msmt.test = L(SimBarcaMSMT)(
    split="test", 
    input_window=30, 
    pred_window=30, 
    step_size=3, 
    sample_per_session=20
)

simbarca_rnd = OmegaConf.create()

simbarca_rnd.train = L(SimBarcaRandomObservation)(
    split="train", 
    ld_cvg=0.1, 
    drone_cvg=0.1, 
    reinit_pos=False, 
    mask_seed=42, 
    use_clean_data=False, 
    noise_seed=114514, 
    drone_noise=0.05, 
    ld_noise=0.15,
    input_window=30, 
    pred_window=30, 
    step_size=3, 
    sample_per_session=20
)

simbarca_rnd.test = L(SimBarcaRandomObservation)(
    split="test", 
    ld_cvg=0.1, 
    drone_cvg=0.1,
    reinit_pos=False,
    mask_seed=42,
    use_clean_data=False,
    noise_seed=114514,
    drone_noise=0.05,
    ld_noise=0.15,
    input_window=30, 
    pred_window=30, 
    step_size=3, 
    sample_per_session=20
)