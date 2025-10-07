"""
This script is called manual test because we can't automatically check the dataset,
one have to manually plot the results and do some sanity checks.
"""

import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from skytraffic.data.datasets import SimBarcaMSMT
from skytraffic.utils.event_logger import setup_logger
logger = setup_logger(name="default", level=logging.INFO)

def visualize_batch(data_dict, pred_dict=None, save_dir="./", batch_num=0, section_num=573, save_note="example"):
    # plot input and output 
    b, s = batch_num, section_num
    cluster_id = data_dict['metadata']['cluster_id']
    drone_in = data_dict['drone_speed'].cpu().numpy()
    ld_in = data_dict['ld_speed'].cpu().numpy()

    label = data_dict['pred_speed'].cpu().numpy()
    label_regional = data_dict['pred_speed_regional'].cpu().numpy()
    in1 = drone_in[b, :, s, 0]
    tin1 = np.linspace(0, 30, len(in1))
    in2 = ld_in[b, :, s, 0]
    tin2 = np.linspace(0, 30, len(in2))
    label1 = label[b, :, s]
    tlabel1 = np.linspace(33, 60, len(label1))
    label2 = label_regional[b, :, cluster_id[s]]
    tlabel2 = np.linspace(33, 60, len(label2))
    
    # draw the model predictions if available
    if pred_dict is not None:
        pred = pred_dict['pred_speed'].cpu().numpy()
        pred_regional = pred_dict['pred_speed_regional'].cpu().numpy()
        out1 = pred[b, :, s]
        tout1 = np.linspace(33, 60, len(out1))
        out2 = pred_regional[b, :, cluster_id[s]]
        tout2 = np.linspace(33, 60, len(out2))
        
    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(tin1, in1, label="Drone Input")
    ax.plot(tin2, in2, label="LD Input")
    ax.plot(tlabel1, label1, label="Segment Label")
    ax.plot(tlabel2, label2, label="Regional Label")
    
    try:
        ax.plot(tout1, out1, label="Segment Pred")
        ax.plot(tout2, out2, label="Regional Pred")
    except Exception as e:
        logger.info("No prediction data available: {}".format(e))
    
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("{}/pred_sample_b{}s{}_{}.pdf".format(save_dir, b, s, save_note))
    plt.close()
    logger.info("Saved the plot to {}/pred_sample_b{}s{}_{}.pdf".format(save_dir, b, s, save_note))

def plot_label_scatter(data_sequence:torch.Tensor, save_dir="./", section_num=100, save_note="example"):
    
    p = section_num
    sample_per_session = 20
    total_num_session = int(data_sequence.shape[0] / sample_per_session)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    pred_speed_sec = data_sequence[:, -1, p, 0].split(sample_per_session)
    xx = np.arange(sample_per_session)
    for s in pred_speed_sec:
        ax.scatter(xx, s, alpha=0.5, s=3)
    ax.set_xticks(np.arange(0, sample_per_session+1, 5))
    ax.set_xlabel("Time Step (per 3 mins)")
    ax.set_ylabel("Speed (m/s)")
    fig.tight_layout()
    fig.savefig("{}/section_30min_label_clusters_{}_{}.pdf".format(save_dir, p, save_note))

class TestSimBarca(unittest.TestCase):

    def test_full_data_loading(self):
        test_set = SimBarcaMSMT(split="test")
        batch = test_set.collate_fn([test_set[0], test_set[1]])
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=test_set.collate_fn)
        for data_dict in test_loader:
            visualize_batch(data_dict, save_note="test")
            break

        test_set = test_set
        section_id_to_index = {v:k for k, v in test_set.index_to_section_id.items()}
        batch = test_set.collate_fn([test_set[450], test_set[450]])
        visualize_batch(batch, section_num=section_id_to_index[9971])
        plot_label_scatter(test_set.pred_speed, section_num=section_id_to_index[9971])
        for i in range(4):
            plot_label_scatter(test_set.pred_speed_regional, section_num=i, save_note="region")
    
if __name__ == "__main__":
    unittest.main()