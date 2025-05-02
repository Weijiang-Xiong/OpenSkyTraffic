import unittest
import logging

from torch.utils.data import DataLoader

from netsanut.data.datasets import SimBarca
from netsanut.utils.event_logger import setup_logger
logger = setup_logger(name="default", level=logging.INFO)

class TestSimBarca(unittest.TestCase):

    def test_full_data_loading(self):
        test_set = SimBarca(split="test", force_reload=False)
        batch = test_set.collate_fn([test_set[0], test_set[1]])
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=test_set.collate_fn)
        for data_dict in test_loader:
            test_set.visualize_batch(data_dict, save_note="test")
            break

        test_set = test_set
        section_id_to_index = {v:k for k, v in test_set.index_to_section_id.items()}
        batch = test_set.collate_fn([test_set[450], test_set[450]])
        test_set.visualize_batch(batch, section_num=section_id_to_index[9971])
        test_set.plot_label_scatter(section_num=section_id_to_index[9971], regional=False)
        for i in range(4):
            test_set.plot_label_scatter(section_num=i, regional=True, save_note="region")
    
if __name__ == "__main__":
    unittest.main()