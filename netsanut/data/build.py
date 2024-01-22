import torch
from typing import List, Tuple, Dict


def tensor_collate(list_of_xy:List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    """ assume the input is a list of (x, y), pack the x's and y's into two tensors
    """
    xs = [torch.as_tensor(xy[0]).unsqueeze(0) for xy in list_of_xy]
    ys = [torch.as_tensor(xy[1]).unsqueeze(0) for xy in list_of_xy]
    
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

def to_contiguous(data:torch.Tensor, label:torch.Tensor) -> Tuple[torch.Tensor]:
    
    return {"source": data.contiguous(), "target": label[..., 0].contiguous()}

tensor_to_contiguous = lambda list_of_xy: to_contiguous(*tensor_collate(list_of_xy))