"""
    follow the logic in detectron2, but not as complete as it
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/events.py
    A difference is, the training in detectron2 is counted by iteration, and there is 
    no concept of epoch there. 是啊
"""
import copy
import traceback 
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np

_STORAGE_STACK = []

class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def data(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data
    
    def values(self) -> List[float]:
        """return a list of recorded value, without iteration number
        """
        return [x for (x, i) in self._data]
    
def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _STORAGE_STACK[-1]


class EventStorage:
    """ This is used to store training log for visualization or summary purpose
        For example: 
            1. the training loss for each batch
            2. average training loss for each epoch
            3. auxiliary metrics or other variable of interests
    """
    
    def __init__(self, start_iter=0) -> None:
        self._iteration = start_iter
        self._history = defaultdict(HistoryBuffer)
        self._latest_scalars = dict()
        
    def __enter__(self):
        _STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert _STORAGE_STACK[-1] == self
        _STORAGE_STACK.pop()
        
        if exc_type is not None:
            print("Encountered Exception: {} {}".format(exc_type, exc_value))
            print("Traceback:")
            traceback.print_tb(exc_traceback)
            
    def put_scalar(self, name, value, suffix=None):
        """ update both the 
        """
        if suffix is not None:
            name += '_{}'.format(suffix)
            
        self._latest_scalars[name] = (value, self.iteration)
        self._history[name].update(float(value), self._iteration)

    def put_scalars(self, suffix=None, **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(name=k, value=v, suffix=suffix)
            
    def latest(self):
        return self._latest_scalars
            
    @property
    def iteration(self):
        return self._iteration
    
    @iteration.setter
    def iteration(self, val):
        self._iteration = int(val)
    
    def step(self):
        self._iteration += 1
        
    def __getitem__(self, key):
        return self._history[key]