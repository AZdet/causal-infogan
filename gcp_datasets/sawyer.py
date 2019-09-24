import os
import random
import numpy as np
import torch

from recursive_planning.datasets.data_loader import FolderSplitVarLenVideoDataset, GlobalSplitVarLenVideoDataset
from recursive_planning.utils.pytorch_utils import combine_dim
from recursive_planning.utils.general_utils import batchwise_index
from recursive_planning.utils.utils import AttrDict
from vpa.gcp_datasets.h36m import GCPDataset

length = 80

config = AttrDict(
    dense_rec_type='',
    img_sz=64,
    use_convs=True,
    randomize_length=False,
    dataset_spec=AttrDict(max_seq_len=length),
)


class Dataset(FolderSplitVarLenVideoDataset):
    
    def __init__(self, phase, mode):
        self.mode = mode
        super().__init__(os.environ['RECPLAN_DATA_DIR'] + '/sim/sawyer/wiggle',
                         config, config, phase=phase, dataset_size=GCPDataset.dataset_size(phase))
        #dataset_size=32)
    
    def __getitem__(self, item):
        inputs = super().__getitem__(item)
        
        return GCPDataset.getitem(self.mode, inputs)
        
        
    