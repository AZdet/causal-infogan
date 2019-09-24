import os
import random
import numpy as np
import torch

from recursive_planning.datasets.data_loader import FolderSplitVarLenVideoDataset, GlobalSplitVarLenVideoDataset
from recursive_planning.utils.pytorch_utils import combine_dim
from recursive_planning.utils.general_utils import batchwise_index
from recursive_planning.utils.utils import AttrDict
from vpa.gcp_datasets.h36m import GCPDataset

length = 100

config = AttrDict(
    dense_rec_type='',
    img_sz=64,
    use_convs=True,
    randomize_length=False,
    dataset_spec=AttrDict(max_seq_len=length, split=AttrDict(train=0.994, val=0.003, test=0.003)),
)


class Dataset(GlobalSplitVarLenVideoDataset):
    
    def __init__(self, phase, mode):
        self.mode = mode
        super().__init__('/parent/nfs/kun1/users/kpertsch/data/recplan_data/sim/nav3d_R9_L100/',
                         config, config, phase=phase, dataset_size=GCPDataset.dataset_size(phase))

    def __getitem__(self, item):
        inputs = super().__getitem__(item)

        return GCPDataset.getitem(self.mode, inputs)
