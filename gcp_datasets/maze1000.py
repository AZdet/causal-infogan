import os
import random
import numpy as np
import torch

from recursive_planning.datasets.data_loader import LengthFilteredVarLenVideoDataset
from recursive_planning.utils.pytorch_utils import combine_dim
from recursive_planning.utils.general_utils import batchwise_index
from recursive_planning.utils.utils import AttrDict
from vpa.gcp_datasets.h36m import GCPDataset

length = 1000

config = AttrDict(
    dense_rec_type='',
    img_sz=16,
    use_convs=True,
    randomize_length=False,
    dataset_spec=AttrDict(max_seq_len=length,
                          split=[([800, 1000], AttrDict(train=0.98, val=0.01, test=0.01))]),
)


class Dataset(LengthFilteredVarLenVideoDataset):
    
    def __init__(self, phase, mode):
        self.mode = mode
        super().__init__('/parent/nfs/kun1/users/kpertsch/data/recplan_data/sim/nav3d_R100L1000_mod/',
                         config, config, phase=phase, dataset_size=GCPDataset.dataset_size(phase))

    def __getitem__(self, item):
        inputs = super().__getitem__(item)

        return GCPDataset.getitem(self.mode, inputs)
