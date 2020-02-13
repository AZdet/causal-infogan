import os
import numpy as np

from recursive_planning.datasets.data_loader import FolderSplitVarLenVideoDataset
from blox import AttrDict

length = 500

config = AttrDict(
    dense_rec_type='',
    img_sz=64,
    use_convs=True,
    randomize_length=False,
    dataset_spec=AttrDict(max_seq_len=length),
)


class GCPDataset:
    
    @staticmethod
    def dataset_size(phase):
        return -1 if phase == 'train' else 32
    
    @staticmethod
    def getitem(mode, inputs):
        if mode == 'train':
            ind = int(np.random.rand(1) * inputs.end_ind)
            return [inputs.demo_seq[ind], 0], \
                   [inputs.demo_seq[ind + 1], 0]
        elif mode == 'plan':
            return inputs.demo_seq[:inputs.end_ind + 1]
        elif mode == 'start':
            return inputs.demo_seq[0], 0
        elif mode == 'goal':
            return inputs.demo_seq[-1], 0


class Dataset(FolderSplitVarLenVideoDataset):
    
    def __init__(self, phase, mode):
        self.mode = mode
        super().__init__(os.environ['RECPLAN_DATA_DIR'] + '/h36m_long',
                         config, config, phase=phase, dataset_size=GCPDataset.dataset_size(phase))
    
    def __getitem__(self, item):
        inputs = super().__getitem__(item)
        
        return GCPDataset.getitem(self.mode, inputs)
        
        
    