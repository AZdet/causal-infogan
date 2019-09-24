import numpy as np
import torch
from recursive_planning.evaluation.compute_metrics import psnr
from recursive_planning.evaluation.ssim_metric import ssim


class EvalPSNR(object):
    def __init__(self, max_level):
        self.max_level = max_level
        self.clear()
    
    def __call__(self, pred, gt, mask=None):
        reshape = lambda x: x.reshape([x.shape[0], -1, 3] + list(x.shape[-2:]))
        pred, gt = reshape(pred), reshape(gt)
        
        for i in range(pred.shape[0]):
            self.psnr += psnr(pred[i], gt[i])
            self.ssim += ssim(torch.from_numpy(pred[i]).cuda(), torch.from_numpy(gt[i]).cuda())
            self.count += 1
    
    def PSNR(self):
        return self.psnr / max(1, self.count)
    
    def SSIM(self):
        return self.ssim / max(1, self.count)
    
    def clear(self):
        self.psnr = 0
        self.ssim = 0
        self.count = 0
