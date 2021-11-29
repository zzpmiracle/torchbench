import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from .RobustVideoMatting.inference_utils import VideoReader, VideoWriter

from .RobustVideoMatting.inference import auto_downsample_ratio
from .train_utils import train_mat, train_seg
from .args import stage1_args

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(BenchmarkModel):
    task = COMPUTER_VISION.OTHER_COMPUTER_VISION
    # Source: https://github.com/PeterL1n/RobustVideoMatting/blob/aff79bfdc056daf478bd395a838d37c5d715a7e2/train.py#L154
    # --batch-size-per-gpu is 1 for default
    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=1):
        super().__init__()
        self.device = device
        self.jit = jit

        # We only run stage 1 for benchmarking
        args = stage1_args()
        args.train_batch_size = train_bs
        args.eval_batch_size = eval_bs

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        for epoch in range(niter):
            if not self.args.disable_validation:
                self.validate()
            for _, (true_fgr, true_pha, true_bgr) in zip(range(self.train_num_batch), self.dataloader_lr_train):
                # Low resolution pass
                train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=1, tag='lr')
                # High resolution pass
                if self.args.train_hr:
                    true_fgr, true_pha, true_bgr = self.load_next_mat_hr_sample()
                    train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=self.args.downsample_ratio, tag='hr')
                
                # Segmentation pass
                if self.step % 2 == 0:
                    true_img, true_seg = self.load_next_seg_video_sample()
                    train_seg(true_img, true_seg, log_label='seg_video')
                else:
                    true_img, true_seg = self.load_next_seg_image_sample()
                    train_seg(true_img.unsqueeze(1), true_seg.unsqueeze(1), log_label='seg_image')
                
                self.step += 1
                    
    def eval(self, niter=1):
        self.eval_model.eval()
        with torch.no_grad():
            for epoch in range(niter):
                for _, src in zip(range(self.eval_num_batch), reader):
                    if downsample_ratio is None:
                        downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                    # src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                    fgr, pha, *rec = self.eval_model(src, *rec, downsample_ratio)