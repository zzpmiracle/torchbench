from torch.cuda.amp import autocast
from .RobustVideoMatting.train_loss import matting_loss, segmentation_loss

def train_mat(self, true_fgr, true_pha, true_bgr, downsample_ratio, tag):
    true_fgr = true_fgr.to(self.rank, non_blocking=True)
    true_pha = true_pha.to(self.rank, non_blocking=True)
    true_bgr = true_bgr.to(self.rank, non_blocking=True)
    true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
    true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
    
    with autocast(enabled=not self.args.disable_mixed_precision):
        pred_fgr, pred_pha = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

    self.scaler.scale(loss['total']).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()

def train_seg(self, true_img, true_seg, log_label):
    true_img = true_img.to(self.rank, non_blocking=True)
    true_seg = true_seg.to(self.rank, non_blocking=True)
    
    true_img, true_seg = self.random_crop(true_img, true_seg)
    
    with autocast(enabled=not self.args.disable_mixed_precision):
        pred_seg = self.model_ddp(true_img, segmentation_pass=True)[0]
        loss = segmentation_loss(pred_seg, true_seg)
    
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()