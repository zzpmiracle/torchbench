import random

from torch.nn import functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import center_crop

from .RobustVideoMatting.model import MattingNetwork
from .RobustVideoMatting.train_loss import matting_loss, segmentation_loss

def random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(w // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results

def init_model(args, rank=0):
    model = MattingNetwork(args.model_variant, pretrained_backbone=True)
    model = model.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_ddp = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
    optimizer = Adam([
            {'params': model.backbone.parameters(), 'lr': args.learning_rate_backbone},
            {'params': model.aspp.parameters(), 'lr': args.learning_rate_aspp},
            {'params': model.decoder.parameters(), 'lr': args.learning_rate_decoder},
            {'params': model.project_mat.parameters(), 'lr': args.learning_rate_decoder},
            {'params': model.project_seg.parameters(), 'lr': args.learning_rate_decoder},
            {'params': model.refiner.parameters(), 'lr': args.learning_rate_refiner},
        ])
    scaler = GradScaler()
    return model_ddp, optimizer, scaler

def train_mat(args, model_ddp, scaler, optimizer, true_fgr, true_pha, true_bgr, downsample_ratio, tag, rank=0):
    true_fgr = true_fgr.to(rank, non_blocking=True)
    true_pha = true_pha.to(rank, non_blocking=True)
    true_bgr = true_bgr.to(rank, non_blocking=True)
    true_fgr, true_pha, true_bgr = random_crop(true_fgr, true_pha, true_bgr)
    true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
    
    with autocast(enabled=not args.disable_mixed_precision):
        pred_fgr, pred_pha = model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
        loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

    scaler.scale(loss['total']).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

def train_seg(args, model_ddp, scaler, optimizer, true_img, true_seg, log_label, rank=0):
    true_img = true_img.to(rank, non_blocking=True)
    true_seg = true_seg.to(rank, non_blocking=True)
    
    true_img, true_seg = random_crop(true_img, true_seg)
    
    with autocast(enabled=not args.disable_mixed_precision):
        pred_seg = model_ddp(true_img, segmentation_pass=True)[0]
        loss = segmentation_loss(pred_seg, true_seg)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()