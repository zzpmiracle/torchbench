""" Full assembly of the parts to form the complete network """

from .unet_parts import *
try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

class UNet(nn.Module):

    # benckmarkmodel.eval() would use this attribute, which will be lost in torch.jit.trace
    # so marked as final to keep this attribute
    # according to https://pytorch.org/docs/stable/jit.html#attributes
    n_classes: Final[int]

    def __init__(self, n_channels, n_classes, bilinear=True, is_eval=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if self.is_eval:
            # only for eval mode
            if self.n_classes == 1:
                logits = (F.sigmoid(logits) > 0.5).float()
            else:
                logits = F.one_hot(logits.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
        return logits
