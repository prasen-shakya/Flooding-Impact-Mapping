"""Hybrid: EvaNet's elevation gating on an ImageNet-pretrained ResNet50 U-Net.

This is NOT part of upstream EvaNet. It is this project's model, built from EvaNet's main idea
(elevation-regulated convolution: image features are multiplied per pixel by sigmoid(conv(elevation)))
and the same ResNet50 U-Net layout as notebooks/unet_dem.ipynb, so the two can be compared directly.

    RGB  -> ResNet50 (ImageNet weights, frozen) -> features at 1/2, 1/4, 1/8, 1/16
                                                    each multiplied by an elevation gate
    DEM  -> elevation stream (avg-pool + conv)  -> elevation features at 1, 1/2, 1/4, 1/8, 1/16
    decoder: transpose conv + skip + 2 gated convs per level (64 -> 512 channels, as in unet_dem)
    output: 2 channels with sigmoid, channel 0 = flood (EvaNet's format, for ElevationLoss)

Encoder skips match unet_dem.ipynb's Keras layer names:
    stem  (conv1_relu)        64 ch @ 1/2
    layer1 (conv2_block3_out) 256 ch @ 1/4
    layer2 (conv3_block4_out) 512 ch @ 1/8
    layer3 (conv4_block6_out) 1024 ch @ 1/16 (bridge)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# The ResNet50 weights expect RGB in 0-1 normalised with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OPEN_GATE_BIAS = 3.0  # sigmoid(3) = 0.95: gates on pretrained features start almost fully open


class ElevationPyramid(nn.Module):
    """DEM (B, 1, H, W) -> list of elevation features at full, 1/2, 1/4, 1/8 and 1/16 resolution.
    Average pooling between levels, like EvaNet's elevation stream."""

    def __init__(self, ch=16, levels=5):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1 if i == 0 else ch, ch, 3, padding=1, padding_mode="replicate"),
                          nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
            for i in range(levels)])

    def forward(self, dem):
        out, h = [], dem
        for i, level in enumerate(self.levels):
            if i:
                h = F.avg_pool2d(h, 2)
            h = level(h)
            out.append(h)
        return out


class EncoderGate(nn.Module):
    """x * sigmoid(1x1 conv(h)). Starts almost fully open, so the pretrained features pass through
    unchanged at first and the model learns how much elevation should hold them back."""

    def __init__(self, elev_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(elev_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, OPEN_GATE_BIAS)

    def forward(self, x, h):
        return x * torch.sigmoid(self.conv(h))


class GatedConv(nn.Module):
    """EvaNet's elevation-regulated convolution: relu(bn(conv(x) * sigmoid(conv(h))))."""

    def __init__(self, in_ch, out_ch, elev_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.elev = nn.Conv2d(elev_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, h):
        return F.relu(self.bn(self.conv(x) * torch.sigmoid(self.elev(h))))


class GatedDecoderBlock(nn.Module):
    """Same shape as unet_dem's decoder_block (2x transpose conv, concat skip, two 3x3 convs),
    with both convs gated by the elevation features at that resolution."""

    def __init__(self, in_ch, skip_ch, out_ch, elev_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = GatedConv(out_ch + skip_ch, out_ch, elev_ch)
        self.conv2 = GatedConv(out_ch, out_ch, elev_ch)

    def forward(self, x, skip, h):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.conv2(self.conv1(x, h), h)


class GatedResNetUNet(nn.Module):
    """ImageNet ResNet50 U-Net with EvaNet-style elevation gating.

    forward(rgb, norm_elev): rgb (B, 3, H, W) normalised with IMAGENET_MEAN/STD,
    norm_elev (B, 1, H, W) scaled to 0-1. H and W must be divisible by 16.
    Returns (B, n_classes, H, W) after a sigmoid; channel 0 is flood.
    """

    def __init__(self, n_classes=2, elev_ch=16, pretrained=True, freeze_encoder=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=weights)
        self.encoder = nn.ModuleDict({
            "stem": nn.Sequential(r.conv1, r.bn1, r.relu),  # 64 @ 1/2
            "pool": r.maxpool,
            "layer1": r.layer1,                             # 256 @ 1/4
            "layer2": r.layer2,                             # 512 @ 1/8
            "layer3": r.layer3,                             # 1024 @ 1/16
        })
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.elevation = ElevationPyramid(elev_ch)
        self.enc_gates = nn.ModuleList([EncoderGate(elev_ch, c) for c in (64, 256, 512, 1024)])

        self.dec1 = GatedDecoderBlock(1024, 512, 512, elev_ch)  # 1/16 -> 1/8
        self.dec2 = GatedDecoderBlock(512, 256, 256, elev_ch)   # 1/8 -> 1/4
        self.dec3 = GatedDecoderBlock(256, 64, 128, elev_ch)    # 1/4 -> 1/2
        self.dec4 = GatedDecoderBlock(128, 3, 64, elev_ch)      # 1/2 -> full, skip = the RGB input
        self.out = nn.Conv2d(64, n_classes, 1)

    def train(self, mode=True):
        """Keep the frozen encoder's BatchNorm in inference mode, like Keras does for trainable=False."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, rgb, norm_elev):
        h = self.elevation(norm_elev)  # h[0] full, h[1] 1/2, h[2] 1/4, h[3] 1/8, h[4] 1/16

        s2 = self.enc_gates[0](self.encoder["stem"](rgb), h[1])
        s3 = self.enc_gates[1](self.encoder["layer1"](self.encoder["pool"](s2)), h[2])
        s4 = self.enc_gates[2](self.encoder["layer2"](s3), h[3])
        b1 = self.enc_gates[3](self.encoder["layer3"](s4), h[4])

        d = self.dec1(b1, s4, h[3])
        d = self.dec2(d, s3, h[2])
        d = self.dec3(d, s2, h[1])
        d = self.dec4(d, rgb, h[0])
        return torch.sigmoid(self.out(d))
