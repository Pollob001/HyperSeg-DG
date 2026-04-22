import math
import torch
import torch.nn as nn
from network.wmamba import wmamba
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv1(attn)
        attn = self.sigmoid(attn)
        return x * attn


class FeatureRefinement(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = CBR(out_c, out_c, kernel_size=3, padding=1)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ca(x)
        x = self.sa(x)
        return x


class ProgressiveDecoder(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.conv1 = CBR(in_c + out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = CBR(out_c, out_c, kernel_size=3, padding=1)
        self.conv3 = CBR(out_c, out_c, kernel_size=3, padding=1)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
        
    def forward(self, x, skip):
        x = self.up(x)
        # Ensure skip connection matches the upsampled size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ca(x)
        x = self.sa(x)
        return x


class EnhancedFusionHead(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4, out_c):
        super().__init__()
        # Feature refinement for each level
        self.refine1 = FeatureRefinement(in_c1, 128)
        self.refine2 = FeatureRefinement(in_c2, 128)
        self.refine3 = FeatureRefinement(in_c3, 128)
        self.refine4 = FeatureRefinement(in_c4, 128)
        
        # Progressive decoder
        self.decoder4 = ProgressiveDecoder(128, 128, scale=2)
        self.decoder3 = ProgressiveDecoder(128, 128, scale=2)
        self.decoder2 = ProgressiveDecoder(128, 128, scale=2)
        
        # Final prediction head with dynamic upsampling
        self.final_conv = nn.Sequential(
            CBR(128, 64, kernel_size=3, padding=1),
            CBR(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        # Refine features
        f1 = self.refine1(x1)  # H/4 × W/4
        f2 = self.refine2(x2)  # H/8 × W/8  
        f3 = self.refine3(x3)  # H/16 × W/16
        f4 = self.refine4(x4)  # H/32 × W/32
        
        # Progressive decoding with skip connections
        d4 = self.decoder4(f4, f3)  # H/16 × W/16
        d3 = self.decoder3(d4, f2)  # H/8 × W/8
        d2 = self.decoder2(d3, f1)  # H/4 × W/4
        
        # Dynamic upsampling to match input size
        target_size = (x1.shape[2] * 4, x1.shape[3] * 4)  # From H/4 to H
        d2 = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=True)
        
        return self.final_conv(d2)


class HyperSegStage1(nn.Module):
    def __init__(self, pretrained_img_size=256, backbone_pretrained_path=None):
        super().__init__()
        
        backbone = wmamba(
            pretrained=True,
            img_size=pretrained_img_size,
            pretrained_path=backbone_pretrained_path
        )
        self.layer0 = backbone.layer0  # [batch_size, 128, h/4, w/4]
        self.layer1 = backbone.layer1  # [batch_size, 256, h/8, w/8]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/16, w/16]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/32, w/32]
        self.head = EnhancedFusionHead(128, 256, 512, 1024, 1)

    def forward(self, image):
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 128, h/4, w/4]
        x2 = self.layer1(x1)  ## [-1, 256, h/8, w/8]
        x3 = self.layer2(x2)  ## [-1, 512, h/16, w/16]
        x4 = self.layer3(x3)  ## [-1, 1024, h/32, w/32]

        # Enhanced fusion with progressive decoder
        pred = self.head(x1, x2, x3, x4)

        return pred


if __name__ == "__main__":
    model = HyperSegStage1().cuda()
    
    # Test with 256×256
    input_tensor_256 = torch.randn(1, 3, 256, 256).cuda()
    output_256 = model(input_tensor_256)
    print(f"256×256 input -> output shape: {output_256.shape}")
    
    # Test with 512×512  
    input_tensor_512 = torch.randn(1, 3, 512, 512).cuda()
    output_512 = model(input_tensor_512)
    print(f"512×512 input -> output shape: {output_512.shape}")
