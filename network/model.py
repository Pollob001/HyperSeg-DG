import math
import torch
import torch.nn as nn
from network.wmamba import wmamba_b
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


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x


"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


"""Auxiliary Head"""


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        
        # Calculate the number of upsampling steps needed to reach 256x256
        # Since input is 8x8 from SwinMamba, we need 5 upsampling steps (8→16→32→64→128→256)
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 8x8 → 16x16
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 16x16 → 32x32
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 32x32 → 64x64
            CBR(128, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 64x64 → 128x128
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 128x128 → 256x256
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 8x8 → 16x16
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 16x16 → 32x32
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 32x32 → 64x64
            CBR(128, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 64x64 → 128x128
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 128x128 → 256x256
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 8x8 → 16x16
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 16x16 → 32x32
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 32x32 → 64x64
            CBR(128, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 64x64 → 128x128
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 128x128 → 256x256
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)

        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.fuse=CBR(in_c*3,in_c, kernel_size=3, padding=1)
        self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
        self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x2 = self.up_2x2(x2)
        x3 = self.up_4x4(x3)

        x = torch.cat([x1, x2, x3], axis=1)
        x=self.fuse(x)

        x=self.up_4x4(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x


class multiscale_feature_aggregation(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = CBR(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = CBR(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = CBR(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = CBR(out_c * 3, out_c, kernel_size=1, padding=0)

        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        return x


class CDFAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x


class LightweightDynamicRelationModule(nn.Module):
    """Memory-efficient version with fixed tensor dimensions"""
    def __init__(self, channels, reduction=8, num_scales=2):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        
        # Fewer dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1*d, dilation=1*d) 
            for d in range(1, num_scales+1)
        ])
        
        # Simplified attention - FIXED: output proper dimensions
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, num_scales, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion = nn.Conv2d(num_scales * channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        
        scale_features = []
        for conv in self.dilated_convs:
            scale_features.append(conv(x))
        
        # FIXED: Proper attention weight handling
        attention_weights = self.attention_net(x)  # [B, num_scales, 1, 1]
        
        weighted_features = []
        for i, feat in enumerate(scale_features):
            # FIXED: Extract proper weight dimensions
            weight = attention_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_features.append(feat * weight)
        
        # FIXED: Concatenate along channel dimension
        fused = self.fusion(torch.cat(weighted_features, dim=1))
        return x + self.gamma * fused

class LightweightContextBridge(nn.Module):
    """Memory-efficient context bridge without heavy attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Local context only (remove global attention)
        self.local_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels, 1),  # Pointwise
            nn.GELU()
        )
        
        # Global context via simple pooling
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # Local context
        local_feat = self.local_path(x)
        
        # Global context (simple channel attention)
        global_weights = self.global_path(x)
        global_feat = x * global_weights
        
        # Simple fusion
        fused = local_feat + global_feat
        return self.out_proj(fused) + x

class LightweightHFCBlock(nn.Module):
    """
    Lightweight version of HFCBlock with reduced memory usage
    """
    def __init__(self, channels, expansion=1, num_scales=2):
        super().__init__()
        self.channels = channels
        hidden_dim = channels * expansion
        
        # Simplified components
        self.input_norm = nn.BatchNorm2d(channels)
        
        # Lightweight dynamic relations
        self.dynamic_relations = LightweightDynamicRelationModule(channels, num_scales=num_scales)
        
        # Lightweight context bridge
        self.context_bridge = LightweightContextBridge(channels)
        
        # Feature enhancement (simplified)
        self.enhancement = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 1)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        
        # Parallel processing paths (reduced)
        path1 = self.dynamic_relations(x)
        path2 = self.context_bridge(x)
        
        # Simple fusion instead of complex gating
        fused = (path1 + path2) / 2
        
        enhanced = self.enhancement(fused)
        
        return residual + self.gamma * enhanced




# Multi-Relation Hybrid Feature Context Block
# Multi-Scale Hyper Feature Context Block
# MR-HFC




class HFCB(nn.Module):
    """
    Memory-efficient version of EnhancedHFCBlock
    """
    def __init__(self, channels, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        
        # Use lightweight blocks
        self.mode_branches = nn.ModuleList([
            LightweightHFCBlock(channels) for _ in range(num_modes)
        ])
        
        # Simplified mode interaction (remove heavy attention)
        self.mode_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_modes, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, num_modes, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, mode_inputs):
        assert len(mode_inputs) == self.num_modes
        
        mode_features = []
        for i, inp in enumerate(mode_inputs):
            mode_feat = self.mode_branches[i](inp)
            mode_features.append(mode_feat)
        
        # Simple weighted fusion (no cross-attention)
        weight_input = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in mode_features], dim=1)
        weights = self.mode_weights(weight_input)
        weights = weights.view(-1, self.num_modes, 1, 1, 1)
        
        final_output = sum(mode_features[i] * weights[:, i] for i in range(self.num_modes))
        
        return final_output, mode_features


class HyperSegStage2(nn.Module):
    def __init__(self, H=256, W=256, backbone_pretrained_path=None):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: SwinMamba Base (matches pretrained model dimensions) """
        backbone = wmamba_b(pretrained=True, pretrained_path=backbone_pretrained_path)
       
        self.layer0 = backbone.layer0  # [batch_size, 128, h/4, w/4]
        self.layer1 = backbone.layer1  # [batch_size, 256, h/8, w/8]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/16, w/16]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/32, w/32]

        """ FEM """
        self.dconv1 = dilated_conv(128, 128)
        self.dconv2 = dilated_conv(256, 128)
        self.dconv3 = dilated_conv(512, 128)
        self.dconv4 = dilated_conv(1024, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = CDFAPreprocess(128, 128, 1)
        self.preprocess_bg4 = CDFAPreprocess(128, 128, 1)
        self.preprocess_uc4 = CDFAPreprocess(128, 128, 1)

        self.preprocess_fg3 = CDFAPreprocess(128, 128, 2)
        self.preprocess_bg3 = CDFAPreprocess(128, 128, 2)
        self.preprocess_uc3 = CDFAPreprocess(128, 128, 2)

        self.preprocess_fg2 = CDFAPreprocess(128, 128,4)
        self.preprocess_bg2 = CDFAPreprocess(128, 128, 4)
        self.preprocess_uc2 = CDFAPreprocess(128, 128, 4)

        self.preprocess_fg1 = CDFAPreprocess(128, 128,8)
        self.preprocess_bg1 = CDFAPreprocess(128, 128, 8)
        self.preprocess_uc1 = CDFAPreprocess(128, 128, 8)

        """ Auxiliary Head """
        self.aux_head = AuxiliaryHead(128)

        """ Lightweight EnhancedHFCBlock (Memory Efficient) """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Use lightweight blocks
        self.enhanced_hfc4 = HFCB(channels=128, num_modes=3)
        self.enhanced_hfc3 = HFCB(channels=128, num_modes=3)
        self.enhanced_hfc2 = HFCB(channels=128, num_modes=3)
        self.enhanced_hfc1 = HFCB(channels=128, num_modes=3)

        """ Feature Fusion Convs """
        self.fusion_conv4 = CBR(128 + 128, 128, kernel_size=3, padding=1)
        self.fusion_conv3 = CBR(128 + 128 + 128, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = CBR(128 + 128 + 128, 128, kernel_size=3, padding=1)
        self.fusion_conv1 = CBR(128 + 128 + 128, 128, kernel_size=3, padding=1)

        """ Decoder """
        self.decoder_small = decoder_block(128, 128, scale=2)
        self.decoder_middle = decoder_block(128, 128, scale=2)
        self.decoder_large = decoder_block(128, 128, scale=2)

        """ Output Block """
        self.output_block = output_block(128, 1)

    def forward(self, image):
        """ Backbone: SwinMamba """
        x0 = image
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        # print('x0', x0.shape)
        # print('x1', x1.shape)
        # print('x2', x2.shape)
        # print('x3', x3.shape)
        # print('x4', x4.shape)

        """ Dilated Conv """
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x2)
        d3 = self.dconv3(x3)
        d4 = self.dconv4(x4)
        
        # print('d1', d1.shape)
        # print('d2', d2.shape)
        # print('d3', d3.shape)
        # print('d4', d4.shape)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x4)

        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)
        
        """ Preprocess features """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_uc4 = self.preprocess_uc4(f_uc)

        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_uc3 = self.preprocess_uc3(f_uc)

        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_uc2 = self.preprocess_uc2(f_uc)

        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)
        f_uc1 = self.preprocess_uc1(f_uc)
        
        # print("fbg1: ", f_bg1.shape)

        # Level 4
        mode_inputs_4 = [f_fg4, f_bg4, f_uc4]
        f4_hfc, _ = self.enhanced_hfc4(mode_inputs_4)
        f4_combined = torch.cat([d4, f4_hfc], dim=1)
        f4 = self.fusion_conv4(f4_combined)
        
        # Level 3
        f4_up = self.up2X(f4)
        mode_inputs_3 = [f_fg3, f_bg3, f_uc3]
        f3_hfc, _ = self.enhanced_hfc3(mode_inputs_3)
        f3_combined = torch.cat([d3, f4_up, f3_hfc], dim=1)
        f3 = self.fusion_conv3(f3_combined)
        
        # Level 2
        f3_up = self.up2X(f3)
        mode_inputs_2 = [f_fg2, f_bg2, f_uc2]
        f2_hfc, _ = self.enhanced_hfc2(mode_inputs_2)
        f2_combined = torch.cat([d2, f3_up, f2_hfc], dim=1)
        f2 = self.fusion_conv2(f2_combined)
        
        # Level 1
        f2_up = self.up2X(f2)
        mode_inputs_1 = [f_fg1, f_bg1, f_uc1]
        f1_hfc, _ = self.enhanced_hfc1(mode_inputs_1)
        f1_combined = torch.cat([d1, f2_up, f1_hfc], dim=1)
        f1 = self.fusion_conv1(f1_combined)

        """ Decoder """
        f_small = self.decoder_small(f2, f1)
        # print('f_small: ', f_small.shape)

        f_middle = self.decoder_middle(f3, f2)
        # print('f_middle: ', f_middle.shape)

        f_large = self.decoder_large(f4, f3)
        # print('f_large: ', f_large.shape)

        """ Output Block """
        mask = self.output_block(f_small, f_middle, f_large)
        # print('mask: ', mask.shape)

        return mask, mask_fg, mask_bg, mask_uc

if __name__ == "__main__":
    model = HyperSegStage2().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output[0].shape)
 


