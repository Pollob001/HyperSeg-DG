import math
import torch
import torch.nn as nn
from network.resnet import resnet50
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
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
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

        x=self.up_2x2(x)
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
    """
    Upgraded: Same interface/behavior as your original block.
    For each ×2 upsample:
        Upsample (bilinear) -> 3x3 CBR -> Swin-lite refine (W-MSA; alt. shifted).
    """
    def __init__(self, in_c, out_c, up_scale,
                 window_size=7, num_heads=4, mlp_ratio=2.0, drop=0.):
        super().__init__()
        # keep the original contract: up_scale must be power-of-two
        assert up_scale >= 1 and (up_scale & (up_scale - 1) == 0), "up_scale must be a power of two"
        up_times = int(math.log2(up_scale))

        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        self.stages = nn.ModuleList()

        for i in range(up_times):
            # same upsample + conv you had
            up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv = CBR(out_c, out_c, kernel_size=3, padding=1)
            # Swin-inspired tiny block (alternate shift to connect windows)
            shift = 0 if (i % 2 == 0) else window_size // 2
            swin = _SwinLiteBlock(out_c, window_size=window_size, shift_size=shift,
                                  num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            self.stages.append(nn.ModuleDict(dict(up=up, conv=conv, swin=swin)))

    def forward(self, x):
        x = self.c1(x)
        for stage in self.stages:
            x = stage["up"](x)
            x = stage["conv"](x)
            x = stage["swin"](x)
        return x



######################################################################################################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers: tiny Swin-style window attention --------------------------------

def _window_partition(x, ws):
    # x: [B, H, W, C] -> [B*nw, ws, ws, C]
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return x

def _window_unpartition(windows, ws, H, W):
    # windows: [B*nw, ws, ws, C] -> [B, H, W, C]
    B = windows.shape[0] // (H // ws * W // ws)
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class _WindowMSA(nn.Module):
    """Minimal window self-attention (no relative bias to keep it small)."""
    def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [Bn, N, C], where N=ws*ws
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [3, Bn, heads, N, d]
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [Bn, heads, N, N]
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj_drop(self.proj(out))
        return out

class _SwinLiteBlock(nn.Module):
    """
    One Swin-ish block: (shift) + window partition -> MSA -> MLP, with residuals.
    """
    def __init__(self, dim, window_size=7, shift_size=0, num_heads=4, mlp_ratio=2.0, drop=0.):
        super().__init__()
        self.ws = window_size
        self.shift = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowMSA(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        # channels-last for LayerNorm & attention
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # pad to multiple of window size
        pad_b = (self.ws - H % self.ws) % self.ws
        pad_r = (self.ws - W % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # pad (C dim unchanged)
        Hp, Wp = x.shape[1], x.shape[2]

        # window partition -> attn -> unpartition
        xw = _window_partition(x, self.ws)                  # [Bn, ws, ws, C]
        xw = xw.view(-1, self.ws * self.ws, C)             # [Bn, N, C]
        xw = self.norm1(xw)
        attn_out = self.attn(xw)
        xw = attn_out + xw                                  # residual 1
        xw = self.norm2(xw)
        xw = self.mlp(xw) + xw                              # residual 2
        xw = xw.view(-1, self.ws, self.ws, C)
        x = _window_unpartition(xw, self.ws, Hp, Wp)        # [B, Hp, Wp, C]

        # remove pad
        x = x[:, :H, :W, :]

        # reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        # back to channels-first
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# --- your upgraded block -------------------------------------------------------

class DualContrastWindowAttn(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.scale = self.head_dim ** -0.5

        # projections
        self.v_proj   = nn.Linear(dim, dim)
        self.w_fg     = nn.Linear(dim, kernel_size * kernel_size * num_heads)
        self.w_bg     = nn.Linear(dim, kernel_size * kernel_size * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool   = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)


        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        """
        x  : [B, C_in, H, W]  -> internally lifted to [B, H, W, dim]
        fg : [B, dim, H, W] or [B, C_fg, H, W] aligned with x after preprocess
        bg : [B, dim, H, W] or [B, C_bg, H, W] aligned with x after preprocess
        returns: [B, dim, H, W]
        """
        # wrap-in convs to match original CDFA API
        x  = self.input_cbr(x).permute(0, 2, 3, 1)   # [B,H,W,dim]
        fg = fg.permute(0, 2, 3, 1)                  # [B,H,W,Cfg]
        bg = bg.permute(0, 2, 3, 1)                  # [B,H,W,Cbg]

        B, H, W, C = x.shape
        KK = self.kernel_size * self.kernel_size
        h  = self.num_heads
        d  = self.head_dim

        # values -> unfold to local k×k patches
        v = self.v_proj(x).permute(0, 3, 1, 2)       # [B,dim,H,W]
        V = self.unfold(v)                           # [B, dim*KK, N]
        N = V.shape[-1]
        V = V.reshape(B, h, d, KK, N).permute(0, 1, 4, 3, 2)  # [B,h,N,KK,d]

        # pool fg/bg onto the same stride grid used for unfold windows
        # (with stride=1 this is a no-op; remains H×W)
        fg_p = self.pool(fg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,H',W',C]
        bg_p = self.pool(bg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,H',W',C]
        # flatten spatial to windows
        fg_p = fg_p.reshape(B, -1, C)  # [B,N,C]
        bg_p = bg_p.reshape(B, -1, C)  # [B,N,C]
        assert fg_p.shape[1] == N and bg_p.shape[1] == N, "window count mismatch"

        # per-window, per-head logits over KK locations (no queries/keys; contrastive weights)
        w_fg = self.w_fg(fg_p).reshape(B, N, h, KK).permute(0, 2, 1, 3)  # [B,h,N,KK]
        w_bg = self.w_bg(bg_p).reshape(B, N, h, KK).permute(0, 2, 1, 3)  # [B,h,N,KK]

        # single contrastive attention distribution
        logits = (w_fg - w_bg) * self.scale                 # [B,h,N,KK]
        attn   = F.softmax(logits, dim=-1)                  # [B,h,N,KK]
        attn   = self.attn_drop(attn).unsqueeze(-1)         # [B,h,N,KK,1]

        # apply attention to values; KEEP KK so we can fold back
        out = (attn * V)                                    # [B,h,N,KK,d]
        out = out.permute(0, 1, 4, 3, 2).contiguous()       # [B,h,d,KK, N]
        out = out.view(B, self.dim * KK, N)                 # [B, dim*KK, N]

        # fold back to spatial map
        out = F.fold(out, output_size=(H, W),
                     kernel_size=self.kernel_size,
                     padding=self.padding,
                     stride=self.stride)                    # [B, dim, H, W]

        # project & wrap-out convs to match original CDFA API
        out = out.permute(0, 2, 3, 1)                       # [B,H,W,dim]
        out = self.proj_drop(self.proj(out))                # [B,H,W,dim]
        out = out.permute(0, 3, 1, 2)                       # [B,dim,H,W]
        out = self.output_cbr(out)                          # [B,dim,H,W]
        return out


#####################################################################################################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers: tiny Swin-style window attention --------------------------------

def _window_partition(x, ws):
    # x: [B, H, W, C] -> [B*nw, ws, ws, C]
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return x

def _window_unpartition(windows, ws, H, W):
    # windows: [B*nw, ws, ws, C] -> [B, H, W, C]
    B = windows.shape[0] // (H // ws * W // ws)
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class _WindowMSA(nn.Module):
    """Minimal window self-attention (no relative bias to keep it small)."""
    def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [Bn, N, C], where N=ws*ws
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [3, Bn, heads, N, d]
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [Bn, heads, N, N]
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj_drop(self.proj(out))
        return out

class _SwinLiteBlock(nn.Module):
    """
    One Swin-ish block: (shift) + window partition -> MSA -> MLP, with residuals.
    """
    def __init__(self, dim, window_size=7, shift_size=0, num_heads=4, mlp_ratio=2.0, drop=0.):
        super().__init__()
        self.ws = window_size
        self.shift = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowMSA(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        # channels-last for LayerNorm & attention
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # pad to multiple of window size
        pad_b = (self.ws - H % self.ws) % self.ws
        pad_r = (self.ws - W % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # pad (C dim unchanged)
        Hp, Wp = x.shape[1], x.shape[2]

        # window partition -> attn -> unpartition
        xw = _window_partition(x, self.ws)                  # [Bn, ws, ws, C]
        xw = xw.view(-1, self.ws * self.ws, C)             # [Bn, N, C]
        xw = self.norm1(xw)
        attn_out = self.attn(xw)
        xw = attn_out + xw                                  # residual 1
        xw = self.norm2(xw)
        xw = self.mlp(xw) + xw                              # residual 2
        xw = xw.view(-1, self.ws, self.ws, C)
        x = _window_unpartition(xw, self.ws, Hp, Wp)        # [B, Hp, Wp, C]

        # remove pad
        x = x[:, :H, :W, :]

        # reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        # back to channels-first
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


####################################################################################

import os
import torch
import torch.nn as nn
import timm
import datetime
import time
import albumentations as A
from torch.utils.data import DataLoader

# Function to load checkpoint (with support for cases when 'state_dict' is not present)
def load_checkpoint(model, checkpoint_path, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Print checkpoint keys for debugging purposes
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # If 'state_dict' exists, load model weights from it
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        # If 'state_dict' doesn't exist, load weights directly
        model.load_state_dict(checkpoint, strict=strict)

# SwinBackbone class with weight loading
class SwinBackbone(nn.Module):
    def __init__(
        self,
        model_name="swinv2_base_window8_256.ms_in1k",
        weight_path="/ghome/aynulislam/ConDSeg2/network/swinv2_base_window8_256.ms_in1k.pth",
    ):
        super().__init__()

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at: {weight_path}")

        # Create the Swin Transformer model (without pre-trained weights)
        self.base = timm.create_model(model_name, pretrained=False)

        # Load the custom checkpoint
        try:
            load_checkpoint(self.base, weight_path, strict=True)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            load_checkpoint(self.base, weight_path, strict=False)  # Try non-strict loading if strict=True fails

        # Define the stem layer (similar to ResNet's initial layers)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Debug: Print the input size before passing to the model
        print(f"Input size to SwinBackbone: {x.shape}")  # This will show the input tensor size
        x = self.stem(x)
        return self.base(x)

# Debug: Inspecting model parameters to make sure none are empty
def print_model_params(model):
    print("Model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")
        else:
            print(f"{name} has no grad")

# Define the ConDSeg model with SwinBackbone
class ConDSeg(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: Swin Transformer """
        self.swin_backbone = SwinBackbone()

        """ FEM """
        self.dconv1 = dilated_conv(64, 128)
        self.dconv2 = dilated_conv(256, 128)
        self.dconv3 = dilated_conv(512, 128)
        self.dconv4 = dilated_conv(1024, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = CDFAPreprocess(128, 128, 1)  # 1/16
        self.preprocess_bg4 = CDFAPreprocess(128, 128, 1)  # 1/16

        self.preprocess_fg3 = CDFAPreprocess(128, 128, 2)  # 1/8
        self.preprocess_bg3 = CDFAPreprocess(128, 128, 2)  # 1/8

        self.preprocess_fg2 = CDFAPreprocess(128, 128, 4)  # 1/4
        self.preprocess_bg2 = CDFAPreprocess(128, 128, 4)  # 1/4

        self.preprocess_fg1 = CDFAPreprocess(128, 128, 8)  # 1/2
        self.preprocess_bg1 = CDFAPreprocess(128, 128, 8)  # 1/2

        """ Auxiliary Head """
        self.aux_head = AuxiliaryHead(128)

        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cdfa4 = ContrastDrivenFeatureAggregation(128, 128, 4)
        self.cdfa3 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa2 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa1 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)

        """ Decoder """
        self.decoder_small = decoder_block(128, 128, scale=2)
        self.decoder_middle = decoder_block(128, 128, scale=2)
        self.decoder_large = decoder_block(128, 128, scale=2)

        """ Output Block """
        self.output_block = output_block(128, 1)

        # Print model parameters for debugging
        print_model_params(self)

    def forward(self, image):
        # Debug: Print the size of the image input before passing it to the model
        print(f"Input image size to ConDSeg: {image.shape}")  # Print image size before passing into the model

        """ Backbone: Swin Transformer """
        x0 = image
        x1 = self.swin_backbone(x0)  # Get the feature map from the Swin Transformer backbone

        # Assuming we extract features from various stages of the Swin transformer (you can adjust this)
        # For simplicity, we assume the feature map `x1` is at a comparable level to ResNet50's last feature map

        """ Dilated Conv """
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x1)  # You may want to adjust these layers based on feature map sizes
        d3 = self.dconv3(x1)
        d4 = self.dconv4(x1)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x1)

        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)

        f4 = self.cdfa4(d4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([d3, f4_up], dim=1)
        f3 = self.cdfa3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([d2, f3_up], dim=1)
        f2 = self.cdfa2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([d1, f2_up], dim=1)
        f1 = self.cdfa1(f_2_1, f_fg1, f_bg1)

        """ Decoder """
        f_small = self.decoder_small(f2, f1)
        f_middle = self.decoder_middle(f3, f2)
        f_large = self.decoder_large(f4, f3)

        """ Output Block """
        mask = self.output_block(f_small, f_middle, f_large)

        return mask, mask_fg, mask_bg, mask_uc


if __name__ == "__main__":
    model = ConDSeg().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)
