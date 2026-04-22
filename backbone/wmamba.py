import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_

# Try to import fast selective-scan backends
selective_scan_fn = selective_scan_ref = None
selective_scan_fn_v1 = selective_scan_ref_v1 = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except Exception:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# ===========================
# Patch embedding / merging
# ===========================

class PatchEmbed2D(nn.Module):
    """Image to patch embedding (NHWC)."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()  # NHWC
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Downsample NHWC by 2."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # If odd, crop last row/col to keep even sizes
        if (H % 2 != 0) or (W % 2 != 0):
            H, W = H - H % 2, W - W % 2
            x = x[:, :H, :W, :]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)          # B,H/2,W/2,4C
        x = self.norm(x)
        x = self.reduction(x)                            # B,H/2,W/2,2C
        return x


# ===========================
# SS2D core (NHWC)
# ===========================

class SS2D(nn.Module):
    """
    NHWC Mamba-style 2D SSM with 4-direction scanning (row/col + reverses).
    Designed to run per window or full map.
    """
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 3,
                 expand: float = 2.0,
                 dt_rank: str = "auto",
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4,
                 dropout: float = 0.0,
                 conv_bias: bool = True,
                 bias: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # in-proj + depthwise conv prefilter
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d  = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=(d_conv - 1) // 2,
                                 groups=self.d_inner, bias=conv_bias, **factory_kwargs)
        self.act = nn.SiLU()

        # directional projections
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2),
                      bias=False, **factory_kwargs)
            for _ in range(4)
        ])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt projections
        self.dt_projs = nn.ModuleList([
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                          dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias   = nn.Parameter(torch.stack([t.bias   for t in self.dt_projs], dim=0))
        del self.dt_projs

        # state-space params
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds     = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else None

        self._scan_backend = self._pick_scan()

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            nn.init.constant_(dt_proj.weight, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @staticmethod
    def _pick_scan():
        if selective_scan_fn is not None: return selective_scan_fn
        if selective_scan_fn_v1 is not None: return selective_scan_fn_v1
        if selective_scan_ref is not None: return selective_scan_ref
        if selective_scan_ref_v1 is not None: return selective_scan_ref_v1
        raise RuntimeError("No selective_scan backend found. Install 'mamba-ssm' or 'selective_scan'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: NHWC (B,H, W, C) — can be a full feature map or a single window.
        """
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # depthwise conv prefilter in NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)

        # 4-direction selective scan
        scan = self._scan_backend
        L = H * W
        K = 4
        # row-major & col-major sequences + reverses
        x_hwwh = torch.stack(
            [x.view(B, -1, L), x.transpose(2, 3).contiguous().view(B, -1, L)],
            dim=1
        ).view(B, 2, -1, L)  # (B,2,C,L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B,4,C,L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_f  = xs.float().view(B, -1, L)
        dts_f = dts.contiguous().float().view(B, -1, L)
        Bs_f  = Bs.float().view(B, K, -1, L)
        Cs_f  = Cs.float().view(B, K, -1, L)
        Ds    = self.Ds.float().view(-1)
        As    = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = scan(xs_f, dts_f, As, Bs_f, Cs_f, Ds,
                     z=None, delta_bias=dt_bias,
                     delta_softplus=True, return_last_state=False).view(B, K, -1, L)

        # sum over 4 directions and return NHWC
        y = out_y.sum(1).view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ===========================
# Window utilities (NHWC)
# ===========================

def window_partition_nhwc(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H/W must be multiples of window_size ({H},{W}) vs {window_size}"
    x = x.view(B, H // window_size, window_size,
                  W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse_nhwc(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    Bn, Ws, _, C = windows.shape
    B = Bn // ((H // Ws) * (W // Ws))
    x = windows.view(B, H // Ws, W // Ws, Ws, Ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


# ===========================
# Blocks (Swin-style)
# ===========================

class NHWC_MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act_layer=nn.GELU):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        x = x.view(B, H, W, C)
        return x


class WMambaBlock(nn.Module):
    """
    LN → (cyclic shift) → window partition → SS2D(in-window) → window reverse → (reverse shift) → +res
       → LN → MLP → +res
    """
    def __init__(self, dim, input_resolution: Tuple[int, int], window_size=7, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 d_state=16, ssm_expand=2.0, ssm_dconv=3,
                 ssm_dt_rank="auto", ssm_drop=0.0, ssm_conv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0

        self.norm1 = norm_layer(dim)
        self.ssm = SS2D(dim, d_state=d_state, d_conv=ssm_dconv, expand=ssm_expand,
                        dt_rank=ssm_dt_rank, dropout=ssm_drop, conv_bias=ssm_conv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = NHWC_MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert (H, W) == self.input_resolution, f"Expected {(H,W)} vs {self.input_resolution}"

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # windowed SS2D
        x_win = window_partition_nhwc(x, self.window_size)  # (B*nW, Ws, Ws, C)
        x_win = self.ssm(x_win)                             # (B*nW, Ws, Ws, C)
        x = window_reverse_nhwc(x_win, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # residuals
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WMambaLayer(nn.Module):
    """depth × WMambaBlock, then optional PatchMerging2D."""
    def __init__(self, dim, depth, input_resolution: Tuple[int, int], window_size=7,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 downsample=None, d_state=16, ssm_expand=2.0, ssm_dconv=3,
                 ssm_dt_rank="auto", ssm_drop=0.0, ssm_conv_bias=True):
        super().__init__()
        H, W = input_resolution
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(WMambaBlock(
                dim, (H, W), window_size, shift, mlp_ratio,
                drop, drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer, d_state, ssm_expand, ssm_dconv,
                ssm_dt_rank, ssm_drop, ssm_conv_bias
            ))
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ===========================
# Full model
# ===========================

class WMamba(nn.Module):
    """
    Window Mamba: Swin hierarchy with Mamba blocks:
    PatchEmbed (NHWC) -> [Layer (blocks, PatchMerging)] x 4 -> GAP -> FC
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 depths=(2, 2, 6, 2), dims=(96, 192, 384, 768), window_size=7,
                 d_state=16, ssm_expand=2.0, ssm_dt_rank="auto",
                 ssm_dconv=3, ssm_drop=0.0, ssm_conv_bias=True,
                 drop_rate=0.0, drop_path_rate=0.1, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        assert len(depths) == len(dims) == 4, "Expect 4 stages."

        self.num_classes = num_classes
        self.dims = list(dims)

        self.patch_embed = PatchEmbed2D(patch_size, in_chans, self.dims[0],
                                        norm_layer if patch_norm else None)

        H = img_size // patch_size
        W = img_size // patch_size

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.layers = nn.ModuleList()
        dp_idx = 0
        for i in range(4):
            depth_i = depths[i]
            layer = WMambaLayer(self.dims[i], depth_i, (H, W), window_size,
                                mlp_ratio, drop_rate, dpr[dp_idx: dp_idx + depth_i],
                                norm_layer,
                                downsample=PatchMerging2D if i < 3 else None,
                                d_state=d_state, ssm_expand=ssm_expand, ssm_dconv=ssm_dconv,
                                ssm_dt_rank=ssm_dt_rank, ssm_drop=ssm_drop, ssm_conv_bias=ssm_conv_bias)
            self.layers.append(layer)
            dp_idx += depth_i
            if i < 3:
                H, W = H // 2, W // 2

        # head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # NHWC
        for layer in self.layers:
            x = layer(x)          # NHWC
        # pool (NCHW)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# ===========================
# Convenience builders
# ===========================

def wmamba_t(img_size=224, num_classes=1000, **kwargs) -> WMamba:
    return WMamba(
        img_size=img_size, num_classes=num_classes,
        depths=(2, 2, 4, 2), dims=(96, 192, 384, 768),
        **kwargs
    )

def wmamba_s(img_size=224, num_classes=1000, **kwargs) -> WMamba:
    return WMamba(
        img_size=img_size, num_classes=num_classes,
        depths=(2, 2, 8, 2), dims=(96, 192, 384, 768),
        **kwargs
    )

def wmamba_b(img_size=224, num_classes=1000, **kwargs) -> WMamba:
    return WMamba(
        img_size=img_size, num_classes=num_classes,
        depths=(2, 2, 12, 2), dims=(128, 256, 512, 1024),
        **kwargs
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 224, 224).to(device)
    model = wmamba_t(num_classes=6, d_state=16).to(device)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)
    print("Model name: WMamba (Window Mamba)")