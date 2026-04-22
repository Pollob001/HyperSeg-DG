import torch
import torch.nn as nn
from pretrained.wmamba import WMamba

class wmamba_backbone(nn.Module):
    """SwinMamba with exact ResNet50-like interface"""
    
    def __init__(self, arch='wmamba_t', pretrained=True, img_size=256, pretrained_path=None, **kwargs):
        super().__init__()
        
        # Create SwinMamba model
        if arch == 'wmamba_t':
            self.backbone = WMamba(
                img_size=img_size, num_classes=1000,
                depths=(2, 2, 4, 2), dims=(96, 192, 384, 768),
                window_size=8, **kwargs
            )
        elif arch == 'wmamba_s':
            self.backbone = WMamba(
                img_size=img_size, num_classes=1000,
                depths=(2, 2, 8, 2), dims=(96, 192, 384, 768),
                window_size=8, **kwargs
            )
        elif arch == 'wmamba_b':
            self.backbone = WMamba(
                img_size=img_size, num_classes=1000,
                depths=(2, 2, 12, 2), dims=(128, 256, 512, 1024),
                window_size=8, **kwargs
            )
        
        # Load pretrained weights
        if pretrained:
            self._load_pretrained(arch, pretrained_path=pretrained_path)
        
        # Remove classification head
        if hasattr(self.backbone, 'head'):
            del self.backbone.head
        if hasattr(self.backbone, 'avgpool'):
            del self.backbone.avgpool
        
        # Create custom layer0 that handles format conversion
        self.layer0 = _SwinMambaLayer0(self.backbone.patch_embed)
        
        # Layer access with wrappers
        self.layer1 = _SwinMambaLayerWrapper(self.backbone.layers[0])  # First stage
        self.layer2 = _SwinMambaLayerWrapper(self.backbone.layers[1])  # Second stage  
        self.layer3 = _SwinMambaLayerWrapper(self.backbone.layers[2])  # Third stage
        self.layer4 = _SwinMambaLayerWrapper(self.backbone.layers[3])  # Fourth stage
    
    def _load_pretrained(self, arch, pretrained_path=None):
        """Load pretrained weights"""
        import os
        if pretrained_path is not None:
            print(f"Using user-provided pretrained path: {pretrained_path}")
        else:
            # Try absolute path first, then fallback to relative path
            absolute_path = '/ghome/aynulislam/HyperSeg_DG/pretrained/swinmamba.pth'
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            relative_path = os.path.join(base_dir, 'pretrained', 'swinmamba.pth')

            # Use absolute path if it exists, otherwise use relative path
            if os.path.exists(absolute_path):
                pretrained_path = absolute_path
                print(f"Using absolute pretrained path: {pretrained_path}")
            else:
                pretrained_path = relative_path
                print(f"Using relative pretrained path: {pretrained_path}")
        
        model_paths = {
            'wmamba_t': pretrained_path,
            'wmamba_s': pretrained_path,
            'wmamba_b': pretrained_path,  # Same file, different architecture
        }
        
        if arch not in model_paths:
            print(f"No pretrained weights available for {arch}")
            return
        
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained model file not found at {pretrained_path}")
            return
        
        try:
            state_dict = torch.load(model_paths[arch], map_location='cpu')
            
            # Clean state dict
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v
            
            # Load weights, ignore head
            model_dict = self.backbone.state_dict()
            pretrained_dict = {k: v for k, v in clean_state_dict.items() 
                             if k in model_dict and not k.startswith('head.')}
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict, strict=False)
            
            print(f"Successfully loaded pretrained weights for {arch}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def forward(self, x):
        # This won't be used directly since you access layers individually
        return x


class _SwinMambaLayer0(nn.Module):
    """Custom layer0 that handles NCHW to NHWC conversion for patch embedding"""
    def __init__(self, patch_embed):
        super().__init__()
        self.patch_embed = patch_embed
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x is in NCHW format
        # Apply patch embedding (automatically converts to NHWC internally)
        x = self.patch_embed(x)  # Output: NHWC [B, H/4, W/4, C]
        
        # Convert back to NCHW for compatibility with your model
        x = x.permute(0, 3, 1, 2).contiguous()  # NCHW [B, C, H/4, W/4]
        
        # Apply ReLU activation
        x = self.relu(x)
        
        return x


class _SwinMambaLayerWrapper(nn.Module):
    """Wrapper to make SwinMamba layers work like ResNet layers"""
    def __init__(self, swin_layer):
        super().__init__()
        self.swin_layer = swin_layer
    
    def forward(self, x):
        # Convert NCHW to NHWC for SwinMamba
        B, C, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Process through SwinMamba layer
        out_nhwc = self.swin_layer(x_nhwc)
        
        # Convert back to NCHW
        out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return out_nchw


# Convenience functions
def swinmamba_t(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_t', pretrained, img_size=img_size, **kwargs)

def swinmamba_s(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_s', pretrained, img_size=img_size, **kwargs)

def swinmamba_b(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_b', pretrained, img_size=img_size, **kwargs)


# WMamba aliases (preferred naming)
def wmamba_t(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_t', pretrained, img_size=img_size, **kwargs)


def wmamba_s(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_s', pretrained, img_size=img_size, **kwargs)


def wmamba_b(pretrained=True, img_size=256, **kwargs):
    return wmamba_backbone('wmamba_b', pretrained, img_size=img_size, **kwargs)


def wmamba(pretrained=True, img_size=256, **kwargs):
    """Default WMamba builder used by stage1 models."""
    return wmamba_b(pretrained=pretrained, img_size=img_size, **kwargs)