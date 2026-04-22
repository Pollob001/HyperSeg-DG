import torch
try:
    from pretrained.wmamba import WMamba
except ImportError:
    from wmamba import WMamba

class WMambaFeatureExtractor(WMamba):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = {}
        self.hooks = []
        
    def _register_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        for i, layer in enumerate(self.layers):
            def hook_fn(module, input, output, layer_idx=i):
                self.features[f'layer_{layer_idx}'] = output
            self.hooks.append(layer.register_forward_hook(hook_fn))
        
        def patch_embed_hook(module, input, output):
            self.features['patch_embed'] = output
        self.hooks.append(self.patch_embed.register_forward_hook(patch_embed_hook))
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        self.features = {}
        
        if return_features and not self.hooks:
            self._register_hooks()
        
        x = self.patch_embed(x)
        self.features['patch_embed'] = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.features[f'layer_{i}'] = x
        
        x_pool = x.permute(0, 3, 1, 2).contiguous()
        x_pool = self.avgpool(x_pool)
        x_pool = torch.flatten(x_pool, 1)
        x_out = self.head(x_pool)
        
        if return_features:
            return x_out, self.features
        return x_out

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model and move to device
model = WMambaFeatureExtractor(
    img_size=256,
    num_classes=1000,
    depths=(2, 2, 4, 2),
    dims=(96, 192, 384, 768),
    window_size=8,
    d_state=16
).to(device)

# Load and clean state dict
state_dict = torch.load('/ghome/aynulislam/HyperSeg_DG/pretrained/swinmamba.pth', map_location='cpu')
clean_state_dict = remove_module_prefix(state_dict)

# Load cleaned state dict
model.load_state_dict(clean_state_dict)
model.eval()  # Set to evaluation mode

# Example usage - make sure input is on the same device
x = torch.randn(2, 3, 256, 256).to(device)

# Extract features
with torch.no_grad():
    outputs, features = model(x, return_features=True)
    
print("Feature shapes:")
for name, feature in features.items():
    print(f"{name}: {feature.shape}")