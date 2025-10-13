"""
PyTorch 2.8+ compatibility patch for torch.load
Adds weights_only=False to torch.load calls to support loading models with OmegaConf
"""

import torch
import functools

# Store the original torch.load function
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    """
    Patched torch.load that adds weights_only=False for backward compatibility
    with PyTorch 2.8+ when loading models that contain custom objects like OmegaConf
    """
    # Only add weights_only if not already specified
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    
    try:
        return _original_torch_load(*args, **kwargs)
    except TypeError:
        # For older PyTorch versions that don't support weights_only parameter
        kwargs.pop('weights_only', None)
        return _original_torch_load(*args, **kwargs)

# Monkey-patch torch.load
torch.load = _patched_torch_load

print("✓ Applied PyTorch 2.8+ compatibility patch for model loading")
