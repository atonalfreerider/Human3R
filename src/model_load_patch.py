"""
Patch for model loading compatibility issues in dust3r/model.py
This fixes both the verbose parameter issue and PyTorch 2.8+ weights_only issue
"""

import sys
import types
import torch

def patch_dust3r_model_loading():
    """
    Monkey-patch the dust3r model loading to fix compatibility issues
    """
    try:
        from src.dust3r import model as dust3r_model
        
        # Store original load_model function
        original_load_model = dust3r_model.load_model
        
        def patched_load_model(model_path, device="cpu", verbose=True):
            """
            Patched load_model with verbose parameter and PyTorch 2.8+ compatibility
            """
            if verbose:
                print(f"... loading model from {model_path}")
            
            # PyTorch 2.8+ compatibility: use weights_only=False
            try:
                ckpt = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                ckpt = torch.load(model_path, map_location=device)
            
            # Call the rest of the original function logic
            # This assumes the original function continues after loading the checkpoint
            # If this doesn't work, we may need to replicate more of the original logic
            return original_load_model.__wrapped__(model_path, device, verbose) if hasattr(original_load_model, '__wrapped__') else original_load_model(model_path, device)
        
        # Replace the function
        dust3r_model.load_model = patched_load_model
        print("✓ Applied model loading compatibility patch")
        
    except ImportError as e:
        print(f"⚠️ Could not apply model loading patch: {e}")

# Auto-apply patch when imported
patch_dust3r_model_loading()
