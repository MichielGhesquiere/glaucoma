#!/usr/bin/env python3
"""
Test script to verify optimization features are working correctly.
"""

import torch
import time

def test_flash_attention():
    """Test if Flash Attention is working."""
    print("=== Testing Flash Attention ===")
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        flash_enabled = torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda, 'flash_sdp_enabled') else False
        print(f"Flash Attention enabled: {flash_enabled}")
        return flash_enabled
    except Exception as e:
        print(f"Flash Attention test failed: {e}")
        return False

def test_amp():
    """Test if Automatic Mixed Precision is working."""
    print("\n=== Testing Automatic Mixed Precision ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            print("AMP GradScaler created successfully")
            
            # Test autocast
            with torch.amp.autocast('cuda'):
                x = torch.randn(2, 3, 224, 224, device=device)
                print(f"Autocast test successful, tensor dtype: {x.dtype}")
            return True
        else:
            print("CUDA not available, AMP not tested")
            return False
    except Exception as e:
        print(f"AMP test failed: {e}")
        return False

def test_torch_compile():
    """Test if torch.compile is working."""
    print("\n=== Testing Torch Compile ===")
    try:
        if hasattr(torch, 'compile'):
            print("torch.compile is available")
            
            # Simple model for testing
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Try to compile
            try:
                compiled_model = torch.compile(model, mode='reduce-overhead')
                print("Model compilation successful")
                
                # Test inference
                x = torch.randn(4, 10, device=device)
                with torch.no_grad():
                    output = compiled_model(x)
                print(f"Compiled model inference successful, output shape: {output.shape}")
                return True
            except Exception as compile_error:
                print(f"Model compilation failed: {compile_error}")
                print("This is expected if Triton is not installed")
                return False
        else:
            print("torch.compile not available (PyTorch < 2.0)")
            return False
    except Exception as e:
        print(f"Torch compile test failed: {e}")
        return False

def test_channels_last():
    """Test channels_last memory format."""
    print("\n=== Testing Channels Last Memory Format ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create tensor in channels_last format
        x = torch.randn(2, 3, 224, 224, device=device)
        x_channels_last = x.to(memory_format=torch.channels_last)
        
        print(f"Original tensor stride: {x.stride()}")
        print(f"Channels last tensor stride: {x_channels_last.stride()}")
        print(f"Is channels last: {x_channels_last.is_contiguous(memory_format=torch.channels_last)}")
        return True
    except Exception as e:
        print(f"Channels last test failed: {e}")
        return False

def benchmark_inference_speed():
    """Simple benchmark to show potential speedup."""
    print("\n=== Inference Speed Benchmark ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running benchmark on: {device}")
        
        if device.type != 'cuda':
            print("Skipping benchmark - CUDA not available")
            return
        
        # Create a simple ViT-like model
        from timm import create_model
        model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model = model.to(device)
        model.eval()
        
        # Test data
        x = torch.randn(16, 3, 224, 224, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark normal inference
        start_time = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        normal_time = time.time() - start_time
        
        # Benchmark with AMP
        start_time = time.time()
        for _ in range(50):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    _ = model(x)
        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        
        # Benchmark with channels_last
        x_cl = x.to(memory_format=torch.channels_last)
        model_cl = model.to(memory_format=torch.channels_last)
        start_time = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model_cl(x_cl)
        torch.cuda.synchronize()
        channels_last_time = time.time() - start_time
        
        print(f"Normal inference time: {normal_time:.3f}s")
        print(f"AMP inference time: {amp_time:.3f}s ({normal_time/amp_time:.1f}x speedup)")
        print(f"Channels last time: {channels_last_time:.3f}s ({normal_time/channels_last_time:.1f}x speedup)")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    """Run all tests."""
    print("PyTorch Optimization Features Test")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    results = {
        'flash_attention': test_flash_attention(),
        'amp': test_amp(),
        'torch_compile': test_torch_compile(),
        'channels_last': test_channels_last()
    }
    
    print("\n=== Summary ===")
    for feature, status in results.items():
        status_str = "✓ Working" if status else "✗ Not working"
        print(f"{feature.replace('_', ' ').title()}: {status_str}")
    
    if torch.cuda.is_available():
        benchmark_inference_speed()
    
    print("\nRecommendations for your training:")
    print("- Use AMP (--use_amp): Always recommended for GPU training")
    if results['flash_attention']:
        print("- Flash Attention: Already enabled automatically")
    if results['channels_last']:
        print("- Channels Last (--channels_last): Try it, may provide 5-15% speedup")
    if not results['torch_compile']:
        print("- Torch Compile (--compile_model): Not available, requires Triton installation")
        print("  To install: pip install triton (may not work on all systems)")

if __name__ == "__main__":
    main()
