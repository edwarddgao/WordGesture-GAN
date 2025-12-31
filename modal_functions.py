"""Modal function definitions - no local imports that would fail on cloud."""
import modal

# Create app and image
app = modal.App("test-cpu")

@app.function(timeout=60)
def hello():
    """Simple CPU function."""
    import sys
    return f"Hello from Modal! Python {sys.version}"


# GPU test app
gpu_app = modal.App("test-gpu")
gpu_image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")

@gpu_app.function(gpu="T4", timeout=120, image=gpu_image)
def test_gpu():
    """GPU test function."""
    import torch
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "result": float(z.sum())
        }
    return {"cuda": False}
