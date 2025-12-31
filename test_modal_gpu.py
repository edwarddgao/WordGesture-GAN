"""Test Modal GPU access."""
import modal

app = modal.App("test-gpu")
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")

@app.function(gpu="T4", timeout=120, image=image)
def test_gpu():
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

@app.local_entrypoint()
def main():
    result = test_gpu.remote()
    print(f"GPU Test Result: {result}")
