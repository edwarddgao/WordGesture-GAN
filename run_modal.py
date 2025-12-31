#!/usr/bin/env python3
"""Run Modal GPU test with proxy patch."""
import os
import asyncio

# Fix token format
token_id = os.environ.get('MODAL_TOKEN_ID', '').strip('"')
if not token_id.startswith('ak-'):
    token_id = 'ak-' + token_id
os.environ['MODAL_TOKEN_ID'] = token_id
os.environ['MODAL_TOKEN_SECRET'] = os.environ.get('MODAL_TOKEN_SECRET', '').strip('"')

# Apply proxy patch BEFORE importing modal
import modal_proxy_patch

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

async def main():
    print("Starting Modal app...")
    async with app.run():
        print("Calling GPU function...")
        result = await test_gpu.remote.aio()
        print(f"✓ GPU Test Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
