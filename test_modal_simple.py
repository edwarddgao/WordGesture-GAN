#!/usr/bin/env python3
"""Simple Modal test to verify GPU execution works."""

import modal_proxy_patch  # Must be first
import modal
import asyncio

app = modal.App('simple-test')

image = modal.Image.debian_slim(python_version='3.11').pip_install('torch>=2.0.0')


@app.function(gpu='T4', image=image, timeout=300)
def test_gpu():
    import torch
    print("=" * 50)
    print("INSIDE MODAL FUNCTION")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Quick matrix multiply to verify GPU works
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"Matrix multiply result shape: {z.shape}")
        return {"status": "success", "gpu": torch.cuda.get_device_name(0)}
    else:
        print("NO CUDA!")
        return {"status": "no_cuda"}


async def main():
    print("Starting simple Modal test...")
    async with app.run():
        result = await test_gpu.remote.aio()
        print(f"Result: {result}")
    return result


if __name__ == '__main__':
    asyncio.run(main())
