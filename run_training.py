#!/usr/bin/env python3
"""Run WordGesture-GAN training on Modal cloud GPUs."""
import asyncio
import argparse

import modal_proxy_patch  # Must be before modal
from modal_train import app, test_gpu, train, list_checkpoints


async def run_test():
    """Test GPU access."""
    print("Testing GPU access...")
    async with app.run():
        result = await test_gpu.remote.aio()
        print(f"  Device: {result['device']}")
        print(f"  GPU: {result['gpu']}")
        print(f"  Status: {result['status']}")


async def run_train(epochs: int, batch_size: int, lr: float):
    """Run training."""
    print(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    async with app.run():
        result = await train.remote.aio(epochs=epochs, batch_size=batch_size, learning_rate=lr)
        print(f"Result: {result}")


async def run_list():
    """List checkpoints."""
    print("Checkpoints:")
    async with app.run():
        result = await list_checkpoints.remote.aio()
        for f in result["files"]:
            print(f"  {f['path']} ({f['size']} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "train", "list"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()

    if args.command == "test":
        asyncio.run(run_test())
    elif args.command == "train":
        asyncio.run(run_train(args.epochs, args.batch_size, args.lr))
    elif args.command == "list":
        asyncio.run(run_list())


if __name__ == "__main__":
    main()
