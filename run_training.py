#!/usr/bin/env python3
"""Run WordGesture-GAN training on Modal cloud GPUs.

This script handles the proxy patch for Claude Code environment
and then runs the training functions on Modal.
"""
import os
import sys
import asyncio
import argparse

# Fix token format for Modal authentication
token_id = os.environ.get('MODAL_TOKEN_ID', '').strip('"')
if not token_id.startswith('ak-'):
    token_id = 'ak-' + token_id
os.environ['MODAL_TOKEN_ID'] = token_id
os.environ['MODAL_TOKEN_SECRET'] = os.environ.get('MODAL_TOKEN_SECRET', '').strip('"')

# Apply proxy patch BEFORE importing modal
import modal_proxy_patch

# Now import training functions
from modal_train import app, train_epoch, full_training_run, check_volume_contents


async def run_test():
    """Run a test epoch to verify GPU training works."""
    print("Running test epoch on Modal GPU...")
    async with app.run():
        result = await train_epoch.remote.aio(
            epoch=0,
            batch_size=512,
            learning_rate=0.0002,
        )
        print(f"\n✓ Test epoch completed!")
        print(f"  Device: {result['device']}")
        print(f"  GPU: {result.get('gpu_name', 'N/A')}")
        print(f"  Status: {result['status']}")
        if 'test_computation' in result:
            print(f"  Test computation: {result['test_computation']:.2f}")
    return result


async def run_full_training(epochs: int, batch_size: int, learning_rate: float):
    """Run full training on Modal GPU."""
    print(f"Starting full training run...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    async with app.run():
        result = await full_training_run.remote.aio(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print(f"\n✓ Training completed!")
        for key, value in result.items():
            print(f"  {key}: {value}")
    return result


async def check_volume():
    """Check contents of the Modal volume."""
    print("Checking Modal volume contents...")
    async with app.run():
        result = await check_volume_contents.remote.aio()
        print(f"\nVolume: {result['volume_path']}")
        print(f"Files: {result['file_count']}")
        for f in result.get('files', []):
            print(f"  {f['path']} ({f['size']} bytes)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run WordGesture-GAN training on Modal")
    parser.add_argument("command", choices=["test", "train", "check-volume"],
                        help="Command to run")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")

    args = parser.parse_args()

    if args.command == "test":
        asyncio.run(run_test())
    elif args.command == "train":
        asyncio.run(run_full_training(args.epochs, args.batch_size, args.lr))
    elif args.command == "check-volume":
        asyncio.run(check_volume())


if __name__ == "__main__":
    main()
