#!/usr/bin/env python3
"""Test Modal CPU function first."""
import os
import asyncio

# Fix token format
token_id = os.environ.get('MODAL_TOKEN_ID', '').strip('"')
if not token_id.startswith('ak-'):
    token_id = 'ak-' + token_id
os.environ['MODAL_TOKEN_ID'] = token_id
os.environ['MODAL_TOKEN_SECRET'] = os.environ.get('MODAL_TOKEN_SECRET', '').strip('"')

# Apply proxy patch
import modal_proxy_patch

import modal

app = modal.App("test-cpu")

@app.function(timeout=60)
def hello():
    import sys
    return f"Hello from Modal! Python {sys.version}"

async def main():
    print("Starting Modal app (CPU only)...")
    async with app.run():
        print("Calling CPU function...")
        result = await hello.remote.aio()
        print(f"✓ CPU Test Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
