#!/usr/bin/env python3
"""Absolutely minimal Modal test - just return a value."""

import os
if not os.environ.get('MODAL_IS_REMOTE'):
    import modal_proxy_patch  # Only patch locally, not in Modal container
import modal
import asyncio

app = modal.App('minimal-test')

@app.function(timeout=60)
def just_return():
    """Return immediately with no side effects."""
    return 42


async def main():
    print("Starting minimal test...")
    async with app.run():
        print("Calling just_return...")
        result = await just_return.remote.aio()
        print(f"Result: {result}")
    return result


if __name__ == '__main__':
    asyncio.run(main())
