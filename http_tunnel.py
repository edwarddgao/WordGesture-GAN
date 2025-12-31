#!/usr/bin/env python3
"""
HTTP CONNECT tunnel for gRPC connections.
Allows gRPC traffic to go through an HTTP proxy.
"""

import asyncio
import os
import socket
import base64
from urllib.parse import urlparse

PROXY_URL = os.environ.get('HTTPS_PROXY', '')
LOCAL_PORT = 9999
TARGET_HOST = 'api.modal.com'
TARGET_PORT = 443


async def create_tunnel(reader, writer, proxy_host, proxy_port, proxy_auth, target_host, target_port):
    """Create HTTP CONNECT tunnel through proxy."""
    # Connect to proxy
    proxy_reader, proxy_writer = await asyncio.open_connection(proxy_host, proxy_port)

    # Send CONNECT request
    connect_request = f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
    connect_request += f"Host: {target_host}:{target_port}\r\n"
    if proxy_auth:
        connect_request += f"Proxy-Authorization: Basic {proxy_auth}\r\n"
    connect_request += "\r\n"

    proxy_writer.write(connect_request.encode())
    await proxy_writer.drain()

    # Read response
    response = await proxy_reader.readline()
    if b'200' not in response:
        print(f"Tunnel failed: {response}")
        writer.close()
        proxy_writer.close()
        return

    # Read remaining headers
    while True:
        line = await proxy_reader.readline()
        if line == b'\r\n':
            break

    # Now relay data between client and proxy
    async def relay(src, dst):
        try:
            while True:
                data = await src.read(8192)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except:
            pass
        finally:
            try:
                dst.close()
            except:
                pass

    await asyncio.gather(
        relay(reader, proxy_writer),
        relay(proxy_reader, writer)
    )


async def handle_client(reader, writer):
    """Handle incoming connection."""
    parsed = urlparse(PROXY_URL)
    proxy_host = parsed.hostname
    proxy_port = parsed.port

    # Extract auth if present
    proxy_auth = None
    if parsed.username and parsed.password:
        auth_string = f"{parsed.username}:{parsed.password}"
        proxy_auth = base64.b64encode(auth_string.encode()).decode()

    await create_tunnel(reader, writer, proxy_host, proxy_port, proxy_auth, TARGET_HOST, TARGET_PORT)


async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', LOCAL_PORT)
    print(f"Tunnel listening on 127.0.0.1:{LOCAL_PORT}")
    print(f"Tunneling to {TARGET_HOST}:{TARGET_PORT} via {urlparse(PROXY_URL).hostname}")
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
