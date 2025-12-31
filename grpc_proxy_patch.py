"""
Monkey-patch grpclib to use HTTP CONNECT proxy.
Import this before importing modal.
"""

import os
import socket
import ssl as ssl_module
import base64
import asyncio
from urllib.parse import urlparse


def get_proxy_settings():
    """Get proxy settings from environment."""
    proxy_url = os.environ.get('HTTPS_PROXY', '')
    if not proxy_url:
        return None, None, None

    parsed = urlparse(proxy_url)
    proxy_host = parsed.hostname
    proxy_port = parsed.port
    proxy_auth = None

    if parsed.username and parsed.password:
        auth_string = f"{parsed.username}:{parsed.password}"
        proxy_auth = base64.b64encode(auth_string.encode()).decode()

    return proxy_host, proxy_port, proxy_auth


class ProxyEventLoop(asyncio.SelectorEventLoop):
    """Event loop that routes connections through HTTP CONNECT proxy."""

    async def create_connection(self, protocol_factory, host=None, port=None, *,
                                ssl=None, family=0, proto=0, flags=0, sock=None,
                                local_addr=None, server_hostname=None,
                                ssl_handshake_timeout=None,
                                ssl_shutdown_timeout=None,
                                happy_eyeballs_delay=None, interleave=None):
        """Create connection, possibly through HTTP proxy."""

        proxy_host, proxy_port, proxy_auth = get_proxy_settings()

        # Only proxy connections to modal.com
        if proxy_host and host and 'modal.com' in str(host):
            try:
                return await self._create_proxy_connection(
                    protocol_factory, host, port,
                    ssl=ssl, server_hostname=server_hostname,
                    proxy_host=proxy_host, proxy_port=proxy_port,
                    proxy_auth=proxy_auth
                )
            except Exception as e:
                print(f"Proxy connection failed: {e}, falling back to direct")

        # Use parent implementation for non-proxied connections
        return await super().create_connection(
            protocol_factory, host, port,
            ssl=ssl, family=family, proto=proto, flags=flags, sock=sock,
            local_addr=local_addr, server_hostname=server_hostname,
            ssl_handshake_timeout=ssl_handshake_timeout,
            ssl_shutdown_timeout=ssl_shutdown_timeout,
            happy_eyeballs_delay=happy_eyeballs_delay, interleave=interleave
        )

    async def _create_proxy_connection(self, protocol_factory, host, port, *,
                                        ssl, server_hostname, proxy_host,
                                        proxy_port, proxy_auth):
        """Create connection through HTTP CONNECT proxy."""

        # Connect to proxy using synchronous socket first
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)

        try:
            await self.sock_connect(sock, (proxy_host, proxy_port))
        except Exception as e:
            sock.close()
            raise ConnectionError(f"Failed to connect to proxy: {e}")

        # Send CONNECT request
        connect_request = f"CONNECT {host}:{port} HTTP/1.1\r\n"
        connect_request += f"Host: {host}:{port}\r\n"
        if proxy_auth:
            connect_request += f"Proxy-Authorization: Basic {proxy_auth}\r\n"
        connect_request += "\r\n"

        await self.sock_sendall(sock, connect_request.encode())

        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = await self.sock_recv(sock, 4096)
            if not chunk:
                sock.close()
                raise ConnectionError("Proxy closed connection")
            response += chunk

        if b'200' not in response.split(b'\r\n')[0]:
            sock.close()
            first_line = response.split(b'\r\n')[0].decode()
            raise ConnectionError(f"Proxy CONNECT failed: {first_line}")

        # Upgrade to SSL if needed
        if ssl:
            ssl_context = ssl if isinstance(ssl, ssl_module.SSLContext) else ssl_module.create_default_context()

            # Wrap socket with SSL
            ssl_sock = ssl_context.wrap_socket(
                sock,
                server_hostname=server_hostname or host,
                do_handshake_on_connect=False
            )

            # Do SSL handshake asynchronously
            ssl_sock.setblocking(False)
            while True:
                try:
                    ssl_sock.do_handshake()
                    break
                except ssl_module.SSLWantReadError:
                    await asyncio.sleep(0.01)
                except ssl_module.SSLWantWriteError:
                    await asyncio.sleep(0.01)

            sock = ssl_sock

        # Create protocol and transport
        protocol = protocol_factory()
        transport = await self._make_socket_transport(sock, protocol, ssl)

        return transport, protocol


def patch_event_loop():
    """Replace the default event loop with our proxy-aware version."""
    # This approach patches at the event loop policy level
    pass


def patch_grpclib():
    """Patch grpclib Channel to use HTTP CONNECT proxy."""
    import grpclib.client

    original_create_connection = grpclib.client.Channel._create_connection

    async def patched_create_connection(self):
        proxy_host, proxy_port, proxy_auth = get_proxy_settings()

        if not proxy_host:
            return await original_create_connection(self)

        host = self._host
        port = self._port

        # Determine server hostname for SSL
        server_hostname = host
        if hasattr(self, '_config') and self._config and hasattr(self._config, 'ssl_target_name_override'):
            if self._config.ssl_target_name_override:
                server_hostname = self._config.ssl_target_name_override

        # Connect to proxy
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((proxy_host, proxy_port))

        # Send CONNECT request
        connect_request = f"CONNECT {host}:{port} HTTP/1.1\r\n"
        connect_request += f"Host: {host}:{port}\r\n"
        if proxy_auth:
            connect_request += f"Proxy-Authorization: Basic {proxy_auth}\r\n"
        connect_request += "\r\n"

        sock.sendall(connect_request.encode())

        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Proxy closed connection")
            response += chunk

        if b'200' not in response.split(b'\r\n')[0]:
            first_line = response.split(b'\r\n')[0].decode()
            raise ConnectionError(f"Proxy CONNECT failed: {first_line}")

        # Upgrade to SSL
        # Note: The proxy does SSL interception, so we disable certificate verification
        ssl_context = ssl_module.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl_module.CERT_NONE

        ssl_sock = ssl_context.wrap_socket(
            sock,
            server_hostname=server_hostname
        )

        # Make non-blocking for asyncio
        ssl_sock.setblocking(False)

        # Create the protocol
        protocol = self._protocol_factory()

        # Create transport for already-SSL-wrapped socket
        loop = self._loop or asyncio.get_event_loop()

        # Manually create socket transport using internal asyncio method
        # Since socket is already SSL-wrapped, we use _make_socket_transport not _make_ssl_transport
        waiter = loop.create_future()

        # Access the underlying raw socket for asyncio transport
        # SSLSocket wraps the real socket, we need to work with it properly
        from asyncio.selector_events import _SelectorSocketTransport
        transport = _SelectorSocketTransport(
            loop, ssl_sock, protocol, waiter, None
        )

        try:
            await waiter
        except:
            transport.close()
            raise

        return protocol

    grpclib.client.Channel._create_connection = patched_create_connection
    print("grpclib patched for HTTP proxy support")


# Auto-patch when imported
patch_grpclib()
