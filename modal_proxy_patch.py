"""Patch grpclib to work through HTTP proxy for Modal."""
import os
import ssl
import asyncio
import urllib.parse
import grpclib.client
from python_socks.sync import Proxy as SyncProxy
from python_socks import ProxyType

def apply_proxy_patch():
    """Apply monkey-patch to grpclib to route through HTTP proxy."""
    proxy_url = os.environ.get('HTTPS_PROXY', '')
    if not proxy_url:
        print("No HTTPS_PROXY set, skipping patch")
        return False

    parsed = urllib.parse.urlparse(proxy_url)

    # Store original method
    original_create_connection = grpclib.client.Channel._create_connection

    async def patched_create_connection(self):
        """Patched _create_connection that uses HTTP proxy."""
        if self._path is not None:
            # Unix socket - use original
            return await original_create_connection(self)

        # Create sync proxy connection to get raw socket
        sync_proxy = SyncProxy(
            proxy_type=ProxyType.HTTP,
            host=parsed.hostname,
            port=parsed.port,
            username=parsed.username,
            password=parsed.password
        )

        # Connect through proxy (sync to get raw socket)
        raw_sock = sync_proxy.connect(
            dest_host=self._host,
            dest_port=self._port,
            timeout=30
        )

        # Make non-blocking for asyncio
        raw_sock.setblocking(False)

        # Create SSL context if needed
        if self._ssl:
            ssl_context = ssl.create_default_context()
            server_hostname = self._config.ssl_target_name_override if self._ssl else None
            if server_hostname is None:
                server_hostname = self._host
        else:
            ssl_context = None
            server_hostname = None

        # Create asyncio transport/protocol using the connected socket
        # Let asyncio handle SSL wrapping
        loop = asyncio.get_event_loop()
        _, protocol = await loop.create_connection(
            self._protocol_factory,
            sock=raw_sock,
            ssl=ssl_context,
            server_hostname=server_hostname
        )

        return protocol

    # Apply the patch
    grpclib.client.Channel._create_connection = patched_create_connection
    print("[modal_proxy_patch] grpclib patched to use HTTP proxy")
    return True

# Auto-apply patch on import
_patched = apply_proxy_patch()
