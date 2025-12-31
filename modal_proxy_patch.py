"""Patch grpclib and aiohttp to work through HTTP proxy for Modal."""
import os
import ssl
import asyncio
import urllib.parse
import grpclib.client
import aiohttp
from python_socks.sync import Proxy as SyncProxy
from python_socks import ProxyType
from aiohttp_socks import ProxyConnector


def _create_proxy_ssl_context():
    """Create SSL context that works with proxy CA certificates."""
    ctx = ssl.create_default_context()
    # Load system CA bundle which includes proxy certificates
    ctx.load_verify_locations(cafile='/etc/ssl/certs/ca-certificates.crt')
    # Set HTTP/2 ALPN protocols
    ctx.set_alpn_protocols(['h2'])
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
    return ctx


def apply_grpclib_patch(proxy_url: str):
    """Apply monkey-patch to grpclib to route through HTTP proxy."""
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

        # Determine SSL context and server hostname
        # Create a custom SSL context with proxy CA and H2 ALPN
        if self._ssl is not None:
            ssl_context = _create_proxy_ssl_context()
            server_hostname = self._config.ssl_target_name_override
            if server_hostname is None:
                server_hostname = self._host
        else:
            ssl_context = None
            server_hostname = None

        # Create asyncio transport/protocol using the connected socket
        # Use self._loop to match the original implementation
        _, protocol = await self._loop.create_connection(
            self._protocol_factory,
            sock=raw_sock,
            ssl=ssl_context,
            server_hostname=server_hostname
        )

        return protocol

    # Apply the patch
    grpclib.client.Channel._create_connection = patched_create_connection
    print("[modal_proxy_patch] grpclib patched to use HTTP proxy")


def apply_aiohttp_patch(proxy_url: str):
    """Patch Modal's http_utils to use HTTP proxy."""
    # Patch Modal's _http_client_with_tls function
    import modal._utils.http_utils as http_utils

    def patched_http_client_with_tls(timeout):
        """Create HTTP client that routes through proxy."""
        import ssl
        from aiohttp import ClientSession, ClientTimeout

        # Use system CA certificates which include proxy certificates
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(cafile='/etc/ssl/certs/ca-certificates.crt')

        connector = ProxyConnector.from_url(proxy_url, ssl=ssl_context)
        return ClientSession(connector=connector, timeout=ClientTimeout(total=timeout))

    http_utils._http_client_with_tls = patched_http_client_with_tls
    print("[modal_proxy_patch] Modal http_utils patched to use HTTP proxy")


def apply_proxy_patch():
    """Apply all necessary proxy patches for Modal."""
    proxy_url = os.environ.get('HTTPS_PROXY', '')
    if not proxy_url:
        print("No HTTPS_PROXY set, skipping patch")
        return False

    apply_grpclib_patch(proxy_url)
    apply_aiohttp_patch(proxy_url)
    return True


# Auto-apply patch on import
_patched = apply_proxy_patch()
