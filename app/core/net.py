"""SSRF guard for outbound URLs configured by admins (HTTP and MCP adapters)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeUrl(ValueError):
    pass


def _is_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url(url: str, *, allow_private: bool, resolve: bool = True) -> None:
    """Raise UnsafeUrl for non-http(s) URLs, or URLs that point at private/loopback ranges."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrl(f"Only absolute http(s) URLs are allowed: {url!r}")
    if allow_private:
        return
    host = parsed.hostname
    if host.lower() in {"localhost", "metadata.google.internal"}:
        raise UnsafeUrl(f"Host {host!r} is not allowed")
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        if not resolve:
            return
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeUrl(f"Host {host!r} does not resolve") from exc
        ips = [ipaddress.ip_address(str(info[4][0])) for info in infos]
    for ip in ips:
        if _is_private(ip):
            raise UnsafeUrl(f"Host {host!r} resolves to a private address ({ip})")
