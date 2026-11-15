"""Company secrets: Fernet-encrypted at rest, decrypted only at call time.

`SecretStore` is the seam for KMS / Vault later.
"""

from __future__ import annotations

from typing import Any, Protocol

from cryptography.fernet import Fernet, InvalidToken

from app.core.errors import SecretsUnavailable


class SecretStore(Protocol):
    def encrypt(self, plaintext: str) -> str: ...
    def decrypt(self, ciphertext: str) -> str: ...


class FernetSecretStore:
    def __init__(self, key: str | None) -> None:
        self._fernet = Fernet(key.encode()) if key else None

    def _f(self) -> Fernet:
        if self._fernet is None:
            raise SecretsUnavailable()
        return self._fernet

    def encrypt(self, plaintext: str) -> str:
        return self._f().encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._f().decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise SecretsUnavailable("Secret could not be decrypted with the current key") from exc


def redact(value: Any, secrets: list[str]) -> Any:
    """Replace any secret plaintext occurring in strings (recursively) with '***'."""
    secrets = [s for s in secrets if s]
    if not secrets:
        return value
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, "***")
        return value
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value
