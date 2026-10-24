"""Verifier bootstrap: the trust bundle this store checks tokens against.

The supervising agent hands this process four things and no key: the
trust bundle file it keeps, the authority that must have signed it, this
machine's recipient name, and a service token of the store's own. With
none of them the store runs unauthenticated (a development run on one
machine); with some but not all it refuses to start rather than run
half-authenticated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import tokens

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    bundle: tokens.BundleFile | None
    """The trust bundle; None when auth is disabled."""

    recipient: str | None
    """This machine, as a token's audience names it."""

    service_token: str | None
    """This store's own token, used to call its local agent."""

    @property
    def auth_disabled(self) -> bool:
        return self.bundle is None

    def verify(self, token: str, *, classes: tuple[str, ...]) -> tokens.Claims:
        if self.bundle is None or self.recipient is None:
            raise tokens.TokenError("this store verifies nothing: it runs without auth")
        bundle = self.bundle.get()
        if bundle is None:
            raise tokens.TokenError(
                "this store holds no trust bundle it can read yet; its agent keeps one "
                "beside node.yaml"
            )
        return tokens.verify(token, bundle=bundle, recipient=self.recipient, classes=classes)


def load_auth_state(
    *,
    trust_bundle_file: str | None,
    trust_authority: str | None,
    auth_recipient: str | None,
    service_token: str | None,
) -> AuthState:
    given = {
        "TRUST_BUNDLE_FILE": trust_bundle_file,
        "TRUST_AUTHORITY": trust_authority,
        "AUTH_RECIPIENT": auth_recipient,
        "SERVICE_TOKEN": service_token,
    }
    present = [name for name, value in given.items() if value]
    if not present:
        log.warning(
            "EUGENE_PLEXUS_STORE_TRUST_BUNDLE_FILE not set -- running unauthenticated "
            "(development mode). The agent always supplies it."
        )
        return AuthState(bundle=None, recipient=None, service_token=None)
    missing = [name for name, value in given.items() if not value]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} missing beside {', '.join(present)}: the agent hands a "
            "child all four or none. Check the supervisor wiring."
        )
    assert trust_bundle_file and trust_authority and auth_recipient and service_token
    try:
        tokens.load_public(trust_authority)
    except ValueError as exc:
        raise ValueError(f"TRUST_AUTHORITY is not an Ed25519 public key: {exc}") from exc
    return AuthState(
        bundle=tokens.BundleFile(trust_bundle_file, authority=trust_authority),
        recipient=auth_recipient,
        service_token=service_token,
    )
