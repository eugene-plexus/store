"""Who may call what: the gateway names owners, the owner's console reads counts.

**The owner header is believed only beside a gateway's service token.**
`sub: gateway`, minted by this machine for its own gateway, or by another
machine whose key holds the `gateway` grant (`tokens.verify` refuses a
`gateway` token from a machine without it). Any other credential that
sends the header is refused, an operator session included: the install
owner does not read people's objects. An object route without a gateway
token is refused before anything is looked up.

The store routes and the config trio take an operator session addressed
to this machine, as every component's do. In a development run with no
trust bundle, nothing is checked and the header is taken as sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from .auth_state import AuthState
from .keeper import Keeper
from .problems import StoreProblem, problem

_bearer_scheme = HTTPBearer(auto_error=False)

OWNER_HEADER = "X-Eugene-Plexus-Owner"


@dataclass(frozen=True)
class Caller:
    owner: str


def _claims(
    request: Request, creds: HTTPAuthorizationCredentials | None, classes: tuple[str, ...]
) -> tokens.Claims | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    try:
        return auth.verify(creds.credentials, classes=classes)
    except tokens.TokenError as exc:
        raise problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc


def is_gateway(claims: tokens.Claims, recipient: str | None) -> bool:
    """A gateway's service token: this machine's, or a granted machine's.

    `tokens.verify` has already refused a `gateway` token sent off a
    machine whose key lacks the grant, so `sub` is what is left to check;
    the local case is spelled out so a widened grant cannot widen this."""
    if not claims.is_service or claims.sub != tokens.SUB_GATEWAY:
        return False
    return claims.is_local_service(str(recipient)) or claims.issuer_node is not None


def require_owner(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    owner: str | None = Header(default=None, alias=OWNER_HEADER),
) -> Caller:
    """An object route: a gateway's token, and the owner it names."""
    auth: AuthState = request.app.state.auth_state
    claims = _claims(
        request,
        creds,
        (tokens.TYP_SERVICE, tokens.TYP_SESSION, tokens.TYP_CLIENT, tokens.TYP_ADMINISTRATION),
    )
    if claims is not None and not is_gateway(claims, auth.recipient):
        raise problem(
            status.HTTP_403_FORBIDDEN,
            "Only the gateway",
            "Files are kept for their owners and reached through the gateway's doors. "
            f"A {claims.sub!r} credential from {claims.iss!r} may not name an owner here.",
        )
    if not owner:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "No owner",
            f"An object route needs {OWNER_HEADER}, which the gateway sets.",
        )
    try:
        return Caller(Keeper.check_owner(owner))
    except StoreProblem as exc:
        raise exc.http() from exc


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """An operator session only: settings, counts, the sweep, held removals."""
    return _claims(request, creds, (tokens.TYP_SESSION,))


def require_reader(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """The store's status: an operator session, or the gateway (it says
    which parts are on)."""
    auth: AuthState = request.app.state.auth_state
    claims = _claims(request, creds, (tokens.TYP_SESSION, tokens.TYP_SERVICE))
    if claims is not None and claims.is_service and not is_gateway(claims, auth.recipient):
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong audience",
            f"A {claims.sub!r} service token may not read the store's status.",
        )
    return claims
