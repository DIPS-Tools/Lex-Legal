"""Keycloak authentication and role-based FastAPI dependencies."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logger import logger
from keycloak_auth.auth import (
    collect_keycloak_groups,
    decode_keycloak_token,
)


load_dotenv()


@dataclass(frozen=True)
class KeycloakSettings:
    """Keycloak locations, clients, and privileged role names."""

    issuer: str
    token_url: str
    jwks_url: str
    login_client_id: str
    login_client_secret: str
    api_client_id: str
    audience: str
    algorithms: tuple[str, ...]
    admin_roles: frozenset[str]

    @classmethod
    def from_environment(cls) -> "KeycloakSettings":
        issuer = (os.getenv("KEYCLOAK_ISSUER") or "").rstrip("/")
        api_client_id = (
            os.getenv("KEYCLOAK_CLIENT_ID") or "lex-legal-api"
        ).strip()
        return cls(
            issuer=issuer,
            token_url=os.getenv("KEYCLOAK_TOKEN_URL")
            or (
                f"{issuer}/protocol/openid-connect/token"
                if issuer
                else ""
            ),
            jwks_url=os.getenv("KEYCLOAK_JWKS_URL")
            or (
                f"{issuer}/protocol/openid-connect/certs"
                if issuer
                else ""
            ),
            login_client_id=(
                os.getenv("KEYCLOAK_LOGIN_CLIENT_ID")
                or "lex-legal-web"
            ).strip(),
            login_client_secret=os.getenv(
                "KEYCLOAK_LOGIN_CLIENT_SECRET",
                "",
            ),
            api_client_id=api_client_id,
            audience=(
                os.getenv("KEYCLOAK_AUDIENCE") or api_client_id
            ).strip(),
            algorithms=tuple(
                value.strip()
                for value in os.getenv(
                    "KEYCLOAK_ALGORITHMS",
                    "RS256",
                ).split(",")
                if value.strip()
            ),
            admin_roles=frozenset(
                value.strip().casefold()
                for value in os.getenv(
                    "LEXLEGAL_ADMIN_ROLES",
                    "admin,lexlegal-admin",
                ).split(",")
                if value.strip()
            ),
        )


@dataclass(frozen=True)
class AuthenticatedUser:
    """Verified identity used by Lex-Legal API authorization checks."""

    subject: str
    username: str
    email: str
    roles: tuple[str, ...]
    groups: tuple[str, ...]
    is_admin: bool


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    subject: str
    username: str
    email: str = ""
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    is_admin: bool = False


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_in: int
    expires_at: int
    user: UserResponse


class LogoutResponse(BaseModel):
    status: str = "logged_out"


keycloak_settings = KeycloakSettings.from_environment()
auth_router = APIRouter(prefix="/auth")


def _configuration_error(settings: KeycloakSettings) -> None:
    if not (
        settings.issuer
        and settings.token_url
        and settings.jwks_url
        and settings.login_client_id
        and settings.api_client_id
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keycloak authentication is not configured.",
        )


def identity_from_claims(
    claims: dict[str, Any],
    *,
    settings: KeycloakSettings = keycloak_settings,
) -> AuthenticatedUser:
    """Map verified claims to the authorization identity used by the APIs."""
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The access token does not identify a user.",
        )
    username = str(
        claims.get("preferred_username")
        or claims.get("email")
        or subject
    ).strip()
    role_values: set[str] = {
        str(role)
        for role in (claims.get("realm_access") or {}).get("roles", [])
        if role
    }
    resource_access = claims.get("resource_access") or {}
    for client_id in (settings.api_client_id, settings.login_client_id):
        for role in (resource_access.get(client_id) or {}).get("roles", []):
            if role:
                role_values.add(str(role))
    roles = tuple(sorted(role_values))
    normalized_roles = {role.casefold() for role in roles}
    return AuthenticatedUser(
        subject=subject,
        username=username,
        email=str(claims.get("email") or "").strip(),
        roles=roles,
        groups=tuple(collect_keycloak_groups(claims)),
        is_admin=bool(normalized_roles.intersection(settings.admin_roles)),
    )


def identity_for_token(
    token: str,
    *,
    settings: KeycloakSettings = keycloak_settings,
) -> AuthenticatedUser:
    """Verify a Keycloak access token and return its Lex-Legal identity."""
    _configuration_error(settings)
    claims = decode_keycloak_token(
        token,
        issuer=settings.issuer,
        jwks_url=settings.jwks_url,
        audience=settings.audience or None,
        algorithms=list(settings.algorithms),
        logger=logger,
        verify_aud=bool(settings.audience),
    )
    return identity_from_claims(claims, settings=settings)


def require_authenticated_user(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthenticatedUser:
    """Require a valid bearer token for a protected API operation."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.casefold() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity_for_token(token.strip())


def require_admin_user(
    identity: Annotated[
        AuthenticatedUser,
        Depends(require_authenticated_user),
    ],
) -> AuthenticatedUser:
    """Require a configured Keycloak administrator role."""
    if not identity.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )
    return identity


def _user_response(identity: AuthenticatedUser) -> UserResponse:
    return UserResponse(
        subject=identity.subject,
        username=identity.username,
        email=identity.email,
        roles=list(identity.roles),
        groups=list(identity.groups),
        is_admin=identity.is_admin,
    )


@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """Exchange a username and password for Keycloak access tokens."""
    username = request.username.strip()
    if not username or not request.password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please enter both username/email and password.",
        )
    settings = keycloak_settings
    _configuration_error(settings)
    form_data = {
        "client_id": settings.login_client_id,
        "grant_type": "password",
        "scope": "openid profile email",
        "username": username,
        "password": request.password,
    }
    if settings.login_client_secret:
        form_data["client_secret"] = settings.login_client_secret

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings.token_url, data=form_data)
    except httpx.RequestError as exc:
        logger.error(f"Keycloak login service unavailable: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The login service is unavailable. Please try again later.",
        ) from exc

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                error_payload.get("error_description")
                or "Invalid username or password."
            ),
        )

    token_data = response.json()
    access_token = str(token_data.get("access_token") or "")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak did not return an access token.",
        )
    identity = identity_for_token(access_token)
    expires_in = int(token_data.get("expires_in") or 0)
    logger.info(
        f"Keycloak login succeeded: username={identity.username}, "
        f"is_admin={identity.is_admin}"
    )
    return LoginResponse(
        access_token=access_token,
        refresh_token=token_data.get("refresh_token"),
        token_type=str(token_data.get("token_type") or "Bearer"),
        expires_in=expires_in,
        expires_at=int((time.time() + expires_in) * 1000),
        user=_user_response(identity),
    )


@auth_router.get("/me", response_model=UserResponse)
async def current_user(
    identity: Annotated[
        AuthenticatedUser,
        Depends(require_authenticated_user),
    ],
) -> UserResponse:
    """Return the identity represented by the current bearer token."""
    return _user_response(identity)


@auth_router.post("/logout", response_model=LogoutResponse)
async def logout(
    _identity: Annotated[
        AuthenticatedUser,
        Depends(require_authenticated_user),
    ],
) -> LogoutResponse:
    """Validate logout; the browser removes its locally held tokens."""
    return LogoutResponse()
