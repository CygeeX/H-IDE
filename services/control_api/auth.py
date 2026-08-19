"""Assistant V2 authentication, CSRF protection, and deterministic RBAC.

The project previously used one deployment-wide Bearer token.  That token is
kept as an explicit legacy developer identity so existing Robot Action Console
tests and internal tooling remain compatible.  Browser users authenticate with
opaque server-side sessions; the raw session secret is never persisted.

The small policy matrix below implements the Casbin RBAC model deterministically
inside the service.  Keeping it local for this migration avoids making process
startup depend on a policy file or optional adapter while preserving the same
subject/role/scope/resource decision boundary for a later PyCasbin adapter.
"""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import hmac
import math
import os
import re
import secrets
from types import MappingProxyType
from typing import Any, Callable, Mapping
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .store import SQLiteStore


SESSION_COOKIE = "hide_session"
CSRF_COOKIE = "hide_csrf"
CSRF_HEADER = "x-csrf-token"
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_USERNAME_PATTERN = re.compile(r"^[\w.-]+$", flags=re.UNICODE)


class Role(StrEnum):
    USER = "USER"
    DEVELOPER = "DEVELOPER"
    ADMIN = "ADMIN"


class Scope(StrEnum):
    API_ACCESS = "api:access"
    ROBOTS_READ = "robots:read"
    ASSISTANT_USE = "assistant:use"
    SPEECH_USE = "speech:use"
    MISSIONS_USE = "missions:use"
    EVENTS_READ = "events:read"
    COMMANDS_MANAGE = "commands:manage"
    MAPPINGS_MANAGE = "mappings:manage"
    SIMULATIONS_MANAGE = "simulations:manage"
    DIAGNOSTICS_READ = "diagnostics:read"
    USERS_MANAGE = "users:manage"


_USER_SCOPES = frozenset(
    {
        Scope.API_ACCESS,
        Scope.ROBOTS_READ,
        Scope.ASSISTANT_USE,
        Scope.SPEECH_USE,
        Scope.MISSIONS_USE,
        Scope.EVENTS_READ,
    }
)
_DEVELOPER_SCOPES = _USER_SCOPES | frozenset(
    {
        Scope.COMMANDS_MANAGE,
        Scope.MAPPINGS_MANAGE,
        Scope.SIMULATIONS_MANAGE,
        Scope.DIAGNOSTICS_READ,
    }
)
ROLE_SCOPES: Mapping[Role, frozenset[Scope]] = MappingProxyType(
    {
        Role.USER: _USER_SCOPES,
        Role.DEVELOPER: _DEVELOPER_SCOPES,
        Role.ADMIN: _DEVELOPER_SCOPES | frozenset({Scope.USERS_MANAGE}),
    }
)


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str
    display_name: str
    role: Role
    site_id: str
    authentication_method: str
    session_id: str | None = None

    @property
    def scopes(self) -> frozenset[Scope]:
        return ROLE_SCOPES[self.role]

    @property
    def is_legacy(self) -> bool:
        return self.authentication_method == "legacy_bearer"

    def public(self) -> dict[str, Any]:
        return {
            "userId": self.user_id,
            "username": self.username,
            "displayName": self.display_name,
            "role": self.role.value,
            "siteId": self.site_id,
            "authenticationMethod": self.authentication_method,
            "scopes": sorted(scope.value for scope in self.scopes),
        }


class AuthorizationDenied(PermissionError):
    """The authenticated principal lacks a policy grant."""


class CSRFValidationFailed(AuthorizationDenied):
    """A valid cookie session did not present its double-submit token."""


class DeterministicRBACPolicy:
    """A fail-closed Casbin-equivalent role/scope/resource policy."""

    def require(
        self,
        principal: Principal,
        scope: Scope,
        *,
        owner_user_id: str | None = None,
        site_id: str | None = None,
        resource_is_bound: bool = True,
    ) -> None:
        if scope not in principal.scopes:
            raise AuthorizationDenied(f"scope {scope.value} is required")
        if principal.role is Role.ADMIN or principal.is_legacy:
            return
        if not resource_is_bound:
            # Pre-authentication data has no trustworthy tenant binding.
            # Developers may diagnose it; normal users may not claim/read it.
            if principal.role is Role.USER:
                raise AuthorizationDenied("resource has no authenticated owner")
            return
        if site_id is not None and site_id != principal.site_id:
            raise AuthorizationDenied("resource belongs to another site")
        if (
            principal.role is Role.USER
            and owner_user_id is not None
            and owner_user_id != principal.user_id
        ):
            raise AuthorizationDenied("resource belongs to another user")


def _encoded(value: bytes) -> str:
    return urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decoded(value: str) -> bytes:
    return urlsafe_b64decode(value + "=" * (-len(value) % 4))


class ScryptPasswordHasher:
    """Versioned stdlib scrypt hashes with per-password random salts."""

    n = 2**14
    r = 8
    p = 1
    dklen = 32
    maxmem = 64 * 1024 * 1024

    def __init__(self) -> None:
        # Unknown users and corrupt rows still pay one normal KDF operation.
        self._dummy_salt = secrets.token_bytes(16)
        self._dummy_digest = self._derive(b"invalid-password", self._dummy_salt)

    def _derive(self, password: bytes, salt: bytes) -> bytes:
        return hashlib.scrypt(
            password,
            salt=salt,
            n=self.n,
            r=self.r,
            p=self.p,
            dklen=self.dklen,
            maxmem=self.maxmem,
        )

    @staticmethod
    def _password_bytes(password: str) -> bytes:
        if not isinstance(password, str):
            raise ValueError("password must be a string")
        encoded = password.encode("utf-8")
        if not 12 <= len(encoded) <= 1_024:
            raise ValueError("password must contain 12 to 1024 UTF-8 bytes")
        if "\x00" in password:
            raise ValueError("password contains an unsupported NUL character")
        return encoded

    def hash(self, password: str) -> str:
        encoded = self._password_bytes(password)
        salt = secrets.token_bytes(16)
        digest = self._derive(encoded, salt)
        return (
            f"scrypt$n={self.n},r={self.r},p={self.p},dklen={self.dklen}"
            f"${_encoded(salt)}${_encoded(digest)}"
        )

    def verify(self, password: str, encoded_hash: str | None) -> bool:
        try:
            password_bytes = password.encode("utf-8")
            if len(password_bytes) > 1_024:
                raise ValueError("password is too large")
        except (AttributeError, UnicodeError, ValueError):
            password_bytes = b"invalid-password"
        valid_format = True
        try:
            algorithm, parameters, salt_value, digest_value = str(encoded_hash).split("$")
            parsed = {
                key: int(value)
                for key, value in (item.split("=", 1) for item in parameters.split(","))
            }
            if algorithm != "scrypt" or parsed != {
                "n": self.n,
                "r": self.r,
                "p": self.p,
                "dklen": self.dklen,
            }:
                raise ValueError("unsupported password hash")
            salt = _decoded(salt_value)
            expected = _decoded(digest_value)
            if len(salt) != 16 or len(expected) != self.dklen:
                raise ValueError("invalid password hash length")
        except (TypeError, ValueError, KeyError):
            valid_format = False
            salt = self._dummy_salt
            expected = self._dummy_digest
        actual = self._derive(password_bytes, salt)
        return valid_format and hmac.compare_digest(actual, expected)


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean")


def _clean_username(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("username is required")
    username = value.strip().casefold()
    if not 3 <= len(username) <= 64 or not _USERNAME_PATTERN.fullmatch(username):
        raise ValueError("username must be 3-64 letters, numbers, '_', '.', or '-'")
    return username


def _clean_display_name(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
        raise ValueError("displayName must be a non-empty string up to 100 characters")
    return value.strip()


def _clean_site_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
        raise ValueError("siteId must be a non-empty string up to 100 characters")
    if any(character.isspace() or ord(character) < 32 for character in value.strip()):
        raise ValueError("siteId must not contain whitespace or control characters")
    if value.strip() == "*":
        raise ValueError("siteId '*' is reserved for the legacy internal identity")
    return value.strip()


class AuthService:
    def __init__(
        self,
        store: SQLiteStore,
        *,
        legacy_bearer_token: str,
        session_ttl_seconds: float | None = None,
        cookie_secure: bool | None = None,
        default_site_id: str | None = None,
        registration_enabled: bool | None = None,
    ) -> None:
        self.store = store
        self.legacy_bearer_token = legacy_bearer_token
        raw_ttl: Any = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else os.getenv("AUTH_SESSION_TTL_SECONDS", str(7 * 24 * 60 * 60))
        )
        try:
            self.session_ttl_seconds = float(raw_ttl)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("AUTH_SESSION_TTL_SECONDS must be a number") from exc
        if (
            not math.isfinite(self.session_ttl_seconds)
            or not 300 <= self.session_ttl_seconds <= 31 * 24 * 60 * 60
        ):
            raise ValueError("AUTH_SESSION_TTL_SECONDS must be between 300 seconds and 31 days")
        self.cookie_secure = (
            _bool_env("AUTH_COOKIE_SECURE", default=False)
            if cookie_secure is None
            else bool(cookie_secure)
        )
        self.registration_enabled = (
            _bool_env("AUTH_REGISTRATION_ENABLED", default=True)
            if registration_enabled is None
            else bool(registration_enabled)
        )
        self.default_site_id = _clean_site_id(
            default_site_id or os.getenv("AUTH_DEFAULT_SITE_ID", "default")
        )
        self.password_hasher = ScryptPasswordHasher()
        self.policy = DeterministicRBACPolicy()

    @staticmethod
    def _secret_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _user_agent_hash(request: Request) -> str | None:
        value = request.headers.get("user-agent")
        return None if not value else hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _remote_address(request: Request) -> str | None:
        return None if request.client is None else request.client.host

    def _principal_from_user(
        self,
        user: Mapping[str, Any],
        *,
        authentication_method: str,
        session_id: str | None = None,
    ) -> Principal:
        return Principal(
            user_id=str(user["userId"]),
            username=str(user["username"]),
            display_name=str(user.get("displayName") or user["username"]),
            role=Role(str(user["role"])),
            site_id=str(user["siteId"]),
            authentication_method=authentication_method,
            session_id=session_id,
        )

    def register(
        self,
        body: Mapping[str, Any],
        *,
        role: Role = Role.USER,
        site_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.registration_enabled and role is Role.USER:
            raise AuthorizationDenied("self-registration is disabled")
        if "role" in body or "siteId" in body or "site_id" in body:
            raise ValueError("public registration cannot choose role or site")
        username = _clean_username(body.get("username"))
        display_name = _clean_display_name(body.get("displayName"), fallback=username)
        password_hash = self.password_hasher.hash(body.get("password"))
        return self.store.create_user(
            user_id=f"user-{uuid.uuid4().hex}",
            username=username,
            display_name=display_name,
            password_hash=password_hash,
            role=role.value,
            site_id=_clean_site_id(site_id or self.default_site_id),
        )

    def bootstrap_admin_from_env(self) -> dict[str, Any] | None:
        username = os.getenv("AUTH_BOOTSTRAP_ADMIN_USERNAME", "").strip()
        password = os.getenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", "")
        if not username and not password:
            return None
        if not username or not password:
            raise ValueError(
                "AUTH_BOOTSTRAP_ADMIN_USERNAME and AUTH_BOOTSTRAP_ADMIN_PASSWORD "
                "must be configured together"
            )
        existing = self.store.get_user_by_username(_clean_username(username))
        if existing is not None:
            if existing.get("role") != Role.ADMIN.value:
                raise ValueError(
                    "bootstrap administrator username is already owned by a non-admin user"
                )
            return existing
        if self.store.count_users() != 0:
            raise ValueError("bootstrap admin is missing but users already exist")
        body = {
            "username": username,
            "password": password,
            "displayName": os.getenv("AUTH_BOOTSTRAP_ADMIN_DISPLAY_NAME", "Administrator"),
        }
        # Bootstrap is an explicit deployment operation, not public signup.
        registration_enabled = self.registration_enabled
        try:
            self.registration_enabled = True
            return self.register(body, role=Role.ADMIN, site_id=self.default_site_id)
        finally:
            self.registration_enabled = registration_enabled

    def login(self, body: Mapping[str, Any], request: Request) -> tuple[Principal, str, str]:
        try:
            username = _clean_username(body.get("username"))
        except ValueError:
            username = str(body.get("username") or "")[:64].strip().casefold()
        password = body.get("password")
        if not isinstance(password, str):
            password = ""
        user = self.store.get_user_by_username(username, include_password=True)
        verified = self.password_hasher.verify(
            password,
            None if user is None else user.get("passwordHash"),
        )
        if user is None or not verified or user.get("status") != "active":
            self.store.record_login_audit(
                username=username or "<invalid>",
                user_id=None if user is None else str(user["userId"]),
                outcome="denied",
                remote_address=self._remote_address(request),
                user_agent_hash=self._user_agent_hash(request),
                detail={"reason": "invalid_credentials_or_disabled"},
            )
            raise AuthorizationDenied("invalid username or password")
        session_secret = secrets.token_urlsafe(32)
        csrf_secret = secrets.token_urlsafe(32)
        session_id = f"session-{uuid.uuid4().hex}"
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.session_ttl_seconds)
        self.store.create_session(
            session_id=session_id,
            user_id=str(user["userId"]),
            token_hash=self._secret_hash(session_secret),
            csrf_token_hash=self._secret_hash(csrf_secret),
            expires_at=expires_at.isoformat(),
            user_agent_hash=self._user_agent_hash(request),
        )
        self.store.record_login_audit(
            username=username,
            user_id=str(user["userId"]),
            outcome="succeeded",
            remote_address=self._remote_address(request),
            user_agent_hash=self._user_agent_hash(request),
        )
        return (
            self._principal_from_user(
                user,
                authentication_method="session",
                session_id=session_id,
            ),
            session_secret,
            csrf_secret,
        )

    def _legacy_principal(self) -> Principal:
        return Principal(
            user_id="legacy-developer",
            username="legacy-developer",
            display_name="Legacy developer token",
            role=Role.DEVELOPER,
            site_id="*",
            authentication_method="legacy_bearer",
        )

    def _authenticate_session(self, token: str) -> tuple[Principal, Mapping[str, Any]]:
        if not token or len(token) > 512:
            raise AuthorizationDenied("authentication required")
        record = self.store.get_session_by_token_hash(self._secret_hash(token))
        if record is None or record.get("revokedAt") is not None:
            raise AuthorizationDenied("authentication required")
        try:
            expires_at = datetime.fromisoformat(str(record["expiresAt"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthorizationDenied("authentication required") from exc
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            self.store.revoke_session(str(record["sessionId"]))
            raise AuthorizationDenied("session has expired")
        if record.get("userStatus") != "active":
            raise AuthorizationDenied("user is disabled")
        principal = Principal(
            user_id=str(record["userId"]),
            username=str(record["username"]),
            display_name=str(record["displayName"]),
            role=Role(str(record["role"])),
            site_id=str(record["siteId"]),
            authentication_method="session",
            session_id=str(record["sessionId"]),
        )
        return principal, record

    def authenticate_request(self, request: Request) -> Principal:
        cached = getattr(request.state, "principal", None)
        if isinstance(cached, Principal):
            return cached
        authorization = request.headers.get("authorization")
        if authorization is not None:
            scheme, separator, supplied = authorization.partition(" ")
            if not (
                separator
                and scheme.casefold() == "bearer"
                and supplied
                and hmac.compare_digest(supplied, self.legacy_bearer_token)
            ):
                raise AuthorizationDenied("authentication required")
            principal = self._legacy_principal()
            request.state.principal = principal
            return principal
        principal, session = self._authenticate_session(request.cookies.get(SESSION_COOKIE, ""))
        if request.method.upper() in _STATE_CHANGING_METHODS:
            cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
            header_csrf = request.headers.get(CSRF_HEADER, "")
            if (
                not cookie_csrf
                or not header_csrf
                or len(cookie_csrf) > 512
                or len(header_csrf) > 512
                or not hmac.compare_digest(cookie_csrf, header_csrf)
                or not hmac.compare_digest(
                    self._secret_hash(cookie_csrf),
                    str(session["csrfTokenHash"]),
                )
            ):
                raise CSRFValidationFailed("CSRF validation failed")
        request.state.principal = principal
        return principal

    def authenticate_websocket(
        self,
        *,
        authorization: str | None,
        session_token: str | None,
        legacy_query_token: str | None = None,
        csrf_cookie: str | None = None,
        csrf_token: str | None = None,
        require_csrf: bool = False,
    ) -> Principal:
        if authorization is not None:
            scheme, separator, supplied = authorization.partition(" ")
            if (
                separator
                and scheme.casefold() == "bearer"
                and supplied
                and hmac.compare_digest(supplied, self.legacy_bearer_token)
            ):
                return self._legacy_principal()
            raise AuthorizationDenied("authentication required")
        if legacy_query_token is not None and hmac.compare_digest(
            legacy_query_token,
            self.legacy_bearer_token,
        ):
            return self._legacy_principal()
        principal, session = self._authenticate_session(session_token or "")
        if require_csrf:
            cookie_value = csrf_cookie or ""
            supplied_value = csrf_token or ""
            if (
                not cookie_value
                or not supplied_value
                or len(cookie_value) > 512
                or len(supplied_value) > 512
                or not hmac.compare_digest(cookie_value, supplied_value)
                or not hmac.compare_digest(
                    self._secret_hash(cookie_value),
                    str(session["csrfTokenHash"]),
                )
            ):
                raise CSRFValidationFailed("CSRF validation failed")
        return principal


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth", None)
    if not isinstance(service, AuthService):
        raise HTTPException(status_code=503, detail="authentication service is unavailable")
    return service


def authenticated_principal(request: Request) -> Principal:
    try:
        return get_auth_service(request).authenticate_request(request)
    except CSRFValidationFailed as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except AuthorizationDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_scope(scope: Scope) -> Callable[[Request], Principal]:
    def dependency(request: Request) -> Principal:
        principal = authenticated_principal(request)
        try:
            get_auth_service(request).policy.require(principal, scope)
        except AuthorizationDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return principal

    dependency.__name__ = f"require_{scope.value.replace(':', '_')}"
    return dependency


def require_resource_access(
    request: Request,
    principal: Principal,
    scope: Scope,
    *,
    resource_type: str,
    resource_id: str,
) -> None:
    service = get_auth_service(request)
    binding = service.store.get_resource_owner(resource_type, resource_id)
    try:
        service.policy.require(
            principal,
            scope,
            owner_user_id=None if binding is None else str(binding["ownerUserId"]),
            site_id=None if binding is None else str(binding["siteId"]),
            resource_is_bound=binding is not None,
        )
    except AuthorizationDenied as exc:
        # Do not reveal cross-owner/cross-site resource existence.
        raise HTTPException(status_code=404, detail="resource not found") from exc


def create_auth_router(service: AuthService) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["authentication"])
    users_admin = require_scope(Scope.USERS_MANAGE)
    developer_diagnostics = require_scope(Scope.DIAGNOSTICS_READ)

    @router.post("/auth/register", status_code=status.HTTP_201_CREATED)
    def register(body: dict[str, Any]):
        try:
            return {"user": service.register(body)}
        except AuthorizationDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            code = 409 if "already registered" in str(exc) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @router.post("/auth/login")
    def login(body: dict[str, Any], request: Request, response: Response):
        try:
            principal, session_secret, csrf_secret = service.login(body, request)
        except AuthorizationDenied as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        max_age = int(service.session_ttl_seconds)
        response.set_cookie(
            SESSION_COOKIE,
            session_secret,
            max_age=max_age,
            path="/",
            secure=service.cookie_secure,
            httponly=True,
            samesite="lax",
        )
        response.set_cookie(
            CSRF_COOKIE,
            csrf_secret,
            max_age=max_age,
            path="/",
            secure=service.cookie_secure,
            httponly=False,
            samesite="lax",
        )
        return {"principal": principal.public(), "csrfToken": csrf_secret}

    @router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(
        response: Response,
        principal: Principal = Depends(authenticated_principal),
    ) -> Response:
        if principal.session_id is not None:
            service.store.revoke_session(principal.session_id)
        response.delete_cookie(SESSION_COOKIE, path="/", secure=service.cookie_secure, httponly=True, samesite="lax")
        response.delete_cookie(CSRF_COOKIE, path="/", secure=service.cookie_secure, httponly=False, samesite="lax")
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.get("/auth/me")
    def me(principal: Principal = Depends(authenticated_principal)):
        return {"principal": principal.public()}

    @router.get("/admin/users", dependencies=[Depends(users_admin)])
    def list_users(limit: int = 200):
        return service.store.list_users(limit=limit)

    @router.patch("/admin/users/{user_id}")
    def update_user(
        user_id: str,
        body: dict[str, Any],
        principal: Principal = Depends(users_admin),
    ):
        allowed = {"role", "status", "siteId"}
        unexpected = set(body) - allowed
        if unexpected:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported user fields: {', '.join(sorted(unexpected))}",
            )
        try:
            role = None if "role" not in body else Role(str(body["role"])).value
            user_status = body.get("status")
            if user_status is not None and user_status not in {"active", "disabled"}:
                raise ValueError("status must be active or disabled")
            site_id = None if "siteId" not in body else _clean_site_id(body["siteId"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if user_id == principal.user_id and (
            (role is not None and role != Role.ADMIN.value) or user_status == "disabled"
        ):
            raise HTTPException(status_code=409, detail="an administrator cannot remove their own access")
        updated = service.store.update_user_access(
            user_id,
            role=role,
            status=user_status,
            site_id=site_id,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="user not found")
        return updated

    @router.get("/admin/login-audits", dependencies=[Depends(users_admin)])
    def list_login_audits(limit: int = 200):
        return service.store.list_login_audits(limit=limit)

    @router.get(
        "/diagnostics/auth",
        dependencies=[Depends(developer_diagnostics)],
    )
    def auth_diagnostics():
        return {
            "policyEngine": "deterministic-casbin-equivalent",
            "schemaMigrations": service.store.schema_migrations(),
            "roles": {
                role.value: sorted(scope.value for scope in scopes)
                for role, scopes in ROLE_SCOPES.items()
            },
            "legacyBearerRole": Role.DEVELOPER.value,
        }

    return router


__all__ = [
    "AuthService",
    "AuthorizationDenied",
    "CSRFValidationFailed",
    "CSRF_COOKIE",
    "CSRF_HEADER",
    "DeterministicRBACPolicy",
    "Principal",
    "ROLE_SCOPES",
    "Role",
    "SESSION_COOKIE",
    "Scope",
    "ScryptPasswordHasher",
    "authenticated_principal",
    "create_auth_router",
    "get_auth_service",
    "require_resource_access",
    "require_scope",
]
