"""
Client implementation for the ServiceTitan REST API.

This module defines the :class:`ServiceTitanClient` class which
authenticates against the ServiceTitan authorization server using
the OAuth2 client credentials grant and performs HTTP requests
against ServiceTitan API endpoints.  The client caches the
access token for the duration specified by ``expires_in`` in the
token response and automatically refreshes it when needed.

Usage
-----

.. code-block:: python

    from servicetitan_api_client import ServiceTitanClient

    # Create a client for the integration environment
    client = ServiceTitanClient(
        client_id="abc123",
        client_secret="shhsecret",
        app_key="myappkey",
        environment="integration"
    )

    # Make a GET request to list employees for a tenant
    employees = client.get("settings/v2/tenant/123456/employees", params={"page": 1})
    for employee in employees.get("data", []):
        print(employee["name"])

The client will automatically fetch a new access token when the
previous one expires and will include your app key in the
``ST-App-Key`` header for every request as required by ServiceTitan.

For details on obtaining your client credentials and app key, as
well as a list of available endpoints, see the ServiceTitan
developer portal.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from datetime import date as _date, time as _time, datetime as _datetime

from .exceptions import ServiceTitanAuthError, ServiceTitanAPIError


class ServiceTitanClient:
    """A simple client for the ServiceTitan REST API.

    Parameters
    ----------
    client_id : str
        Your ServiceTitan OAuth client identifier.
    client_secret : str
        Your ServiceTitan OAuth client secret.
    app_key : str
        The application key generated when you create an app in
        ServiceTitan.  This value is sent in the ``ST-App-Key``
        header on every request.
    tenant : str, optional
        A default tenant identifier to prefix on relative paths.
        When provided, the tenant will be automatically inserted at
        the start of the path when the request URL begins with
        ``tenant/``.
    environment : str, optional
        Which environment to target.  Use ``"integration"`` to
        connect to the integration (sandbox) servers and
        ``"production"`` to connect to the live servers.  The
        default is ``"integration"``.
    auth_url : str, optional
        Override the token endpoint URL.  When provided, this
        parameter overrides the URL derived from the environment.
    base_url : str, optional
        Override the API base URL.  When provided, this parameter
        overrides the URL derived from the environment.

    Notes
    -----
    The client caches the access token and its expiry time.  If
    multiple requests are made in quick succession, the cached
    token will be reused until it is about to expire (within
    60 seconds).  This behaviour both improves performance and
    avoids ServiceTitan's throttling of token requests.
    """

    # Default endpoints for each environment
    _DEFAULT_AUTH_URLS = {
        "integration": "https://auth-integration.servicetitan.io/connect/token",
        "production": "https://auth.servicetitan.io/connect/token",
    }
    _DEFAULT_BASE_URLS = {
        "integration": "https://api-integration.servicetitan.io",
        "production": "https://api.servicetitan.io",
    }

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        app_key: str,
        app_guid: Optional[str] = None,
        tenant: Optional[str] = None,
        environment: str = "integration",
        auth_url: Optional[str] = None,
        base_url: Optional[str] = None,
        local_timezone: str = "Australia/Sydney",
    ) -> None:
        if not client_id:
            raise ValueError("client_id must be provided")
        if not client_secret:
            raise ValueError("client_secret must be provided")
        if not app_key:
            raise ValueError("app_key must be provided")

        environment = environment.lower()
        if environment not in {"integration", "production"}:
            raise ValueError(
                "environment must be either 'integration' or 'production', got %r"
                % environment
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.app_key = app_key
        self.app_guid = app_guid
        self.tenant = tenant
        self.environment = environment
        self.auth_url = auth_url or self._DEFAULT_AUTH_URLS[environment]
        self.base_url = base_url or self._DEFAULT_BASE_URLS[environment]

        # Configure the local timezone for date conversions.  This
        # string will be resolved via ZoneInfo when needed by
        # `_get_user_zone`.  Changing this value after initialisation
        # will affect future calls to the date helper methods.
        self.local_timezone = local_timezone

        # Internal token cache
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0  # epoch seconds when token expires

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _refresh_access_token(self) -> None:
        """Retrieve a new access token from the ServiceTitan auth server.

        This method posts to the OAuth token endpoint with the
        ``client_id`` and ``client_secret`` using the client
        credentials grant.  On success, it stores the ``access_token``
        and calculates the absolute expiry time from the ``expires_in``
        value.  If the server responds with a non‑success status,
        :class:`ServiceTitanAuthError` is raised.
        """
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = requests.post(self.auth_url, data=payload, headers=headers)
        except Exception as exc:
            raise ServiceTitanAuthError(f"Failed to connect to auth server: {exc}") from exc

        if not response.ok:
            # Attempt to extract JSON error details if available
            err_text = response.text
            try:
                err_json = response.json()
                err_text = str(err_json)
            except Exception:
                pass
            raise ServiceTitanAuthError(
                f"Authentication failed with status {response.status_code}: {err_text}"
            )

        token_info: Dict[str, Any] = response.json()
        access_token = token_info.get("access_token")
        if not access_token:
            raise ServiceTitanAuthError(
                "Authentication response did not contain an access_token"
            )
        expires_in = token_info.get("expires_in")
        # Default to 900 seconds if not specified, as per docs
        if not isinstance(expires_in, (int, float)):
            expires_in = 900
        # Current time plus expiry minus a small safety margin (10 seconds)
        self._access_token = access_token
        self._token_expiry = time.time() + float(expires_in)

    def _get_access_token(self) -> str:
        """Return a valid access token, refreshing it if expired.

        This helper checks whether the cached token is still valid
        (with a 60‑second buffer).  If it has expired or doesn't exist,
        it triggers a refresh by calling :meth:`_refresh_access_token`.
        """
        now = time.time()
        # Refresh token if expired or about to expire within 60 seconds
        if not self._access_token or now >= (self._token_expiry - 60):
            self._refresh_access_token()
        assert self._access_token is not None
        return self._access_token

    # ------------------------------------------------------------------
    # HTTP request helpers
    # ------------------------------------------------------------------
    def _prepare_url(self, path: str) -> str:
        """Build the full request URL from a relative or absolute path.

        If the path is an absolute URL (starts with "http"), it is
        returned as‑is.  Otherwise, it is joined to the client's
        ``base_url``.  If ``tenant`` is configured and the path
        begins with ``tenant/``, the tenant ID is interpolated into
        the URL in place of the placeholder ``{tenant}``.
        """
        if path.startswith("http://") or path.startswith("https://"):
            return path
        # Remove any leading slashes for consistency
        clean_path = path.lstrip("/")
        # Insert tenant into path if necessary
        if self.tenant and clean_path.startswith("tenant/"):
            clean_path = clean_path.replace("tenant/", f"tenant/{self.tenant}/", 1)
        return f"{self.base_url.rstrip('/')}/{clean_path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform an HTTP request against the ServiceTitan API.

        This method handles inserting the access token and app key
        headers, building the full URL, and raising an exception if
        the response indicates an error.

        Parameters
        ----------
        method : str
            The HTTP verb, such as ``"GET"``, ``"POST"``, ``"PUT"`` or ``"DELETE"``.
        path : str
            The API endpoint path relative to the base URL.  If an absolute
            URL is supplied, it will be used as‑is.  Paths beginning
            with ``tenant/`` will automatically include the default
            tenant ID if one is configured.
        params : dict, optional
            Query parameters to include in the request.
        json : object, optional
            A JSON‑serialisable body to include in the request for POST
            and PUT requests.
        headers : dict, optional
            Additional HTTP headers to merge with the defaults.
        timeout : float, optional
            Timeout in seconds for the underlying HTTP request.

        Returns
        -------
        Any
            When the server indicates JSON content, this method returns
            the parsed Python object.  For image or other binary
            content it returns the raw byte sequence.  Otherwise it
            returns the decoded text response.

        Raises
        ------
        ServiceTitanAPIError
            If the HTTP response status is not a success code (2xx).
        ServiceTitanAuthError
            If token refresh fails.
        """
        url = self._prepare_url(path)
        token = self._get_access_token()
        # Build headers
        req_headers = {
            "Authorization": f"Bearer {token}",
            "ST-App-Key": self.app_key,
        }
        if headers:
            # Avoid accidental overwrite of critical headers
            for key, value in headers.items():
                # Normalise header keys to avoid case sensitivity issues
                if key.lower() in {"authorization", "st-app-key"}:
                    continue
                req_headers[key] = value
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json,
                headers=req_headers,
                timeout=timeout,
            )
        except Exception as exc:
            raise ServiceTitanAPIError(f"Failed to connect to {url}: {exc}") from exc

        if response.status_code >= 400:
            # Attempt to return JSON error details
            err_text = response.text
            try:
                err_json = response.json()
                err_text = str(err_json)
            except Exception:
                pass
            raise ServiceTitanAPIError(
                f"{response.status_code} Error for {url}: {err_text}"
            )

        # Try to parse JSON when the response advertises JSON content.  If
        # JSON decoding fails or the content type indicates something
        # else (e.g. an image), choose an appropriate representation.
        content_type = response.headers.get("Content-Type", "").lower()
        # If the server says it's JSON, attempt to decode
        if content_type.startswith("application/json"):
            try:
                return response.json()
            except ValueError:
                # Fall back to text if JSON parsing fails
                return response.text
        # If it's an image or other binary data, return bytes
        if content_type.startswith("image/") or content_type.startswith("application/octet-stream"):
            return response.content
        # Otherwise return the decoded text
        return response.text

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform a GET request.

        See :meth:`_request` for full parameter documentation.
        """
        return self._request("GET", path, params=params, headers=headers, timeout=timeout)
    
    def get_all(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Performs GET requests until no more pages
        """
        output = []
        page = 1
        if params:
            params['page'] = page
        else:
            params = {'page': page}

        while True:
            params['page'] = page
            try:
                resp = self.get(path, params=params, headers=headers, timeout=timeout)
            except Exception:
                break        
            if not isinstance(resp, dict):
                break
            data = resp.get("data") or []
            output.extend(data)
            has_more = resp.get("hasMore")
            if has_more:
                page += 1
                continue
            break
        return output
    
    def get_all_id_filter(
        self,
        path: str,
        ids: list,
        *,
        id_filter_name: str = 'ids',
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not params:
            params = {}
        id_len = len(ids)
        if id_len < 50:
            params[id_filter_name] = ','.join(ids)
            resp = self.get(path, params=params, headers=headers, timeout=timeout)
            data = resp.get("data") or []
            return data
        elif id_len % 50 == 0:
            data = []
            for i in range(id_len // 50):
                tmp_ids = ids[50*i:50*(i+1)]
                params['ids'] = ','.join(tmp_ids)
                resp = self.get(path, params=params, headers=headers, timeout=timeout)
                data.extend(resp.get("data"))
            return data
        else:
            data = []
            for i in range((id_len // 50) + 1):
                tmp_ids = ids[50*i:50*(i+1)]
                params['ids'] = ','.join(tmp_ids)
                resp = self.get(path, params=params, headers=headers, timeout=timeout)
                data.extend(resp.get("data"))
            return data

    def post(
        self,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform a POST request.

        See :meth:`_request` for full parameter documentation.
        """
        return self._request(
            "POST",
            path,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

    def patch(
        self,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform a PATCH request.

        See :meth:`_request` for full parameter documentation.
        """
        return self._request(
            "PATCH",
            path,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

    def put(
        self,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform a PUT request.

        See :meth:`_request` for full parameter documentation.
        """
        return self._request(
            "PUT",
            path,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

    def delete(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Perform a DELETE request.

        See :meth:`_request` for full parameter documentation.
        """
        return self._request(
            "DELETE",
            path,
            params=params,
            headers=headers,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # URL construction helpers
    # ------------------------------------------------------------------
    def build_url(
        self,
        folder: str,
        endpoint: str,
        *,
        version: Any = 2,
        resource_id: Optional[Any] = None,
        modifier: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> str:
        """Construct a full API URL from logical components.

        This helper builds a path of the form::

            <base_url>/<folder>/v<version>/tenant/<tenant>/<endpoint>[/<id>[/<modifier>]]

        It can be used to generate resource URLs consistently without
        remembering the exact URL format.  If a ``tenant`` is not
        provided, the client's default tenant (if any) will be used.
        When neither the argument nor the default is available, the
        literal string ``"{tenant}"`` will be inserted.

        Parameters
        ----------
        folder : str
            The top‑level API folder (e.g. ``"jpm"`` for the Job
            Planning module).
        endpoint : str
            The resource endpoint name (e.g. ``"jobs"``).
        version : int or str, optional
            The major version number of the API (without the leading
            ``"v"``).  Defaults to ``2``.  The letter ``"v"`` is
            prepended automatically.
        resource_id : str or int, optional
            Identifier of the resource.  When provided, this value is
            appended after the endpoint.
        modifier : str, optional
            Additional path segment used to address a sub‑resource (e.g.
            ``"notes"`` to access job notes).
        tenant : str, optional
            Explicit tenant ID to use.  If not supplied, the client's
            ``tenant`` attribute is used; if that is also unset, the
            placeholder ``"{tenant}"`` is used instead.

        Returns
        -------
        str
            A fully qualified URL pointing to the requested resource.

        Examples
        --------

        >>> client.build_url(
        ...     folder="jpm", endpoint="jobs", version=2,
        ...     resource_id=123, modifier="notes"
        ... )
        'https://api-integration.servicetitan.io/jpm/v2/tenant/123456/jobs/123/notes'

        """
        if not folder:
            raise ValueError("folder must not be empty")
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        # Determine tenant ID
        tenant_id = tenant or self.tenant or "{tenant}"
        # Normalise version string
        version_str = str(version)
        if version_str.lower().startswith("v"):
            version_segment = version_str.lower()
        else:
            version_segment = f"v{version_str}"
        # Build path segments
        segments = [folder.strip("/"), version_segment, "tenant", str(tenant_id), endpoint.strip("/")]
        if resource_id is not None:
            segments.append(str(resource_id))
        if modifier:
            segments.append(modifier.strip("/"))
        # Join segments into a path and delegate to _prepare_url
        path = "/".join(segments)
        return self._prepare_url(path)
    
    # ------------------------------------------------------------------
    # Date and time helpers
    # ------------------------------------------------------------------
    def _get_user_zone(self) -> "ZoneInfo":
        """Return the configured local time zone as a :class:`~zoneinfo.ZoneInfo`.

        ServiceTitan expects all timestamps in UTC; however, your
        application likely works in a local time zone (defaulting to
        Australia/Sydney).  This helper resolves the local time zone
        name provided at construction into a ``ZoneInfo`` instance.
        The resulting zone object is cached so that subsequent
        conversions are inexpensive.

        If the standard ``zoneinfo`` module is not available (on
        Python 3.8 or earlier), it attempts to import
        ``backports.zoneinfo``.  If both imports fail, an
        ``ImportError`` is raised.
        """
        try:
            from zoneinfo import ZoneInfo  # type: ignore
        except ImportError:
            try:
                from backports.zoneinfo import ZoneInfo  # type: ignore
            except ImportError:
                raise ImportError(
                    "Timezone conversion requires Python 3.9+ or the backports.zoneinfo package"
                )
        # Cache the zoneinfo object to avoid recreating it.  Use
        # the configured local timezone string from ``self.local_timezone``.
        if not hasattr(self, "_user_zone"):
            setattr(self, "_user_zone", ZoneInfo(self.local_timezone))
        return getattr(self, "_user_zone")

    def to_utc(self, dt: "datetime") -> "datetime":
        """Convert a naive or timezone‑aware datetime to UTC.

        If ``dt`` is naive (has no ``tzinfo``), it is assumed to be in
        the client's configured local time zone.  The returned
        datetime will be timezone‑aware in UTC.

        Parameters
        ----------
        dt : datetime
            The datetime to convert.  Naive datetimes are assumed to
            represent local time in Australia/Sydney.

        Returns
        -------
        datetime
            The same moment in time represented in UTC with timezone
            information attached.
        """
        from datetime import timezone

        if dt.tzinfo is None:
            # Assume local zone
            dt = dt.replace(tzinfo=self._get_user_zone())
        return dt.astimezone(timezone.utc)

    def to_utc_string(self, dt: "datetime", *, fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
        """Convert a datetime to an ISO‑like string in UTC.

        This is a convenience wrapper around :meth:`to_utc` that
        formats the result as a string using ``datetime.strftime``.
        The default format produces a string such as ``"2025-11-07T15:30:00Z"``.

        Parameters
        ----------
        dt : datetime
            The datetime to convert and format.
        fmt : str, optional
            A strftime format string.  Defaults to ISO 8601 without
            fractional seconds and with a trailing ``Z`` to denote UTC.

        Returns
        -------
        str
            The formatted UTC timestamp.
        """
        dt_utc = self.to_utc(dt)
        return dt_utc.strftime(fmt)

    def from_utc(self, dt: "datetime" | str) -> "datetime":
        """Convert a UTC datetime or ISO string to local time.

        This function converts a timestamp in UTC to the client's local
        time zone.  If a string is provided, it is parsed using
        ``datetime.fromisoformat`` which accepts many ISO 8601
        variants.  Strings that end with ``'Z'`` are treated as UTC.

        Parameters
        ----------
        dt : datetime or str
            A timezone‑aware datetime in UTC or an ISO 8601 string.

        Returns
        -------
        datetime
            A timezone‑aware datetime in the configured local time zone.
        """
        from datetime import datetime, timezone

        if isinstance(dt, str):
            # Replace trailing 'Z' with '+00:00' for ISO parsing
            if dt.endswith("Z"):
                iso_str = dt[:-1] + "+00:00"
            else:
                iso_str = dt
            # Use fromisoformat which returns a naive datetime if no tz
            parsed = datetime.fromisoformat(iso_str)
        else:
            parsed = dt
        # Attach UTC tzinfo if naive
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        # Convert to local zone
        return parsed.astimezone(self._get_user_zone())

    def from_utc_string(self, s: str) -> "datetime":
        """Parse an ISO string in UTC and convert it to local time.

        This is a thin wrapper around :meth:`from_utc` that accepts
        only a string and returns a timezone‑aware datetime in
        Australia/Sydney.

        Parameters
        ----------
        s : str
            An ISO 8601 timestamp in UTC.  A trailing ``Z`` denotes UTC.

        Returns
        -------
        datetime
            A timezone‑aware datetime in the configured local time zone.
        """
        return self.from_utc(s)

    def format_local(self, dt: "datetime", *, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """Format a datetime for display in the local time zone.

        If the provided datetime is naive, it is assumed to be in
        the client's configured local time zone.  Otherwise, it is
        converted from its existing timezone.  The default format
        produces strings such as ``"2025-11-07 02:30:00 AEDT"``.  You can supply a custom
        strftime format string via ``fmt``.

        Parameters
        ----------
        dt : datetime
            The datetime to format.  Naive datetimes are assumed to
            represent local time.
        fmt : str, optional
            A strftime format string.  Defaults to a human‑readable
            local representation including the timezone abbreviation.

        Returns
        -------
        str
            The formatted local time string.
        """
        # Ensure dt is timezone aware in local zone
        local_dt: "datetime"
        if dt.tzinfo is None:
            local_dt = dt.replace(tzinfo=self._get_user_zone())
        else:
            local_dt = dt.astimezone(self._get_user_zone())
        return local_dt.strftime(fmt)
    
    def st_date_to_local(self, st_date_str: str, *, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        return self.format_local(self.from_utc_string(st_date_str), fmt=fmt)

    def local_start_of_day(self, d: _date) -> _datetime:
        """
        Return a timezone‑aware datetime representing midnight on date `d`
        in the client’s local time zone.
        """
        tz = self._get_user_zone()
        return _datetime.combine(d, _time.min, tzinfo=tz)

    def local_end_of_day(self, d: _date) -> _datetime:
        """
        Return a timezone‑aware datetime representing 23:59:59 on date `d`
        in the client’s local time zone.
        """
        tz = self._get_user_zone()
        return _datetime.combine(d, _time(23, 59, 59), tzinfo=tz)

    def start_of_day_utc(self, d: _date) -> _datetime:
        """
        Convert midnight of date `d` in the local timezone to a UTC datetime.
        """
        return self.to_utc(self.local_start_of_day(d))

    def end_of_day_utc(self, d: _date) -> _datetime:
        """
        Convert 23:59:59 of date `d` in the local timezone to a UTC datetime.
        """
        return self.to_utc(self.local_end_of_day(d))

    def start_of_day_utc_string(self, d: _date) -> str:
        """
        Convert midnight of date `d` in the local timezone to an ISO 8601
        UTC string (ending in 'Z').
        """
        dt = self.start_of_day_utc(d)
        return dt.isoformat().replace("+00:00", "Z")

    def end_of_day_utc_string(self, d: _date) -> str:
        """
        Convert 23:59:59 of date `d` in the local timezone to an ISO 8601
        UTC string (ending in 'Z').
        """
        dt = self.end_of_day_utc(d)
        return dt.isoformat().replace("+00:00", "Z")