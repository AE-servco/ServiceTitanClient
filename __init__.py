"""
Python client for interacting with the ServiceTitan REST API.

This package provides a simple `ServiceTitanClient` class that handles
OAuth2 client‑credentials authentication against the ServiceTitan
authorization server and makes authenticated requests to API endpoints.

The client caches access tokens for their entire lifetime and
automatically requests a new token when the current one expires.

Examples
--------

```python
from servicetitan_api_client import ServiceTitanClient

# Initialise the client with your app credentials
client = ServiceTitanClient(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    app_key="YOUR_APP_KEY",
    environment="integration",  # or "production"
)

# Perform a GET request
employees = client.get(
    "settings/v2/tenant/123456/employees",
    params={"page": 1, "pageSize": 50},
)

# employees now contains the parsed JSON response
```

See Also
--------
The ServiceTitan developer documentation provides details on creating
applications, retrieving your client ID, client secret and app key,
and lists all available API endpoints.

References
----------
ServiceTitan's "Make Your First API Call" guide describes how to
obtain an access token using the client credentials grant by
POSTing to the token endpoint with `client_id`, `client_secret`
and `grant_type=client_credentials`. The returned JSON includes an
`access_token` and `expires_in` value which should be cached and
included in the `Authorization` header for subsequent API calls【861456388967273†L72-L119】.
The ServiceTitan API also requires your application key to be
included in requests via the `ST-App-Key` header【861456388967273†L122-L140】.
"""

from .client import ServiceTitanClient
from .exceptions import ServiceTitanAuthError, ServiceTitanAPIError

__all__ = [
    "ServiceTitanClient",
    "ServiceTitanAuthError",
    "ServiceTitanAPIError",
]