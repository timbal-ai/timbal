"""NetSuite OAuth TBA signature tests."""

from timbal.tools.netsuite import _netsuite_auth_header


def test_auth_signature_includes_query_params() -> None:
    url = "https://6897035.suitetalk.api.netsuite.com/services/rest/query/v1/suiteql"
    creds = {
        "account_id": "6897035",
        "consumer_key": "ck",
        "consumer_secret": "cs",
        "token_id": "tid",
        "token_secret": "ts",
    }

    without_params = _netsuite_auth_header("POST", url, query_params=None, **creds)
    with_params = _netsuite_auth_header("POST", url, query_params={"limit": 1, "offset": 0}, **creds)

    assert without_params != with_params
    assert "limit=" not in without_params
    assert "limit=" not in with_params
    assert "offset=" not in with_params


def test_auth_header_uses_oauth_params_only() -> None:
    header = _netsuite_auth_header(
        "GET",
        "https://6897035.suitetalk.api.netsuite.com/services/rest/record/v1/account",
        account_id="6897035",
        consumer_key="ck",
        consumer_secret="cs",
        token_id="tid",
        token_secret="ts",
        query_params={"limit": 5, "offset": 0},
    )
    assert header.startswith('OAuth realm="6897035"')
    assert "oauth_consumer_key=" in header
    assert "limit=" not in header
    assert "offset=" not in header
