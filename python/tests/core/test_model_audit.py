"""Live audits must distinguish retirement from account and deployment failures."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/audit_models.py"
    spec = importlib.util.spec_from_file_location("audit_models", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status_code", [401, 403, 408, 429, 500, 503])
def test_account_and_transient_errors_are_unverified(audit, status_code):
    assert audit._classify_error(
        "request failed", status_code, requires_activation=False, dedicated_only=False,
        catalog=None, api_name="test",
    ) == audit.Status.SKIPPED


@pytest.mark.parametrize("status_code,dedicated,expected", [
    (404, False, "dead"),
    (404, True, "dedicated_only"),
    (410, False, "dead"),
    (410, True, "dead"),
])
def test_missing_serverless_deployment_is_not_a_retired_model(audit, status_code, dedicated, expected):
    assert audit._classify_error(
        "Model not found, inaccessible, and/or not deployed", status_code,
        requires_activation=False, dedicated_only=dedicated, catalog=None, api_name="test",
    ).value == expected


@pytest.mark.parametrize("dedicated,expected", [(True, "dedicated_only"), (False, "skipped")])
def test_stopped_endpoint_is_not_dead(audit, dedicated, expected):
    assert audit._classify_error(
        '{"code":"endpoint_not_ready"}', 400, requires_activation=False,
        dedicated_only=dedicated, catalog=None, api_name="test",
    ).value == expected


def test_activation_required_still_classified_before_access_errors(audit):
    assert audit._classify_error(
        "ModelNotOpen: activate this model", 403, requires_activation=True,
        dedicated_only=False, catalog=None, api_name="test",
    ) == audit.Status.ACTIVATION_REQUIRED


def test_summary_does_not_count_unverified_models_as_available(audit, capsys):
    audit._print_summary([
        audit.ModelResult(str(i), "test", str(i), status)
        for i, status in enumerate([audit.Status.OK, audit.Status.SKIPPED, audit.Status.NO_KEY])
    ])
    output = capsys.readouterr().out
    assert "3 models" in output
    assert "1 OK" in output
    assert "1 SKIPPED, 1 NO_KEY" in output
    assert "unverified, not confirmed available" in output
