"""The voice demo page (``examples/voice_demos``): its demo list and what each variant starts."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HUB = ROOT / "examples" / "voice_demos"


@pytest.fixture(scope="module")
def hub():
    spec = importlib.util.spec_from_file_location("voice_demos_hub", HUB / "hub.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_listed_demo_points_at_an_agent_in_the_repository(hub) -> None:
    listed = json.loads((HUB / "demos.json").read_text(encoding="utf-8"))["demos"]
    loaded = hub.cargar_demos(None)
    assert [d["id"] for d in loaded] == [d["id"] for d in sorted(listed, key=lambda d: d.get("orden", 100))]
    for demo in loaded:
        path, _, name = demo["import_spec"].partition("::")
        assert (ROOT / path).is_file(), demo["id"]
        assert name
        assert demo["probar"], demo["id"]


def test_a_demo_whose_agent_file_is_missing_is_left_out(hub, tmp_path: Path) -> None:
    local = tmp_path / "demos.local.json"
    local.write_text(
        json.dumps({"demos": [{"id": "gone", "name": "Gone", "import_spec": "examples/does_not_exist.py::agent"}]}),
        encoding="utf-8",
    )
    assert "gone" not in [d["id"] for d in hub.cargar_demos(local)]


def test_default_variant_starts_the_demo_as_it_is_configured(hub) -> None:
    demo = {"id": "d", "import_spec": "x.py::agent", "model_env": "D_MODEL", "tuning_env": "D_TUNING"}
    env, hello = hub.plan_de_arranque(demo, "", "")
    assert env == {}
    assert hello == {}


def test_model_and_naturalness_reach_the_demo_through_its_env(hub) -> None:
    demo = {"id": "d", "import_spec": "x.py::agent", "model_env": "D_MODEL", "tuning_env": "D_TUNING"}
    env, hello = hub.plan_de_arranque(demo, "gemini-2.5-flash", "ambos")
    assert env["D_MODEL"] == "google/gemini-2.5-flash"
    tuning = json.loads(env["D_TUNING"])
    assert tuning["tts_rhythm"] is True
    assert tuning["hesitation"]["rate"] == hub.HESITATION_RATE
    assert tuning["hesitation"]["openers"]
    assert hello == {}

    env, _ = hub.plan_de_arranque(demo, "", "ninguno")
    assert json.loads(env["D_TUNING"]) == {"hesitation": False, "tts_rhythm": False}


def test_a_demo_without_a_model_variable_gets_the_model_in_the_call_hello(hub) -> None:
    demo = {"id": "d", "import_spec": "x.py::agent"}
    env, hello = hub.plan_de_arranque(demo, "gemini-2.5-flash", "")
    assert env == {}
    assert hello == {"model": "google/gemini-2.5-flash"}


def test_unknown_variants_and_unsupported_tuning_are_refused(hub) -> None:
    demo = {"id": "d", "import_spec": "x.py::agent"}
    with pytest.raises(ValueError):
        hub.plan_de_arranque(demo, "no-such-model", "")
    with pytest.raises(ValueError):
        hub.plan_de_arranque(demo, "", "titubeos")


def test_call_variants_only_use_keys_a_browser_may_set(hub) -> None:
    from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS

    for group in hub.VARIANTES:
        for option in group["opciones"]:
            assert set(option.get("hello", {})) <= set(CLIENT_SETTABLE_VOICE_FIELDS), (group["id"], option["id"])
