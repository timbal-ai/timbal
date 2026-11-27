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
    from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, CLIENT_TUNING_TTS_EXTRA

    for group in hub.VARIANTES:
        for option in group["opciones"]:
            hello = option.get("hello", {})
            assert set(hello) <= set(CLIENT_SETTABLE_VOICE_FIELDS), (group["id"], option["id"])
            assert set(hello.get("tts_extra", {})) <= set(CLIENT_TUNING_TTS_EXTRA), (group["id"], option["id"])


def test_the_gemini_voice_option_names_the_language_and_the_region(hub) -> None:
    """Without the tag Gemini picks the accent again in every request, and it changes inside one reply."""
    voices = next(group for group in hub.VARIANTES if group["id"] == "voz")
    gemini = next(option for option in voices["opciones"] if option["id"] == "gemini")
    assert gemini["hello"]["tts_extra"] == {"language_code": "es-ES"}


def _voice(voice_id: str, name: str, category: str = "professional", **labels: str) -> dict:
    return {"voice_id": voice_id, "name": name, "category": category, "labels": labels}


def test_account_voices_are_listed_by_language_with_the_account_s_own_first(hub) -> None:
    listed = hub.voces_elevenlabs(
        [
            _voice("en1", "Adam", gender="male", language="en", accent="american"),
            _voice("es2", "Zoe", gender="female", language="es", accent="mexican"),
            _voice("es1", "Marta   de  Madrid", gender="female", language="es", accent="peninsular"),
            _voice("ca1", "Núria", gender="female", language="ca"),
            _voice("own", "Mi clon", category="cloned"),
            _voice("ro1", "Liviu", gender="male", language="ro", accent="standard"),
            _voice("", "sin id"),
            {"voice_id": "raro", "name": "x" * 90, "labels": None},
        ]
    )
    assert [(v["id"], v["grupo"]) for v in listed] == [
        ("own", hub.GRUPO_PROPIAS),
        ("es1", "Español"),  # Spain before the other accents
        ("es2", "Español"),
        ("ca1", "Catalán"),
        ("en1", "Inglés"),
        ("ro1", hub.GRUPO_OTROS),
        ("raro", hub.GRUPO_OTROS),
    ]
    by_id = {v["id"]: v for v in listed}
    assert by_id["es1"] == {"id": "es1", "nombre": "Marta de Madrid", "detalle": "mujer, España", "grupo": "Español"}
    assert by_id["own"]["detalle"] == ""
    assert len(by_id["raro"]["nombre"]) <= 48 and by_id["raro"]["nombre"].endswith("…")


def test_gemini_voices_offered_are_the_ones_the_framework_takes(hub) -> None:
    from timbal.voice.gemini_tts import DEFAULT_VOICE, GEMINI_VOICES

    names = [name for name, _ in hub.VOCES_GEMINI]
    assert {name.lower() for name in names} == set(GEMINI_VOICES)
    assert len(names) == len(set(names))
    assert DEFAULT_VOICE in names
    assert all(detalle for _, detalle in hub.VOCES_GEMINI)


class _Page:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_account_voices_are_read_page_by_page_with_the_key_in_a_header(hub, monkeypatch) -> None:
    pages = [
        {"voices": [{"voice_id": "a", "name": "A"}], "has_more": True, "next_page_token": "p2"},
        {"voices": [{"voice_id": "b", "name": "B"}], "has_more": False, "next_page_token": None},
    ]
    asked = []

    def urlopen(request, **_):
        asked.append(request)
        return _Page(pages[len(asked) - 1])

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_key")
    monkeypatch.setattr(hub.urllib.request, "urlopen", urlopen)
    assert [v["voice_id"] for v in hub._pedir_voces_elevenlabs()] == ["a", "b"]
    assert [r.full_url for r in asked] == [
        hub.ELEVENLABS_VOCES_URL + "?page_size=100",
        hub.ELEVENLABS_VOCES_URL + "?page_size=100&next_page_token=p2",
    ]
    assert all(r.get_header("Xi-api-key") == "sk_test_key" for r in asked)


def test_a_voice_list_that_cannot_be_read_says_why_without_the_key(hub, monkeypatch) -> None:
    def refused(request, **_):
        raise hub.urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_key")
    monkeypatch.setattr(hub.urllib.request, "urlopen", refused)
    monkeypatch.setattr(hub, "_VOCES_LEIDAS", {"voces": None, "cuando": 0.0})
    catalog = hub.catalogo_de_voces()
    assert catalog["elevenlabs"]["voces"] == []
    assert "401" in catalog["elevenlabs"]["error"]
    assert "sk_test_key" not in json.dumps(catalog)
    # Gemini's list needs no account: it is still offered.
    assert len(catalog["gemini"]["voces"]) == len(hub.VOCES_GEMINI)

    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr(hub.playground, "_read_dotenv", lambda _path: {})
    assert "ELEVENLABS_API_KEY" in hub.catalogo_de_voces()["elevenlabs"]["error"]


def test_account_voices_are_asked_for_once_and_kept(hub, monkeypatch) -> None:
    calls = []

    def read():
        calls.append(1)
        return [_voice("es1", "Marta", gender="female", language="es", accent="spain")]

    monkeypatch.setattr(hub, "_VOCES_LEIDAS", {"voces": None, "cuando": 0.0})
    monkeypatch.setattr(hub, "_pedir_voces_elevenlabs", read)
    first, second = hub.catalogo_de_voces(), hub.catalogo_de_voces()
    assert len(calls) == 1
    assert first == second
    assert first["elevenlabs"] == {
        "voces": [{"id": "es1", "nombre": "Marta", "detalle": "mujer, España", "grupo": "Español"}],
        "error": "",
    }

    # Once the list is old it is read again; an account that does not answer then leaves the last one.
    def down():
        raise hub.SinVoces("no se pudo hablar con ElevenLabs.")

    hub._VOCES_LEIDAS["cuando"] -= hub._VOCES_VALEN_SECS + 1
    monkeypatch.setattr(hub, "_pedir_voces_elevenlabs", down)
    assert hub.catalogo_de_voces() == first


def test_the_page_gets_each_demo_s_voice_provider_and_none_of_its_wiring(hub) -> None:
    demo = {"id": "x", "name": "X", "import_spec": "examples/x.py::agent", "model_env": "X_MODEL", "probar": ["Hola"]}
    assert hub.demo_publica(demo) == {"id": "x", "name": "X", "probar": ["Hola"], "proveedor_voz": "elevenlabs"}
    assert hub.demo_publica({**demo, "proveedor_voz": "gemini"})["proveedor_voz"] == "gemini"


def test_the_voice_and_the_speed_the_page_sends_are_settings_a_caller_may_set() -> None:
    from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS

    assert {"voice", "tts_tempo", "tts_provider"} <= set(CLIENT_SETTABLE_VOICE_FIELDS)
    page = (HUB / "index.html").read_text(encoding="utf-8")
    assert "hello.voice = " in page
    assert "hello.tts_tempo = " in page
