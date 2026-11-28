"""Demo de retos: dictado, identificación, herramienta que falla, cambios con confirmación y config de voz."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("voice_retos_demo", ROOT / "examples" / "voice_retos_demo.py")
assert _SPEC and _SPEC.loader
demo = importlib.util.module_from_spec(_SPEC)
sys.modules["voice_retos_demo"] = demo
_SPEC.loader.exec_module(demo)


def _use(session_id: str):
    from timbal.voice.session import _current_voice_session

    class _Session:
        pass

    holder = _Session()
    holder.session_id = session_id
    return _current_voice_session.set(holder)


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    monkeypatch.setattr(demo, "LENTA_SECS", 0.0)
    monkeypatch.setattr(demo, "RAPIDA_SECS", 0.0)
    demo.reset_demo_store()
    token = _use("llamada-1")
    yield
    from timbal.voice.session import _current_voice_session

    _current_voice_session.reset(token)
    demo.reset_demo_store()


@pytest.mark.parametrize(
    ("dictado", "esperado"),
    [
        ("uno dos tres cuatro cinco seis siete ocho zeta", "12345678Z"),
        ("mi DNI es 1 2 3 4 5 6 7 8 Z", "12345678Z"),
        ("12345678z", "12345678Z"),
        ("cuarenta y siete tres ocho dos nueve uno cinco uve", "47382915V"),
        ("doble cero siete", "007"),
        ("equis uno dos tres cuatro cinco seis siete ele", "X1234567L"),
        ("cuatro siete tres ocho dos nueve uno cinco uve doble", "47382915W"),
    ],
)
def test_dictado_a_codigo(dictado: str, esperado: str) -> None:
    assert demo.dictado_a_codigo(dictado) == esperado


def test_solo_cifras_no_convierte_palabras_en_letras() -> None:
    assert demo.dictado_a_codigo("el cinco de diciembre") != "5"  # "de" suena a letra D
    assert demo.dictado_a_codigo("el cinco de diciembre", letras=False) == "5"
    assert demo.dictado_a_codigo("veinte", letras=False) == "20"


async def test_identifica_con_cifras_en_palabras() -> None:
    out = await demo.identificar_cliente("uno dos tres cuatro cinco seis siete ocho zeta")
    assert "Marta Soler" in out
    assert "4,6 kilovatios" in out


async def test_letra_equivocada_no_revela_la_buena() -> None:
    out = await demo.identificar_cliente("uno dos tres cuatro cinco seis siete ocho te")
    assert "no cuadra" in out
    assert "Z" not in out  # la letra correcta no sale en el mensaje
    assert await demo.ultima_factura() == demo._SIN_IDENTIFICAR


async def test_pide_la_letra_y_rechaza_dictados_incompletos() -> None:
    assert "letra" in await demo.identificar_cliente("uno dos tres cuatro cinco seis siete ocho")
    assert "No he sacado" in await demo.identificar_cliente("uno dos tres")


async def test_sin_identificar_ninguna_herramienta_da_datos() -> None:
    for tool in (demo.ultima_factura, demo.historico_consumo, demo.estado_suministro):
        assert await tool() == demo._SIN_IDENTIFICAR
    assert await demo.preparar_cambio("dia_cargo", "veinte") == demo._SIN_IDENTIFICAR


async def test_la_factura_de_marta_es_casi_el_doble_de_la_media() -> None:
    await demo.identificar_cliente("12345678Z")
    out = await demo.ultima_factura()
    assert "115,24 euros" in out
    assert "64,80 euros" in out
    consumo = await demo.historico_consumo()
    assert "octubre 412" in consumo and "mayo 180" in consumo


def test_el_total_sale_de_las_partidas() -> None:
    for c in demo._SEED_CLIENTES.values():
        d = demo.desglose(c["factura"])
        parts = d["potencia"] + d["energia"] + d["impuesto"] + d["alquiler"] + d["iva"]
        assert d["total"] == pytest.approx(parts, abs=0.011)


async def test_el_suministro_de_lola_falla_una_sola_vez() -> None:
    await demo.identificar_cliente("28954731P")
    assert "Error" in await demo.estado_suministro()
    assert "corte programado" in await demo.estado_suministro()
    assert "corte programado" in await demo.estado_suministro()


async def test_cada_llamada_tiene_su_propio_estado() -> None:
    await demo.identificar_cliente("28954731P")
    assert "Error" in await demo.estado_suministro()
    from timbal.voice.session import _current_voice_session

    token = _use("llamada-2")
    try:
        await demo.identificar_cliente("28954731P")
        assert "Error" in await demo.estado_suministro()  # su propio primer fallo
    finally:
        _current_voice_session.reset(token)


async def test_un_cambio_no_se_guarda_hasta_confirmar() -> None:
    await demo.identificar_cliente("12345678Z")
    out = await demo.preparar_cambio("dia_cargo", "quince")
    assert "sin guardar" in out and "día 15" in out
    assert demo._cliente()["dia_cargo"] == 5
    assert "Guardado" in await demo.confirmar_cambio()
    assert demo._cliente()["dia_cargo"] == 15


async def test_corregirse_sustituye_al_cambio_anterior() -> None:
    await demo.identificar_cliente("12345678Z")
    await demo.preparar_cambio("dia_cargo", "quince")
    await demo.preparar_cambio("dia_cargo", "veinte")
    assert "día 20" in await demo.confirmar_cambio()
    assert demo._cliente()["dia_cargo"] == 20
    assert "No hay nada preparado" in await demo.confirmar_cambio()  # no se aplica dos veces


@pytest.mark.parametrize("tipo", ["dia_pago", "Día de cargo", "fecha_cargo"])
async def test_el_tipo_de_cambio_tolera_sinonimos(tipo: str) -> None:
    await demo.identificar_cliente("12345678Z")
    assert "día 15" in await demo.preparar_cambio(tipo, "quince")


async def test_cancelar_descarta() -> None:
    await demo.identificar_cliente("12345678Z")
    await demo.preparar_cambio("dia_cargo", "veinte")
    assert "Descartado" in await demo.cancelar_cambio()
    assert "No hay nada preparado" in await demo.confirmar_cambio()
    assert demo._cliente()["dia_cargo"] == 5


@pytest.mark.parametrize("dia", ["treinta", "cero", "cuarenta"])
async def test_dia_de_cargo_fuera_de_rango(dia: str) -> None:
    await demo.identificar_cliente("12345678Z")
    assert "entre el uno y el veintiocho" in await demo.preparar_cambio("dia_cargo", dia)


async def test_potencia_inexistente_ofrece_las_cercanas() -> None:
    await demo.identificar_cliente("12345678Z")
    out = await demo.preparar_cambio("potencia", "5")
    assert "no es una potencia" in out and "4,6" in out and "5,75" in out
    assert "sin guardar" in await demo.preparar_cambio("potencia", "5,75")
    assert "5,75 kilovatios" in await demo.confirmar_cambio()
    assert demo._cliente()["potencia"] == 5.75


async def test_reclamacion_lleva_referencia() -> None:
    await demo.identificar_cliente("12345678Z")
    await demo.preparar_cambio("reclamacion", "factura de octubre demasiado alta")
    assert "referencia R 1" in await demo.confirmar_cambio()


async def test_pasar_a_persona_termina_la_llamada_sin_esperar_al_modelo() -> None:
    """El modelo colgaba un turno después con end_call, que se niega si el cliente solo ha dicho «vale»."""
    from timbal.voice.session import _current_voice_session

    colgados: list[bool] = []

    class _Session:
        session_id = "llamada-3"

        def end_call(self) -> None:
            colgados.append(True)

    token = _current_voice_session.set(_Session())
    try:
        dicho = await demo.pasar_a_persona("contador a cero")
    finally:
        _current_voice_session.reset(token)
    assert colgados == [True]
    assert "no llames a end_call" in dicho


async def test_pasar_a_persona_sin_llamada_no_falla() -> None:
    assert "Transferencia simulada" in await demo.pasar_a_persona("contador a cero")


def test_la_config_de_voz_es_valida_y_pone_a_pablo() -> None:
    cfg = demo.retos_voice_config()
    assert cfg["voice"] == demo._PABLO
    assert cfg["tts_model"] == "eleven_v3_conversational"
    assert cfg["hesitation"]["rate"] > 0
    assert cfg["entity_patience"] is True
    assert "Luzia" in cfg["greeting"]["text"]


def test_el_tuning_de_entorno_sobrescribe(monkeypatch) -> None:
    monkeypatch.setenv("TIMBAL_VOICE_DEMO_TUNING", '{"tts_tempo": 1.1}')
    assert demo.retos_voice_config()["tts_tempo"] == 1.1


def test_el_agente_tiene_todas_las_herramientas() -> None:
    names = {getattr(t, "__name__", None) or getattr(t, "name", "") for t in demo.agent.tools}
    assert {
        "identificar_cliente", "ultima_factura", "historico_consumo", "estado_suministro",
        "preparar_cambio", "confirmar_cambio", "cancelar_cambio", "pasar_a_persona", "end_call",
    } <= names  # fmt: skip


def test_gemini_flash_no_piensa_en_una_llamada() -> None:
    # Con razonamiento, una petición de 0,4 s tardaba 5-16 s en dar el primer token: línea muerta en la llamada.
    assert demo._MODEL_PARAMS == (
        {"reasoning_effort": "none"} if demo._MODEL.startswith("google/gemini-2.5-flash") else {}
    )
    assert demo.agent.model_params == demo._MODEL_PARAMS
