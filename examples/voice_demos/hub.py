"""Página sencilla para probar las demos de voz: eliges una demo, pulsas «Llamar» y hablas.

    uv run python examples/voice_demos/hub.py [--port 7777] [--no-open]

Sirve ``index.html`` en ``/``, la lista de demos en ``/demos`` y arranca por debajo un ``timbal.server`` con la demo
elegida (el mismo ``ChildServer`` del playground). El playground completo sigue disponible en ``/voice``.

Las demos salen de ``demos.json`` y, si existe, de ``demos.local.json`` (no se sube al repositorio: es donde van las
demos con marca de cliente). ``--local otra/ruta.json`` o ``VOICE_DEMOS_LOCAL`` cambian ese segundo fichero; un
valor vacío lo desactiva. ``GET /estado`` sirve ``estado.local.html`` si existe.

Qué cambia cada variante:

- Proveedor de voz, Voz, Velocidad y Escucha viajan en el primer mensaje de la llamada (el servidor las acepta del
  navegador): basta colgar y volver a llamar. La voz se elige entre las del proveedor: ``GET /api/voces`` da las de
  la cuenta de ElevenLabs (las pide este proceso con ``ELEVENLABS_API_KEY``; la clave no llega al navegador) y las
  de Gemini. La velocidad es ``tts_tempo``, que vale para cualquier proveedor.
- Modelo y Naturalidad son configuración del agente (``hesitation`` y ``tts_rhythm`` no se pueden pedir desde el
  navegador), así que se pasan como variables de entorno a la demo y hay que rearrancarla.

Solo biblioteca estándar, como el playground.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]

try:
    from timbal.server import playground
except ImportError:  # lanzado con un python sin el paquete instalado
    sys.path.insert(0, str(REPO / "python"))
    from timbal.server import playground

TITUBEOS = ["eh...", "mm...", "a ver,"]
"""Sonidos de titubeo cuando la demo no trae los suyos. Una lista vale para cualquier idioma de respuesta."""

VARIANTES: list[dict[str, Any]] = [
    {
        "id": "voz",
        "titulo": "Proveedor de voz",
        "aplica": "llamada",
        "opciones": [
            {"id": "", "nombre": "Por defecto"},
            {
                "id": "eleven-v3",
                "nombre": "Eleven v3 conversacional",
                "hello": {"tts_provider": "elevenlabs", "tts_model": "eleven_v3_conversational"},
            },
            {
                "id": "eleven-flash",
                "nombre": "Eleven Flash v2.5",
                "hello": {"tts_provider": "elevenlabs", "tts_model": "eleven_flash_v2_5"},
            },
            {
                "id": "gemini",
                "nombre": "Gemini TTS",
                "hello": {"tts_provider": "gemini", "tts_model": "gemini-3.8-flash-tts", "voice": "Fenrir"},
            },
        ],
    },
    {
        "id": "modelo",
        "titulo": "Modelo",
        "aplica": "arranque",
        "opciones": [
            {"id": "", "nombre": "Por defecto"},
            {"id": "gemini-2.5-flash", "nombre": "Gemini 2.5 Flash", "model": "google/gemini-2.5-flash"},
            {
                "id": "gemini-2.5-flash-lite",
                "nombre": "Gemini 2.5 Flash-Lite",
                "model": "google/gemini-2.5-flash-lite",
            },
            {"id": "grok-4.20", "nombre": "Grok 4.20 (algo más lento)", "model": "xai/grok-4.20-0309-non-reasoning"},
        ],
    },
    {
        "id": "naturalidad",
        "titulo": "Naturalidad",
        "aplica": "arranque",
        "opciones": [
            {"id": "", "nombre": "Por defecto"},
            {"id": "titubeos", "nombre": "Titubeos", "titubeos": True, "ritmo": False},
            {"id": "ritmo", "nombre": "Ritmo de pausas", "titubeos": False, "ritmo": True},
            {"id": "ambos", "nombre": "Los dos", "titubeos": True, "ritmo": True},
            {"id": "ninguno", "nombre": "Ninguno", "titubeos": False, "ritmo": False},
        ],
    },
    {
        "id": "escucha",
        "titulo": "Escucha",
        "aplica": "llamada",
        "opciones": [
            {"id": "", "nombre": "Por defecto"},
            {
                "id": "flux",
                "nombre": "Deepgram Flux",
                "hello": {"stt_provider": "deepgram-flux", "stt_model": "flux-general-multi"},
            },
            {
                "id": "nova",
                "nombre": "Deepgram Nova-3",
                "hello": {"stt_provider": "deepgram-nova", "stt_model": "nova-3"},
            },
            {
                "id": "scribe",
                "nombre": "ElevenLabs Scribe v3",
                "hello": {"stt_provider": "elevenlabs", "stt_model": "scribe_v3_live"},
            },
        ],
    },
]

HESITATION_RATE = 0.6
"""Con el 0,15 de ``hesitation=True`` una llamada corta puede no tener ningún titubeo: en la demo se tienen que oír.
Solo titubean las respuestas que empiezan por una frase de tres palabras o más, y nunca dos seguidas, así que con
0,35 salían entre cero y dos en una llamada de cinco frases."""


PROVEEDOR_DE_VOZ = "elevenlabs"
"""El proveedor de voz de una demo que no dice otro (``proveedor_voz`` en su entrada): el de ``VoiceConfig``."""

VOCES_GEMINI = [
    ("Zephyr", "luminosa"), ("Puck", "animada"), ("Charon", "informativa"), ("Kore", "firme"),
    ("Fenrir", "enérgica"), ("Leda", "juvenil"), ("Orus", "firme"), ("Aoede", "desenfadada"),
    ("Callirrhoe", "relajada"), ("Autonoe", "luminosa"), ("Enceladus", "susurrante"), ("Iapetus", "clara"),
    ("Umbriel", "relajada"), ("Algieba", "suave"), ("Despina", "suave"), ("Erinome", "clara"),
    ("Algenib", "rasgada"), ("Rasalgethi", "informativa"), ("Laomedeia", "animada"), ("Achernar", "delicada"),
    ("Alnilam", "firme"), ("Schedar", "equilibrada"), ("Gacrux", "madura"), ("Pulcherrima", "directa"),
    ("Achird", "cercana"), ("Zubenelgenubi", "informal"), ("Vindemiatrix", "tranquila"), ("Sadachbia", "vivaz"),
    ("Sadaltager", "experta"), ("Sulafat", "cálida"),
]  # fmt: skip
"""Las 30 voces de Gemini y cómo las describe Google (ai.google.dev/gemini-api/docs/speech-generation)."""

ELEVENLABS_VOCES_URL = "https://api.elevenlabs.io/v2/voices"
GRUPO_PROPIAS = "Propias de esta cuenta"
GRUPO_OTROS = "Otros idiomas"
_IDIOMAS = {"es": "Español", "ca": "Catalán", "en": "Inglés", "ar": "Árabe"}
_GENEROS = {"male": "hombre", "female": "mujer", "neutral": "neutra"}
_ACENTOS = {
    "peninsular": "España", "es-peninsular": "España", "spain": "España", "castillian": "España",
    "andalusian": "andaluz", "canary islands": "canario", "galician": "gallego", "argentine": "argentino",
    "colombian": "colombiano", "chilean": "chileno", "venezuelan": "venezolano", "mexican": "mexicano",
    "peruvian": "peruano", "ecuadorian": "ecuatoriano", "dominican": "dominicano",
    "latin american": "latinoamericano", "american": "EE. UU.", "british": "británico", "standard": "estándar",
    "moroccan": "marroquí",
}  # fmt: skip
_NOMBRE_MAX = 48


def voces_elevenlabs(crudas: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Las voces de una cuenta de ElevenLabs como las enseña la página: ``{id, nombre, detalle, grupo}``.

    Un grupo por idioma, con las clonadas o creadas en la cuenta delante. En cada grupo, primero las de España y
    al final las que no dicen acento.
    """
    grupos = [GRUPO_PROPIAS, *_IDIOMAS.values(), GRUPO_OTROS]
    voces: list[tuple[tuple[int, int, str, str], dict[str, str]]] = []
    for v in crudas:
        voice_id, nombre = str(v.get("voice_id") or "").strip(), " ".join(str(v.get("name") or "").split())
        if not voice_id or not nombre:
            continue
        etiquetas = v.get("labels") if isinstance(v.get("labels"), dict) else {}
        acento_crudo = str(etiquetas.get("accent") or "").strip().lower()
        acento = _ACENTOS.get(acento_crudo, acento_crudo)
        genero = _GENEROS.get(str(etiquetas.get("gender") or "").strip().lower(), "")
        if v.get("category") in ("cloned", "generated"):
            grupo = GRUPO_PROPIAS
        else:
            grupo = _IDIOMAS.get(str(etiquetas.get("language") or "").strip().lower(), GRUPO_OTROS)
        if len(nombre) > _NOMBRE_MAX:
            nombre = nombre[: _NOMBRE_MAX - 1].rstrip() + "…"
        orden = (grupos.index(grupo), 0 if acento == "España" else 1 if acento else 2, acento, nombre.casefold())
        voces.append(
            (
                orden,
                {
                    "id": voice_id,
                    "nombre": nombre,
                    "detalle": ", ".join(x for x in (genero, acento) if x),
                    "grupo": grupo,
                },
            )
        )
    voces.sort(key=lambda par: par[0])
    return [voz for _, voz in voces]


class SinVoces(Exception):
    """Las voces del proveedor no se han podido leer. El mensaje es para la página."""


def _clave_elevenlabs() -> str:
    """La del entorno o, como hace el playground con la demo, la del ``.env`` del repositorio."""
    clave = os.environ.get("ELEVENLABS_API_KEY") or playground._read_dotenv(REPO / ".env").get("ELEVENLABS_API_KEY")
    return (clave or "").strip()


def _pedir_voces_elevenlabs() -> list[dict[str, Any]]:
    clave = _clave_elevenlabs()
    if not clave:
        raise SinVoces("falta ELEVENLABS_API_KEY en el entorno o en el .env.")
    voces: list[dict[str, Any]] = []
    pagina_siguiente = ""
    for _ in range(10):  # 100 voces por página
        consulta = {"page_size": "100"}
        if pagina_siguiente:
            consulta["next_page_token"] = pagina_siguiente
        peticion = urllib.request.Request(
            f"{ELEVENLABS_VOCES_URL}?{urllib.parse.urlencode(consulta)}", headers={"xi-api-key": clave}
        )
        try:
            with urllib.request.urlopen(peticion, timeout=15) as respuesta:
                pagina = json.load(respuesta)
        except urllib.error.HTTPError as e:
            raise SinVoces(f"ElevenLabs respondió {e.code} al pedir las voces de la cuenta.") from None
        except (OSError, ValueError):
            raise SinVoces("no se pudo hablar con ElevenLabs para pedir las voces de la cuenta.") from None
        if not isinstance(pagina, dict):
            break
        voces += [v for v in pagina.get("voices") or [] if isinstance(v, dict)]
        pagina_siguiente = str(pagina.get("next_page_token") or "")
        if not pagina.get("has_more") or not pagina_siguiente:
            break
    return voces


_VOCES_VALEN_SECS = 600.0
_VOCES_LEIDAS: dict[str, Any] = {"voces": None, "cuando": 0.0}
_VOCES_CANDADO = threading.Lock()


def catalogo_de_voces() -> dict[str, dict[str, Any]]:
    """Las voces que se pueden elegir con cada proveedor: ``{proveedor: {"voces": [...], "error": ""}}``.

    Las de ElevenLabs se piden a la cuenta y valen diez minutos: una voz nueva aparece sin reiniciar. Si la cuenta
    no contesta se dan las últimas leídas, y sin ninguna, el motivo.
    """
    error = ""
    with _VOCES_CANDADO:
        if _VOCES_LEIDAS["voces"] is None or time.monotonic() - _VOCES_LEIDAS["cuando"] > _VOCES_VALEN_SECS:
            try:
                _VOCES_LEIDAS["voces"] = voces_elevenlabs(_pedir_voces_elevenlabs())
                _VOCES_LEIDAS["cuando"] = time.monotonic()
            except SinVoces as e:
                error = "" if _VOCES_LEIDAS["voces"] else str(e)
        de_elevenlabs = list(_VOCES_LEIDAS["voces"] or [])
    de_gemini = [{"id": nombre, "nombre": nombre, "detalle": detalle, "grupo": ""} for nombre, detalle in VOCES_GEMINI]
    return {
        "elevenlabs": {"voces": de_elevenlabs, "error": error},
        "gemini": {"voces": de_gemini, "error": ""},
    }


def demo_publica(demo: dict[str, Any]) -> dict[str, Any]:
    """Lo que la página necesita de una demo. Sin la ruta del agente ni sus variables de entorno."""
    publica = {k: demo[k] for k in ("id", "name", "description", "probar", "nota", "ocultar") if demo.get(k)}
    publica["proveedor_voz"] = str(demo.get("proveedor_voz") or PROVEEDOR_DE_VOZ)
    return publica


def _opcion(grupo: str, opcion_id: str) -> dict[str, Any]:
    for g in VARIANTES:
        if g["id"] == grupo:
            for o in g["opciones"]:
                if o["id"] == (opcion_id or ""):
                    return o
    raise ValueError(f"Variante desconocida: {grupo}={opcion_id!r}")


def cargar_demos(local: Path | None) -> list[dict[str, Any]]:
    """``demos.json`` más el fichero local si existe. Una demo cuyo fichero de agente no está se omite."""
    demos: list[dict[str, Any]] = []
    vistos: set[str] = set()
    for ruta in (AQUI / "demos.json", local):
        if ruta is None or not ruta.is_file():
            continue
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[hub] {ruta.name} no se puede leer: {e}", file=sys.stderr)  # noqa: T201
            continue
        for d in datos.get("demos", []) if isinstance(datos, dict) else datos:
            if not isinstance(d, dict) or not d.get("id") or not d.get("import_spec") or d["id"] in vistos:
                continue
            fichero = REPO / str(d["import_spec"]).split("::")[0]
            if not fichero.is_file():
                continue
            vistos.add(d["id"])
            demos.append(d)
    demos.sort(key=lambda d: d.get("orden", 100))
    return demos


def plan_de_arranque(demo: dict[str, Any], modelo: str, naturalidad: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Variables de entorno para la demo y lo que la página tiene que añadir al saludo de la llamada."""
    env = {str(k): str(v) for k, v in (demo.get("env") or {}).items()}
    # "hello": claves que el servidor acepta del navegador para esta demo en cada llamada (p. ej. un saludo).
    hello: dict[str, Any] = dict(demo.get("hello") or {})
    model = _opcion("modelo", modelo).get("model") or demo.get("default_model")
    if model:
        if demo.get("model_env"):
            env[demo["model_env"]] = model
        else:
            hello["model"] = model  # sin variable propia: el servidor acepta el modelo en el saludo
    nat = _opcion("naturalidad", naturalidad)
    if nat["id"]:
        if not demo.get("tuning_env"):
            raise ValueError("Esta demo no admite cambiar la naturalidad.")
        ajuste: dict[str, Any] = {
            "hesitation": (
                {"rate": HESITATION_RATE, "openers": demo.get("titubeos") or TITUBEOS} if nat["titubeos"] else False
            ),
            "tts_rhythm": bool(nat["ritmo"]),
        }
        env[demo["tuning_env"]] = json.dumps(ajuste, ensure_ascii=False)
    return env, hello


class HubChild(playground.ChildServer):
    """El ``ChildServer`` del playground, con variables de entorno por arranque.

    ``ChildServer.spawn`` construye el entorno del hijo a partir de ``os.environ``. Para no copiar ese método, las
    variables de la variante se ponen ahí mientras dura el ``spawn`` y se quitan después. Las que gestiona el hub
    (modelo y ajuste de voz de cada demo) se borran siempre antes, así «Por defecto» es lo que configura la demo.
    """

    def __init__(self) -> None:
        super().__init__()
        self._arranque = threading.Lock()
        self.clave: str | None = None
        self.hello: dict[str, Any] = {}
        self.gestionadas: set[str] = set()

    def spawn(  # type: ignore[override]
        self,
        import_spec: str,
        port: int | None = None,
        transport: str = "ws",
        *,
        extra_env: dict[str, str] | None = None,
        clave: str | None = None,
        hello: dict[str, Any] | None = None,
    ) -> dict:
        extra_env = extra_env or {}
        with self._arranque:
            tocadas = self.gestionadas | set(extra_env)
            antes = {k: os.environ.get(k) for k in tocadas}
            try:
                for k in tocadas:
                    os.environ.pop(k, None)
                os.environ.update(extra_env)
                estado = super().spawn(import_spec, port=port, transport=transport)
            finally:
                for k, v in antes.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.clave = clave
            self.hello = hello or {}
            return estado

    def stop(self) -> dict:
        with self._arranque:
            self.clave = None
            self.hello = {}
            return super().stop()

    def viva(self, clave: str) -> dict | None:
        """El estado del hijo si ya está arrancado (o arrancando) con esta misma demo y variante."""
        with self._arranque:
            if self.clave != clave:
                return None
            estado = self.status()
            return estado if estado.get("state") in ("running", "starting") else None


def _estado_publico(child: HubChild) -> dict[str, Any]:
    estado = child.status()
    estado["logs"] = [playground.redact_child_log(line) for line in estado.get("logs", [])]
    return {"child": estado, "clave": child.clave, "hello": child.hello}


def _make_handler(child: HubChild, local: Path | None):
    base = playground._make_handler(child, "")
    estado_html = AQUI / "estado.local.html"

    class Handler(base):  # type: ignore[misc, valid-type]
        server_version = "timbal-voice-demos"

        def _send_file(self, ruta: Path) -> None:
            body = ruta.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/":
                self._send_file(AQUI / "index.html")
                return
            if path == "/demos":
                demos = cargar_demos(local)
                child.gestionadas = {d[k] for d in demos for k in ("model_env", "tuning_env") if d.get(k)}
                self._send_json(
                    200,
                    {
                        "demos": [demo_publica(d) for d in demos],
                        "variantes": VARIANTES,
                        "estado": estado_html.is_file(),
                        "playground": "/voice",
                    },
                )
                return
            if path == "/estado":
                if estado_html.is_file():
                    self._send_file(estado_html)
                else:
                    self._send_json(404, {"error": "not found"})
                return
            if path == "/api/status":
                self._send_json(200, _estado_publico(child))
                return
            if path == "/api/voces":
                self._send_json(200, catalogo_de_voces())
                return
            super().do_GET()  # /voice y /launcher/*: el playground completo

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/stop":
                child.stop()
                self._send_json(200, _estado_publico(child))
                return
            if path != "/api/spawn":
                super().do_POST()
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Petición mal formada."})
                return
            demos = cargar_demos(local)
            child.gestionadas = {d[k] for d in demos for k in ("model_env", "tuning_env") if d.get(k)}
            demo = next((d for d in demos if d["id"] == body.get("demo")), None)
            if demo is None:
                self._send_json(404, {"error": "Esa demo no existe."})
                return
            modelo = str(body.get("modelo") or "")
            naturalidad = str(body.get("naturalidad") or "")
            clave = f"{demo['id']}|{modelo}|{naturalidad}"
            try:
                if child.viva(clave) is None:
                    env, hello = plan_de_arranque(demo, modelo, naturalidad)
                    child.spawn(str(REPO / demo["import_spec"]), extra_env=env, clave=clave, hello=hello)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            except OSError as e:
                self._send_json(500, {"error": f"No se pudo arrancar la demo: {e}"})
                return
            self._send_json(200, _estado_publico(child))

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Página sencilla de demos de voz de Timbal.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--no-open", action="store_true", help="No abrir el navegador.")
    parser.add_argument(
        "--local",
        default=os.environ.get("VOICE_DEMOS_LOCAL", str(AQUI / "demos.local.json")),
        help="Fichero con las demos que no van al repositorio. Vacío: ninguno.",
    )
    args = parser.parse_args(argv)

    # El playground lee el .env del directorio de trabajo y resuelve desde él las rutas de los agentes.
    os.chdir(REPO)
    # Si este entorno ya tiene el servidor y la voz instalados, la demo arranca sin que uv toque el entorno.
    if all(importlib.util.find_spec(m) for m in ("fastapi", "uvicorn", "onnxruntime")):
        os.environ.setdefault("UV_NO_SYNC", "1")

    local = Path(args.local).expanduser() if args.local.strip() else None
    child = HubChild()
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(child, local))
    except OSError as e:
        sys.exit(f"No se puede usar el puerto {args.port} ({e.strerror or e}). Prueba con --port 7790.")
    url = playground.playground_page_url(args.host, args.port)
    demos = cargar_demos(local)
    print(f"Demos de voz en {url}  ({len(demos)} demos: {', '.join(d['name'] for d in demos)})")  # noqa: T201
    if not args.no_open:
        webbrowser.open(url)

    def _terminar(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt  # un `kill` también tiene que parar la demo arrancada

    signal.signal(signal.SIGTERM, _terminar)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        child.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
