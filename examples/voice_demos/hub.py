"""Página sencilla para probar las demos de voz: eliges una demo, pulsas «Llamar» y hablas.

    uv run python examples/voice_demos/hub.py [--port 7777] [--no-open]

Sirve ``index.html`` en ``/``, la lista de demos en ``/demos`` y arranca por debajo un ``timbal.server`` con la demo
elegida (el mismo ``ChildServer`` del playground). El playground completo sigue disponible en ``/voice``.

Las demos salen de ``demos.json`` y, si existe, de ``demos.local.json`` (no se sube al repositorio: es donde van las
demos con marca de cliente). ``--local otra/ruta.json`` o ``VOICE_DEMOS_LOCAL`` cambian ese segundo fichero; un
valor vacío lo desactiva. ``GET /estado`` sirve ``estado.local.html`` si existe.

Qué cambia cada variante:

- Voz y Escucha viajan en el primer mensaje de la llamada (el servidor las acepta del navegador): basta colgar y
  volver a llamar.
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
        "titulo": "Voz",
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
                publicas = [
                    {k: d.get(k) for k in ("id", "name", "description", "probar", "nota", "ocultar") if d.get(k)}
                    for d in demos
                ]
                self._send_json(
                    200,
                    {
                        "demos": publicas,
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
