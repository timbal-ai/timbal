# ruff: noqa: T201
"""Comprueba las demos de voz de punta a punta, sin micrófono: un cliente sintetizado llama a cada una.

    uv run python examples/voice_demos/hub.py --port 7790 --no-open      # en otra terminal
    uv run python examples/voice_demos/verify.py --hub http://127.0.0.1:7790

Para cada demo, y para cada opción de cada variante, pide al hub que arranque la demo (el mismo camino que la
página), abre la llamada por WebSocket con el mismo primer mensaje que mandaría la página, espera el saludo, dice
la primera frase de «Qué probar» con una voz de ElevenLabs y comprueba que el agente contesta con audio. Con
``--guion`` la llamada por defecto dice el guion entero.

Necesita ``ELEVENLABS_API_KEY`` (voz del cliente) además de las claves de la demo. El audio del cliente se guarda en
el directorio temporal del sistema para no sintetizarlo dos veces.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
RATE = 16000
FRAME = RATE * 2 * 80 // 1000  # 80 ms, como la página
VOZ_CLIENTE = "UOIqAnmS11Reiei1Ytkc"
CACHE = Path(tempfile.gettempdir()) / "timbal_voice_demos_verify"


def _env() -> None:
    for raw in (REPO / ".env").read_text(encoding="utf-8").splitlines() if (REPO / ".env").is_file() else []:
        line = raw.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _http(hub: str, path: str, body: dict | None = None) -> dict:
    req = Request(
        hub + path,
        data=None if body is None else json.dumps(body).encode(),
        method="POST" if body is not None else "GET",
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", str(e))}


def audio_cliente(texto: str) -> bytes:
    CACHE.mkdir(exist_ok=True)
    ruta = CACHE / (hashlib.sha1(texto.encode()).hexdigest()[:16] + ".pcm")
    if ruta.is_file():
        return ruta.read_bytes()
    req = Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOZ_CLIENTE}?output_format=pcm_16000",
        data=json.dumps({"text": texto, "model_id": "eleven_flash_v2_5", "language_code": "es"}).encode(),
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as r:
        pcm = r.read()
    ruta.write_bytes(pcm)
    return pcm


def arrancar(hub: str, demo: str, modelo: str, naturalidad: str, espera: float = 180.0) -> dict:
    r = _http(hub, "/api/spawn", {"demo": demo, "modelo": modelo, "naturalidad": naturalidad})
    if r.get("error"):
        raise RuntimeError(r["error"])
    limite = time.time() + espera
    while True:
        estado = r["child"]["state"]
        if estado == "running":
            return r
        if estado in ("exited", "stopped"):
            raise RuntimeError("la demo no arrancó: " + " | ".join(r["child"].get("logs", [])[-3:])[-400:])
        if time.time() > limite:
            raise RuntimeError("la demo no arrancó a tiempo")
        time.sleep(0.5)
        r = _http(hub, "/api/status")


async def llamar(port: int, hello: dict, frases: list[str]) -> dict[str, Any]:
    """Una llamada: saludo, y por cada frase del cliente, su transcripción y la respuesta hablada."""
    import websockets

    s: dict[str, Any] = {
        "bytes": 0, "ultimo": 0.0, "primero_tras": None, "hechos": 0, "textos": [], "oido": [], "errores": [],
        "metricas": [], "meta": None, "vivo": True, "hablando": False, "fin_cliente": None, "play": 0.0,
    }  # fmt: skip
    out: dict[str, Any] = {"turnos": []}

    async with websockets.connect(f"ws://127.0.0.1:{port}/voice/ws", max_size=None) as ws:
        await ws.send(json.dumps({"sample_rate": RATE, **hello}))

        async def lector() -> None:
            try:
                async for m in ws:
                    if isinstance(m, bytes):
                        continue
                    d = json.loads(m)
                    ty = d.get("type")
                    if ty == "audio":
                        n = len(base64.b64decode(d["data"]))
                        ahora = time.time()
                        s["bytes"] += n
                        s["ultimo"] = ahora
                        s["play"] = max(s["play"], ahora) + n / 2 / RATE
                        if s["fin_cliente"] and s["primero_tras"] is None:
                            s["primero_tras"] = ahora - s["fin_cliente"]
                    elif ty == "session_started":
                        s["meta"] = d
                    elif ty == "agent_text_done":
                        s["hechos"] += 1
                        s["textos"].append(d.get("text") or "")
                    elif ty == "transcript_committed":
                        s["oido"].append(d.get("text") or "")
                    elif ty == "metrics":
                        s["metricas"].append(d.get("metrics") or {})
                    elif ty == "error":
                        s["errores"].append(d.get("message") or "")
                    elif ty == "session_ended":
                        break
            except Exception as e:  # noqa: BLE001
                s["errores"].append(f"socket: {type(e).__name__}")
            s["vivo"] = False

        async def linea() -> None:
            # Silencio entre frases y acuses de reproducción, como el navegador.
            try:
                while s["vivo"]:
                    if not s["hablando"]:
                        await ws.send(bytes(FRAME))
                    pendiente = max(0.0, s["play"] - time.time())
                    await ws.send(
                        json.dumps({"type": "playback", "played_ms": (s["bytes"] / 2 / RATE - pendiente) * 1000})
                    )
                    await asyncio.sleep(0.08)
            except Exception:  # noqa: BLE001
                s["vivo"] = False

        async def esperar_respuesta(hechos_antes: int, bytes_antes: int, limite_secs: float) -> None:
            limite = time.time() + limite_secs
            while time.time() < limite and s["vivo"]:
                await asyncio.sleep(0.25)
                if s["hechos"] > hechos_antes and s["bytes"] > bytes_antes and time.time() - s["ultimo"] > 1.5:
                    break
            # Deja que "suene" entero antes de hablar: lo contrario sería una interrupción.
            await asyncio.sleep(min(20.0, max(0.0, s["play"] - time.time()) + 0.4))

        tl = asyncio.create_task(lector())
        ts = asyncio.create_task(linea())
        await esperar_respuesta(0, 0, 40)
        out["saludo"] = " ".join(s["textos"])
        out["saludo_ms"] = round(s["bytes"] / 2 / RATE * 1000)
        s["textos"].clear()
        for frase in frases:
            if not s["vivo"]:
                break
            pcm = await asyncio.to_thread(audio_cliente, frase)
            antes_h, antes_b, antes_o, antes_m = s["hechos"], s["bytes"], len(s["oido"]), len(s["metricas"])
            s["primero_tras"], s["fin_cliente"] = None, None
            s["hablando"] = True
            for j in range(0, len(pcm), FRAME):
                await ws.send(pcm[j : j + FRAME])
                await asyncio.sleep(0.08)
            s["hablando"] = False
            s["fin_cliente"] = time.time()
            await esperar_respuesta(antes_h, antes_b, 45)
            m = s["metricas"][antes_m:] or [{}]
            out["turnos"].append(
                {
                    "cliente": frase,
                    "oido": " ".join(s["oido"][antes_o:]),
                    "agente": " ".join(s["textos"]),
                    "audio_ms": round((s["bytes"] - antes_b) / 2 / RATE * 1000),
                    "espera_s": None if s["primero_tras"] is None else round(s["primero_tras"], 2),
                    "eou_to_first_audio_ms": m[0].get("eou_to_first_audio_ms"),
                    "speech_end_to_transcript_ms": m[0].get("speech_end_to_transcript_ms"),
                    "hesitations": sum(x.get("hesitations") or 0 for x in m),
                }
            )
            s["textos"].clear()
        tl.cancel()
        ts.cancel()
    out["meta"] = {k: (s["meta"] or {}).get(k) for k in ("stt_provider", "stt_model", "tts_provider", "model")}
    out["errores"] = s["errores"]
    return out


def veredicto(r: dict[str, Any], espera: dict[str, str]) -> tuple[bool, str]:
    if r.get("excepcion"):
        return False, r["excepcion"]
    if r["saludo_ms"] < 300:
        return False, "no llegó el audio del saludo"
    if not r["turnos"]:
        return False, "la llamada se cortó tras el saludo"
    t = r["turnos"][0]
    if not t["oido"]:
        return False, "no transcribió al cliente"
    if t["audio_ms"] < 300 or not t["agente"].strip():
        return False, "no hubo respuesta hablada" + (f" ({r['errores'][0][:80]})" if r["errores"] else "")
    for k, v in espera.items():
        if (r["meta"].get(k) or "") != v:
            return False, f"el servidor no aplicó {k}={v} (usó {r['meta'].get(k)})"
    return True, f"saludo {r['saludo_ms']} ms de audio; respuesta a los {t['espera_s']} s"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hub", default="http://127.0.0.1:7790")
    ap.add_argument("--demo", action="append", help="Solo estas demos (id). Se puede repetir.")
    ap.add_argument("--grupo", action="append", help="Solo estas variantes (voz, modelo, naturalidad, escucha, base).")
    ap.add_argument("--opcion", action="append", help="Solo estas opciones (id). Se puede repetir.")
    ap.add_argument("--guion", action="store_true", help="La llamada por defecto dice el guion entero.")
    ap.add_argument("--frases", type=int, default=1, help="Frases del guion que dice cada una de las demás llamadas.")
    ap.add_argument("--paralelo", type=int, default=3, help="Llamadas a la vez contra la misma demo.")
    ap.add_argument("--out", default=str(CACHE / "resultados.json"))
    args = ap.parse_args()
    _env()
    hub = args.hub.rstrip("/")
    cat = _http(hub, "/demos")
    grupos = {g["id"]: g for g in cat["variantes"]}
    resultados: list[dict[str, Any]] = []

    def quiere(grupo: str, opcion: str) -> bool:
        return (not args.grupo or grupo in args.grupo) and (not args.opcion or opcion in args.opcion or grupo == "base")

    for demo in cat["demos"]:
        if args.demo and demo["id"] not in args.demo:
            continue
        ocultas = set(demo.get("ocultar") or [])

        async def una(
            grupo: str, opcion: dict, port: int, hello_hub: dict, frases: list[str], demo: dict = demo
        ) -> None:
            hello = {**hello_hub, **(opcion.get("hello") or {})}
            espera = {k: v for k, v in (opcion.get("hello") or {}).items() if k in ("stt_provider", "tts_provider")}
            if opcion.get("model"):
                espera["model"] = opcion["model"]
            try:
                r = await llamar(port, hello, frases)
            except Exception as e:  # noqa: BLE001
                r = {"excepcion": f"{type(e).__name__}: {e}"[:200]}
            ok, detalle = veredicto(r, espera)
            resultados.append(
                {
                    "demo": demo["id"],
                    "grupo": grupo,
                    "opcion": opcion["id"] or "por-defecto",
                    "ok": ok,
                    "detalle": detalle,
                    "llamada": r,
                }
            )
            print(
                f"{'OK   ' if ok else 'FALLO'} {demo['id']:8} {grupo:12} {opcion['id'] or 'por-defecto':22} {detalle}",
                flush=True,
            )

        # Por defecto, Voz y Escucha: la misma demo arrancada, cambia solo el primer mensaje de la llamada.
        tandas: list[tuple[str, dict]] = []
        if quiere("base", ""):
            tandas.append(("base", {"id": ""}))
        for gid in ("voz", "escucha"):
            for o in grupos.get(gid, {}).get("opciones", []):
                if o["id"] and f"{gid}:{o['id']}" not in ocultas and quiere(gid, o["id"]):
                    tandas.append((gid, o))
        if tandas:
            try:
                st = await asyncio.to_thread(arrancar, hub, demo["id"], "", "")
            except RuntimeError as e:
                print(f"FALLO {demo['id']:8} arranque: {e}", flush=True)
                resultados.append(
                    {"demo": demo["id"], "grupo": "base", "opcion": "arranque", "ok": False, "detalle": str(e)}
                )
                continue
            sem = asyncio.Semaphore(args.paralelo)

            async def limitada(g: str, o: dict, st: dict = st, sem: asyncio.Semaphore = sem, demo: dict = demo) -> None:
                async with sem:
                    frases = demo["probar"] if (g == "base" and args.guion) else demo["probar"][: args.frases]
                    await una(g, o, st["child"]["port"], st.get("hello") or {}, frases)

            await asyncio.gather(*(limitada(g, o) for g, o in tandas))

        # Modelo y Naturalidad: hay que rearrancar la demo con cada opción.
        for gid in ("modelo", "naturalidad"):
            for o in grupos.get(gid, {}).get("opciones", []):
                if not o["id"] or f"{gid}:{o['id']}" in ocultas or not quiere(gid, o["id"]):
                    continue
                try:
                    st = await asyncio.to_thread(
                        arrancar,
                        hub,
                        demo["id"],
                        o["id"] if gid == "modelo" else "",
                        o["id"] if gid == "naturalidad" else "",
                    )
                except RuntimeError as e:
                    print(f"FALLO {demo['id']:8} {gid:12} {o['id']:22} {e}", flush=True)
                    resultados.append(
                        {"demo": demo["id"], "grupo": gid, "opcion": o["id"], "ok": False, "detalle": str(e)}
                    )
                    continue
                await una(gid, o, st["child"]["port"], st.get("hello") or {}, demo["probar"][: args.frases])
                if gid == "naturalidad":
                    # Lo que de verdad se oyó: el servidor anota cada titubeo y cada respuesta con las pausas retocadas.
                    registro = "\n".join((_http(hub, "/api/status").get("child") or {}).get("logs") or [])
                    oido = {
                        "titubeos": registro.count('"event": "tts_hesitation"'),
                        "pausas_retocadas": registro.count('"event": "tts_rhythm_shaped"'),
                    }
                    resultados[-1]["oido"] = oido
                    print(f"      {demo['id']:8} {gid:12} {o['id']:22} en el registro: {oido}", flush=True)

    _http(hub, "/api/stop", {})
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    fallos = [r for r in resultados if not r["ok"]]
    print(f"\n{len(resultados) - len(fallos)} de {len(resultados)} bien. Detalle en {args.out}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
