"""Demo de retos: un agente de atención al cliente de una eléctrica que se prueba hablando.

Luzia Energía es ficticia. Clientes, facturas, consumos y cambios viven en este
proceso, copiados por llamada: no se manda nada, no se cobra nada.

Los retos están metidos en los datos y en las herramientas, no en el prompt:

1. Dictado: el DNI llega como "uno dos tres cuatro cinco seis siete ocho zeta" (Flux entrega los códigos en
   palabras). ``identificar_cliente`` lo entiende y comprueba la letra; si un número se oyó mal, no cuadra y el
   agente tiene que volver a pedirlo sin decir cuál es la letra buena.
2. Correcciones: "no, perdona, el veinte". ``preparar_cambio`` se puede volver a llamar y sustituye al anterior.
3. Herramienta lenta: la factura de Jordi tarda 3,5 s. Se oye una frase del agente mientras espera.
4. Herramienta que falla: el suministro de Lola da error la primera vez. El agente lo dice y reintenta solo.
5. Varias herramientas a la vez: "¿por qué es tan alta?" pide factura y consumo en el mismo paso.
6. Escrituras con confirmación: día de cargo, potencia y reclamación se preparan, se leen y solo se guardan con
   un sí claro. Una potencia que no existe se rechaza con las que sí.
7. Interrupciones: pide "desglósamelo" y córtale a mitad.
8. Lo que no sabe: precio del kilovatio de mañana. No se lo inventa.
9. Pasar a una persona.

Clientes de prueba: Marta Soler, DNI 12345678Z (Madrid, factura casi el doble de lo normal). Jordi Puig,
47382915V (Girona, factura lenta). Lola Fernández, 28954731P (Sevilla, el sistema de incidencias falla una vez).
Para el reto de la letra, dicta una letra equivocada o cambia una cifra.

Voz: Pablo (ElevenLabs v3) con titubeos de apertura. En el panel, el preset "Gemini 3.8" cambia a Gemini con el
mismo agente. ``RETOS_NOMBRE`` cambia el nombre del agente si cambias a una voz de mujer.
``TIMBAL_VOICE_DEMO_TUNING='{"tts_tempo": 1.1}'`` sobrescribe cualquier clave de la configuración de voz.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import unicodedata
from typing import Any, Literal

from timbal import Agent
from timbal.voice import current_voice_session
from timbal.voice.speculative import documented_speculative_preset

# gemini-3.5-flash-lite sometimes answers a tool step with plain text («A ver, déjame mirarlo.ultima_factura»)
# instead of a tool call: 6 of 20 replays of the same request, and the voice then says the tool name. gemini-2.5-flash
# made the call 20 of 20 on that request, with a lower first-token time (353 ms vs 472 ms median).
_MODEL = os.environ.get("RETOS_MODEL") or os.environ.get("TIMBAL_VOICE_DEMO_MODEL") or "google/gemini-2.5-flash"
# gemini-2.5-flash "thinks" before it answers: the same slow request gave a first token in 0.4 s without thinking
# and 5-16 s with it, a dead line in a call. Voice turns are short, so thinking is off.
_MODEL_PARAMS: dict[str, Any] = {"reasoning_effort": "none"} if _MODEL.startswith("google/gemini-2.5-flash") else {}
_NOMBRE = os.environ.get("RETOS_NOMBRE", "Pablo")
_PABLO = "HIYif4jehvc9P9A8DYbX"
LENTA_SECS = 3.5
RAPIDA_SECS = 0.4

# --- Dictado -------------------------------------------------------------------------------------------------

_NIF_LETRAS = "TRWAGMYFPDXBNJZSQVHLCKE"
_CIFRAS = {
    "cero": "0", "uno": "1", "una": "1", "un": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
}  # fmt: skip
_DECENAS = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}  # fmt: skip
_NOMBRES_LETRA = {
    "be": "B", "ce": "C", "de": "D", "efe": "F", "ge": "G", "hache": "H", "jota": "J", "ka": "K", "ele": "L",
    "eme": "M", "ene": "N", "pe": "P", "cu": "Q", "erre": "R", "ese": "S", "te": "T", "uve": "V", "equis": "X",
    "ye": "Y", "zeta": "Z", "w": "W",
}  # fmt: skip
_REPETIR = {"doble": 2, "triple": 3}


def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))


def dictado_a_codigo(texto: str, *, letras: bool = True) -> str:
    """Lo que el cliente dictó, como código: cifras y letras, mayúsculas, sin las palabras de relleno.

    Entiende cifras sueltas ("uno dos"), números hasta noventa y nueve ("cuarenta y siete"), "doble cero",
    nombres de letra ("zeta", "uve doble") y el código ya escrito ("12345678Z"). Con ``letras=False`` solo
    devuelve cifras: así "el 5 de diciembre" no se convierte en "5D".
    """
    t = _sin_acentos(texto)
    t = re.sub(r"\b(uve doble|doble uve)\b", "w", t)
    t = re.sub(r"\bi griega\b", "ye", t)
    tokens = re.findall(r"[a-z0-9]+", t)
    out: list[str] = []
    i = 0
    veces = 1
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok in _REPETIR:
            veces = _REPETIR[tok]
            continue
        pieza: str | None = None
        if tok in _DECENAS:
            valor = _DECENAS[tok]
            if valor >= 30 and i + 1 < len(tokens) and tokens[i] == "y" and tokens[i + 1] in _CIFRAS:
                valor += int(_CIFRAS[tokens[i + 1]])
                i += 2
            pieza = str(valor)
        elif tok in _CIFRAS:
            pieza = _CIFRAS[tok]
        elif tok in _NOMBRES_LETRA:
            pieza = _NOMBRES_LETRA[tok]
        elif re.fullmatch(r"\d+", tok):
            pieza = tok
        elif re.fullmatch(r"[xyz]?\d+[a-z]?", tok):
            pieza = tok.upper()  # ya escrito: 12345678z, x1234567l
        elif len(tok) == 1 and tok != "y":
            pieza = tok.upper()  # una letra suelta: "a", "o", "e"
        elif tok == "y" and (not out or i >= len(tokens)):
            pieza = "Y"  # solo vale como letra al principio o al final del código
        if pieza is not None and not letras and not pieza.isdigit():
            pieza = None
        if pieza is not None:
            out.append(pieza * veces)
            veces = 1
    return "".join(out)


def letra_nif(numero: int) -> str:
    return _NIF_LETRAS[numero % 23]


def _nif(numero: int) -> str:
    return f"{numero:08d}{letra_nif(numero)}"


def _euros(x: float) -> str:
    return f"{x:.2f}".replace(".", ",") + " euros"


def _kw(x: float) -> str:
    return f"{x:g}".replace(".", ",")


# --- Datos ---------------------------------------------------------------------------------------------------

_IVA = 0.21
_IMPUESTO_ELECTRICO = 0.0511
_EUROS_POR_KW_MES = 3.20
_POTENCIAS = (1.15, 2.3, 3.45, 4.6, 5.75, 6.9, 8.05, 9.2, 10.35, 11.5)

_SEED_CLIENTES: dict[str, dict[str, Any]] = {
    _nif(12345678): {
        "nombre": "Marta Soler Vidal",
        "ciudad": "Madrid",
        "contrato": "4471 0922",
        "tarifa": "Estable 2.0TD",
        "potencia": 4.6,
        "dia_cargo": 5,
        "factura": {
            "mes": "octubre",
            "vence": "5 de diciembre",
            "estado": "pendiente de pago",
            "kwh": 412,
            "potencia": 15.20,
            "energia": 74.60,
            "alquiler": 0.85,
            "media_anterior": 64.80,
        },
        "consumo": [
            ("mayo", 180),
            ("junio", 175),
            ("julio", 190),
            ("agosto", 205),
            ("septiembre", 230),
            ("octubre", 412),
        ],
    },
    _nif(47382915): {
        "nombre": "Jordi Puig Ferrer",
        "ciudad": "Girona",
        "contrato": "7720 1135",
        "tarifa": "Estable 2.0TD",
        "potencia": 3.45,
        "dia_cargo": 10,
        "lenta": True,
        "factura": {
            "mes": "octubre",
            "vence": "10 de diciembre",
            "estado": "pendiente de pago",
            "kwh": 198,
            "potencia": 11.40,
            "energia": 35.70,
            "alquiler": 0.85,
            "media_anterior": 52.30,
        },
        "consumo": [
            ("mayo", 160),
            ("junio", 150),
            ("julio", 170),
            ("agosto", 175),
            ("septiembre", 185),
            ("octubre", 198),
        ],
    },
    _nif(28954731): {
        "nombre": "Lola Fernández Ortega",
        "ciudad": "Sevilla",
        "contrato": "3018 5567",
        "tarifa": "Noches 2.0TD",
        "potencia": 5.75,
        "dia_cargo": 15,
        "suministro_falla_una_vez": True,
        "factura": {
            "mes": "octubre",
            "vence": "15 de diciembre",
            "estado": "pendiente de pago",
            "kwh": 301,
            "potencia": 18.40,
            "energia": 52.10,
            "alquiler": 0.85,
            "media_anterior": 71.90,
        },
        "consumo": [
            ("mayo", 240),
            ("junio", 280),
            ("julio", 340),
            ("agosto", 360),
            ("septiembre", 310),
            ("octubre", 301),
        ],
        "corte": "Hay un corte programado por obras de mantenimiento en su barrio: mañana de nueve de la mañana a una de la tarde.",
    },
}


def desglose(f: dict[str, Any]) -> dict[str, float]:
    """Las partidas de una factura, con el total calculado a partir de ellas."""
    base = f["potencia"] + f["energia"]
    impuesto = round(base * _IMPUESTO_ELECTRICO, 2)
    subtotal = base + impuesto + f["alquiler"]
    iva = round(subtotal * _IVA, 2)
    return {
        "potencia": f["potencia"], "energia": f["energia"], "impuesto": impuesto,
        "alquiler": f["alquiler"], "iva": iva, "total": round(subtotal + iva, 2),
    }  # fmt: skip


# --- Estado por llamada --------------------------------------------------------------------------------------

_BAGS: dict[str, dict[str, Any]] = {}


def _session_key() -> str:
    session = current_voice_session()
    if session is None or not getattr(session, "session_id", None):
        return "sin-sesion"
    return str(session.session_id)


def _bag() -> dict[str, Any]:
    key = _session_key()
    bag = _BAGS.get(key)
    if bag is None:
        bag = {"clientes": copy.deepcopy(_SEED_CLIENTES), "cliente": None, "pendiente": None, "fallos": 0, "refs": 0}
        _BAGS[key] = bag
    return bag


def reset_demo_store() -> None:
    """Olvida todas las llamadas. Lo usan los tests."""
    _BAGS.clear()


def _cliente() -> dict[str, Any] | None:
    bag = _bag()
    return bag["clientes"].get(bag["cliente"]) if bag["cliente"] else None


_SIN_IDENTIFICAR = "Todavía no sabes con quién hablas. Pídele el documento de identidad o el número de extranjero, con la letra, y llama a identificar_cliente."


# --- Herramientas --------------------------------------------------------------------------------------------


async def identificar_cliente(documento: str) -> str:
    """Identifica al cliente por su DNI o NIE con letra. Pásalo tal como lo dictó, cifras dichas en palabras incluidas."""
    await asyncio.sleep(RAPIDA_SECS)
    codigo = dictado_a_codigo(documento)
    m = re.search(r"([XYZ]?)(\d{7,8})([A-Z]?)", codigo)
    if not m:
        return (
            f"No he sacado un documento de identidad ni un número de extranjero de lo que me han pasado (entendido: {codigo or 'nada'}). "
            "Pídele que lo repita despacio, cifra a cifra, con la letra al final."
        )
    prefijo, cifras, letra = m.groups()
    if not letra:
        return "Falta la letra del documento. Pídesela."
    numero = int({"": "", "X": "0", "Y": "1", "Z": "2"}[prefijo] + cifras)
    if letra != letra_nif(numero):
        return (
            f"La letra no cuadra con los números (he entendido {codigo}). Seguramente se ha oído mal algún dato. "
            "No le digas cuál es la letra correcta: pídele los números otra vez, cifra a cifra, y la letra."
        )
    bag = _bag()
    clave = prefijo + cifras + letra
    if clave not in bag["clientes"]:
        return f"No hay ningún cliente con el documento {codigo}. No inventes datos."
    bag["cliente"] = clave
    c = bag["clientes"][clave]
    return (
        f"Cliente identificado: {c['nombre']}, de {c['ciudad']}. Contrato {c['contrato']}, tarifa {c['tarifa']}, "
        f"potencia contratada {_kw(c['potencia'])} kilovatios, cargo domiciliado el día {c['dia_cargo']} de cada mes."
    )


async def ultima_factura() -> str:
    """Datos de la última factura del cliente identificado: importe, periodo, estado, desglose y comparación."""
    c = _cliente()
    if c is None:
        return _SIN_IDENTIFICAR
    await asyncio.sleep(LENTA_SECS if c.get("lenta") else RAPIDA_SECS)
    f = c["factura"]
    d = desglose(f)
    return (
        f"Factura de {f['mes']}: total {_euros(d['total'])}, {f['estado']}, vence el {f['vence']}. "
        f"Consumo {f['kwh']} kilovatios hora. Desglose: potencia {_euros(d['potencia'])}, energía {_euros(d['energia'])}, "
        f"impuesto eléctrico {_euros(d['impuesto'])}, alquiler del contador {_euros(d['alquiler'])}, IVA {_euros(d['iva'])}. "
        f"La media de los seis meses anteriores es {_euros(f['media_anterior'])}."
    )


async def historico_consumo() -> str:
    """Consumo mensual en kilovatios hora de los últimos seis meses del cliente identificado."""
    c = _cliente()
    if c is None:
        return _SIN_IDENTIFICAR
    await asyncio.sleep(RAPIDA_SECS)
    return "Consumo por mes: " + "; ".join(f"{mes} {kwh} kilovatios hora" for mes, kwh in c["consumo"]) + "."


async def estado_suministro() -> str:
    """Si hay una incidencia o un corte programado en el punto de suministro del cliente identificado."""
    bag = _bag()
    c = _cliente()
    if c is None:
        return _SIN_IDENTIFICAR
    await asyncio.sleep(RAPIDA_SECS + 0.2)
    if c.get("suministro_falla_una_vez") and bag["fallos"] == 0:
        bag["fallos"] = 1
        return "Error: el sistema de incidencias no ha respondido. Díselo y vuelve a intentarlo una vez."
    return c.get("corte") or "Sin incidencias ni cortes programados en su punto de suministro."


_TIPOS = {
    "dia_cargo": "dia_cargo", "dia_de_cargo": "dia_cargo", "dia_pago": "dia_cargo", "fecha_cargo": "dia_cargo",
    "potencia": "potencia", "potencia_contratada": "potencia", "reclamacion": "reclamacion",
}  # fmt: skip


async def preparar_cambio(tipo: Literal["dia_cargo", "potencia", "reclamacion"], valor: str) -> str:
    """Prepara un cambio SIN guardarlo. valor: el día (1 a 28), los kilovatios o el motivo de la reclamación."""
    bag = _bag()
    c = _cliente()
    if c is None:
        return _SIN_IDENTIFICAR
    await asyncio.sleep(0.2)
    tipo = _TIPOS.get(_sin_acentos(tipo).strip().replace(" ", "_"), "")
    if tipo == "dia_cargo":
        dia = dictado_a_codigo(valor, letras=False)
        if not dia.isdigit() or not 1 <= int(dia) <= 28:
            return "El día de cargo tiene que ser entre el uno y el veintiocho. Pregúntale qué día prefiere."
        bag["pendiente"] = {"tipo": tipo, "valor": int(dia)}
        return (
            f"Preparado, sin guardar: pasar el cargo del día {c['dia_cargo']} al día {int(dia)}. "
            "Léeselo y espera un sí claro antes de llamar a confirmar_cambio."
        )
    if tipo == "potencia":
        try:
            kw = float(valor.replace(",", ".").split()[0])
        except (ValueError, IndexError):
            return "No he entendido la potencia. Pregúntale cuántos kilovatios quiere."
        if kw not in _POTENCIAS:
            cerca = sorted(_POTENCIAS, key=lambda p: abs(p - kw))[:2]
            return (
                f"{_kw(kw)} kilovatios no es una potencia que se pueda contratar. Las más cercanas son "
                f"{_kw(cerca[0])} y {_kw(cerca[1])}. Ofrécele esas dos."
            )
        cambio = round((kw - c["potencia"]) * _EUROS_POR_KW_MES, 2)
        bag["pendiente"] = {"tipo": tipo, "valor": kw}
        sentido = "más" if cambio > 0 else "menos"
        return (
            f"Preparado, sin guardar: pasar de {_kw(c['potencia'])} a {_kw(kw)} kilovatios. Costaría unos {_euros(abs(cambio))} "
            f"{sentido} al mes. Léeselo y espera un sí claro antes de llamar a confirmar_cambio."
        )
    if tipo == "reclamacion":
        motivo = " ".join(valor.split())
        if not motivo:
            return "Pregúntale el motivo de la reclamación."
        bag["pendiente"] = {"tipo": tipo, "valor": motivo}
        return (
            f"Preparada, sin abrir: reclamación por «{motivo}». Léesela y espera un sí claro antes de confirmar_cambio."
        )
    return "Solo se pueden preparar dia_cargo, potencia o reclamacion."


async def confirmar_cambio() -> str:
    """Guarda el cambio preparado. Solo tras un sí claro del cliente."""
    bag = _bag()
    c = _cliente()
    pendiente = bag.get("pendiente")
    if c is None or not pendiente:
        return "No hay nada preparado. No digas que se ha guardado."
    await asyncio.sleep(0.3)
    bag["pendiente"] = None
    if pendiente["tipo"] == "dia_cargo":
        c["dia_cargo"] = pendiente["valor"]
        return f"Guardado: el cargo pasa al día {pendiente['valor']} desde la próxima factura."
    if pendiente["tipo"] == "potencia":
        c["potencia"] = pendiente["valor"]
        return (
            f"Guardado: potencia de {_kw(pendiente['valor'])} kilovatios. Tarda unos tres días laborables en aplicarse."
        )
    bag["refs"] += 1
    return f"Reclamación abierta, referencia R {bag['refs']}. Le contestarán en un máximo de diez días."


async def cancelar_cambio() -> str:
    """Descarta el cambio preparado. No deshace uno ya guardado."""
    bag = _bag()
    if not bag.get("pendiente"):
        return "No había nada preparado."
    bag["pendiente"] = None
    return "Descartado. No se ha guardado nada."


async def pasar_a_persona(motivo: str) -> str:
    """Pasa la llamada a una persona del equipo. Simulado: no marca a nadie."""
    await asyncio.sleep(0.2)
    return (
        f"Transferencia simulada a una compañera del equipo humano (motivo: {motivo}). Espera estimada: un minuto. "
        "Dile que le pasas con ella, despídete en una frase y llama a end_call."
    )


async def end_call() -> str:
    """Cuelga solo si el cliente se queda callado. Si dice que ha hablado otra vez, sigue ayudándole."""
    session = current_voice_session()
    if session is None:
        return "No hay llamada. No cuelgues."
    if not await session.commit_hangup():
        return "No cuelgues. Sigue ayudándole."
    return "Colgando."


# --- Prompt y configuración de voz ---------------------------------------------------------------------------

_PROMPT = f"""\
Eres {_NOMBRE}, del servicio de atención al cliente de Luzia Energía, una comercializadora de luz de España. \
Hablas por teléfono, en español de España. No te describas con adjetivos de género.

CÓMO SUENAS
- Como una persona real al teléfono, no como un guion. Cercano, tranquilo, con oficio. Tuteas; si el cliente te \
trata de usted, pasas a usted.
- Frases cortas. Una o dos por turno. Tres solo si te piden que se lo expliques con detalle.
- Empieza reaccionando a lo que te han dicho ("Vale.", "Ya veo.", "Claro.", "Vaya, entiendo.") y luego contesta. \
No repitas la misma reacción dos turnos seguidos.
- Nada de "estimado cliente", "le informo de que", "no dude en", "¡genial!" ni "¡perfecto!" a cada paso. Si el \
cliente está enfadado o preocupado, ve más despacio y no finjas alegría.
- Nada de listas, viñetas, markdown, emojis ni paréntesis: todo lo que escribes se dice en voz alta.
- Cifras, importes y fechas siempre con letras, como se dicen: «ciento quince euros con veinticuatro», \
«cuatrocientos doce kilovatios hora», «el cinco de diciembre». Nunca símbolos ni abreviaturas.
- Nada de siglas: di «documento de identidad» y «número de extranjero», nunca DNI ni NIE ni letras sueltas. El \
sintetizador las dice a trompicones, letra por letra y con pausas.
- Di el nombre del cliente una vez por llamada como mucho, y solo si suena natural.
- Al cerrar un asunto, como mucho una vez, «¿algo más?». Sin discursos de despedida.
- Si te interrumpen, para y atiende lo nuevo; no retomes lo que ibas diciendo.

QUÉ HACES
- Para hablar de facturas, consumo, suministro o cambios necesitas saber con quién hablas: pide el documento de \
identidad o el número de extranjero, con la letra. Cuando lo dicte, léelo de vuelta una vez, cifra a cifra, sin juntar cifras en números, en bloques de \
tres separados por punto y coma, no por comas: «uno dos tres; cuatro cinco seis; siete ocho; zeta» (el uno se \
dice «uno»; el último bloque puede ser de dos cifras o de una; en un número de extranjero empiezas por la letra: «equis; siete \
ocho nueve; …»). El punto y coma marca la pausa entre bloques y mantiene el ritmo estable. Espera el sí. Llama a \
identificar_cliente con lo que oíste, tal cual: entiende las cifras dichas en palabras.
- En cuanto identificar_cliente conteste, sigue sin preguntar con lo que el cliente pidió al principio (la \
factura, el consumo, el cambio): llama a la herramienta que toque. Nunca termines un turno con «un momento» o \
«déjame mirarlo»: o hablas de lo que has encontrado, o llamas a la herramienta.
- Si la letra no cuadra, no digas cuál es la buena: di que algún dato no te ha quedado claro y pídeselo otra vez, \
cifra a cifra.
- Si el cliente se corrige a mitad de dictado ("no, perdona, un cuatro"), cambia solo lo que corrige.
- Factura: ultima_factura. Si pregunta por qué es alta o qué ha pasado, llama a ultima_factura y a \
historico_consumo en el mismo paso. Contesta primero con el importe y la causa principal en una frase; el \
desglose solo si lo pide.
- Antes de una consulta que puede tardar, di una frase corta y natural ("a ver, déjame mirarlo") y llama a la \
herramienta; no esperes en silencio.
- Si una herramienta falla, dilo con naturalidad ("me ha dado un error, lo vuelvo a intentar") y reinténtalo una \
vez sin pedir permiso. Si vuelve a fallar, discúlpate y ofrece pasarle con una persona.
- Cambios (día de cargo, potencia, reclamación): preparar_cambio, léeselo y espera un sí claro; entonces \
confirmar_cambio. Si cambia de idea ("mejor el veinte"), vuelve a llamar a preparar_cambio. Si dice que no, \
cancelar_cambio. Nunca digas que algo está guardado hasta que confirmar_cambio lo diga.
- Si algo no lo sabes (precios de mañana, condiciones que no tienes), dilo tal cual y ofrece pasarle con una persona. \
No inventes.
- Si pide hablar con una persona, o está muy enfadado después de dos intentos, usa pasar_a_persona.
- Cuelga con end_call solo cuando el cliente se despide o te lo pide, y di antes un adiós corto. Si te da las \
gracias o te dice que sigas, quédate en la línea. Si end_call dice que no cuelgues, sigue ayudando.
"""


def retos_voice_config(model: str | None = None) -> dict[str, Any]:
    """Voz de España, titubeos de apertura, relleno corto en esperas largas y silencios que se retoman."""
    preset = documented_speculative_preset(model or _MODEL)
    config: dict[str, Any] = {
        "language": "es",
        "reply_language": "es",
        "voice": _PABLO,
        "tts_model": "eleven_v3_conversational",
        "keyterms": ["Luzia Energía", "kilovatios", "potencia contratada", "DNI", "NIE", "reclamación"],
        # Escritas así la voz las dice siempre igual; como siglas, unas veces de corrido y otras letra a letra.
        "pronunciations": {"DNI": "deneí", "NIE": "níe"},
        "ignore_backchannels": True,
        "speculative_generation": {"enabled": preset.enabled, "preemptive_tts": preset.preemptive_tts},
        "max_tts_pause_secs": 0.55,
        "first_flush_min_words": 4,
        "entity_patience": True,
        "hesitation": {"rate": 0.15, "openers": ["eh...", "mm...", "a ver,"]},
        "filler": {
            "enabled": True,
            "delay_secs": 1.2,
            "phrases": {"es": ["Un momento, que lo miro.", "A ver, déjame mirarlo.", "Mm, un segundo."]},
            "repeat_secs": 5.0,
            "max_per_turn": 2,
        },
        "greeting": {"text": f"Luzia Energía, buenas. Te atiende {_NOMBRE}, ¿en qué te puedo ayudar?"},
        "user_idle": {
            "timeout_secs": 7.0,
            "text": ["¿Sigues ahí?", "¿Me oyes bien?"],
            "max_count": 2,
            "hangup_after_secs": 30.0,
        },
        "hangup_grace_secs": 1.2,
    }
    tuning = os.environ.get("TIMBAL_VOICE_DEMO_TUNING")
    if tuning:
        config.update(json.loads(tuning))

    from timbal.voice import VoiceConfig

    VoiceConfig.model_validate(config)
    return config


voice_config = retos_voice_config()

agent = Agent(
    name="luzia_retos",
    model=_MODEL,
    model_params=_MODEL_PARAMS,
    system_prompt=_PROMPT,
    tools=[
        identificar_cliente,
        ultima_factura,
        historico_consumo,
        estado_suministro,
        preparar_cambio,
        confirmar_cambio,
        cancelar_cambio,
        pasar_a_persona,
        end_call,
    ],
    voice_config=voice_config,
)


if __name__ == "__main__":
    print(json.dumps({"model": _MODEL, "voice_config": voice_config}, indent=2, ensure_ascii=False))  # noqa: T201
