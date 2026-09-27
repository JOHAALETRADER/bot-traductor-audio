import os
import io
import json
import zipfile
import subprocess
import tempfile
import urllib.request
import shutil
import re
import wave
import uuid
from pathlib import Path
from typing import Optional

# DeepL (requests, sin aiohttp)
import asyncio
import requests

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# ========= Acceso por múltiples IDs (NUEVO) =========
# Si no se define ALLOWED_USERS en env, por defecto permite 5958164558
ALLOWED_USERS = os.getenv("ALLOWED_USERS", "5958164558").split(",")
ALLOWED_USERS = [uid.strip() for uid in ALLOWED_USERS if uid.strip()]

def is_allowed(update: Update) -> bool:
    try:
        uid = str(update.effective_user.id)
        return (uid in ALLOWED_USERS) if ALLOWED_USERS else True
    except Exception:
        return False

# ========= Utilidades de entorno =========
def getenv_stripped(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val

def normalize_lang(code: str) -> str:
    """Normaliza 'es' / 'en' / 'unknown'."""
    if not code:
        return "unknown"
    s = code.lower().strip()
    if s.startswith("es"):
        return "es"
    if s.startswith("en"):
        return "en"
    return "unknown"

# ========= Modelos Vosk (ES y EN) =========
MODELS_DIR = Path("/app/models")
# ⚠️ Español: modelo GRANDE para mucha mejor precisión.
ES_MODEL_DIR = MODELS_DIR / "vosk-model-es-0.42"
EN_MODEL_DIR = MODELS_DIR / "vosk-model-small-en-us-0.15"

ES_URL = "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip"
EN_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

def ensure_model(model_dir: Path, url: str):
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        return
    zip_path = model_dir.parent / (model_dir.name + ".zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir.parent)
    zip_path.unlink(missing_ok=True)

def ffmpeg_to_wav_mono16k(input_path: str, out_path: str) -> bool:
    """Convierte cualquier audio a WAV mono 16k. Si no hay ffmpeg, falla con gracia."""
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except FileNotFoundError:
        # ffmpeg no está instalado
        return False

# ---------- heurística de idioma basada en stopwords ----------
ES_STOPS = {
    " el "," la "," de "," que "," y "," para "," con "," por "," los "," las ",
    " una "," un "," en "," como "," pero "," porque "," aquí "," eso "," este ",
    " esta "," esto "," así "," entonces "," hola "," gracias "," si "," no "
}
EN_STOPS = {
    " the "," and "," you "," is "," are "," this "," that "," for "," with ",
    " to "," in "," of "," on "," at "," it "," we "," they "," but "
}

def stop_hits(text: str, stops: set) -> int:
    s = f" {text.lower()} "
    return sum(1 for w in stops if w in s)

def guess_lang_by_stops(text: str) -> str:
    es_hits = stop_hits(text, ES_STOPS)
    en_hits = stop_hits(text, EN_STOPS)
    if es_hits >= 2 and es_hits > en_hits:
        return "es"
    if en_hits >= 2 and en_hits > es_hits:
        return "en"
    return "unknown"

def pick_lang_by_score(text_es: str, text_en: str):
    """score = num_palabras + 2*stopwords; desempate favorece ES."""
    n_es = len(text_es.split())
    n_en = len(text_en.split())
    h_es = stop_hits(text_es, ES_STOPS)
    h_en = stop_hits(text_en, EN_STOPS)
    score_es = n_es + 2*h_es
    score_en = n_en + 2*h_en
    if score_es == 0 and score_en == 0:
        return "", "unknown"
    if abs(score_es - score_en) <= 2:
        if h_es >= 1 and n_es >= 2:
            return text_es, "es"
        if h_en >= 1 and n_en >= 2:
            return text_en, "en"
    if score_es > score_en:
        return (text_es, "es" if n_es >= 2 else "unknown")
    else:
        return (text_en, "en" if n_en >= 2 else "unknown")

# === Utilidades de robustez ===
def is_spanishish(text: str) -> bool:
    if any(c in text for c in "áéíóúñ¿¡"):
        return True
    return stop_hits(text, ES_STOPS) >= 2

def jaccard_similarity(a: str, b: str) -> float:
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)

def strip_laughter_noises(t: str) -> str:
    # elimina risas/onomatopeyas y duplicados de espacios
    t = re.sub(r"\b(ja+|ha+|jaja+|jeje+|jiji+|ahaha+|eh+|uh+|mmm+)\b", " ", t, flags=re.I)
    t = re.sub(r"[^\wáéíóúñüÁÉÍÓÚÑÜ¿¡\s\.,;:\?!\-']", " ", t)  # limpia símbolos raros
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ========= Limpieza de "emoji words" =========
# Vosk/STT a veces convierte emojis en palabras ("party popper", "red heart", etc.)
# Esto las elimina antes de detectar idioma / traducir.
EMOJI_WORDS = [
    "party popper",
    "red heart",
    "blue heart",
    "green heart",
    "yellow heart",
    "purple heart",
    "black heart",
    "white heart",
    "hundred points",
    "trophy",
    "clapping hands",
    "smiling face",
    "grinning face",
    "face with tears of joy",
    "sparkles",
    "check mark",
    "warning",
]

_EMOJI_WORDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in EMOJI_WORDS) + r")\b",
    flags=re.IGNORECASE
)

def remove_emoji_words(text: str) -> str:
    if not text:
        return text
    t = _EMOJI_WORDS_RE.sub(" ", text)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ========= Reconocimiento Vosk =========
FORCE_STT_LANG = getenv_stripped("FORCE_STT_LANG", "auto").lower()  # 'es' | 'en' | 'auto'

def vosk_transcribe_both(wav_path: str, spoken_lang: Optional[str] = None):
    """
    Retorna (text_best, src_hint, text_es, text_en)
    src_hint: 'es' | 'en' | 'unknown'
    Con spoken_lang usa únicamente el modelo del idioma elegido.
    """
    if spoken_lang is not None and spoken_lang not in ("es", "en"):
        raise ValueError("Idioma de audio inválido")
    try:
        import vosk
    except Exception:
        return "", "unknown", "", ""

    text_es = ""
    text_en = ""

    # --- ES ---
    try:
        if spoken_lang == "en":
            raise ValueError("Modelo español no solicitado")
        ensure_model(ES_MODEL_DIR, ES_URL)
        model_es = __import__("vosk").Model(str(ES_MODEL_DIR))
        rec_es = __import__("vosk").KaldiRecognizer(model_es, 16000)
        chunks = []
        with wave.open(wav_path, "rb") as f:
            while True:
                data = f.readframes(4000)
                if not data:
                    break
                if rec_es.AcceptWaveform(data):
                    chunks.append(json.loads(rec_es.Result() or "{}").get("text", ""))
        chunks.append(json.loads(rec_es.FinalResult() or "{}").get("text", ""))
        text_es = strip_laughter_noises(" ".join(chunks).strip())
    except Exception:
        text_es = ""

    # --- EN (siempre en AUTO, y también si se fuerza EN) ---
    run_en = (spoken_lang == "en") or (spoken_lang is None and FORCE_STT_LANG in ("auto", "en"))
    if run_en:
        try:
            ensure_model(EN_MODEL_DIR, EN_URL)
            model_en = __import__("vosk").Model(str(EN_MODEL_DIR))
            rec_en = __import__("vosk").KaldiRecognizer(model_en, 16000)
            chunks = []
            with wave.open(wav_path, "rb") as f:
                while True:
                    data = f.readframes(4000)
                    if not data:
                        break
                    if rec_en.AcceptWaveform(data):
                        chunks.append(json.loads(rec_en.Result() or "{}").get("text", ""))
            chunks.append(json.loads(rec_en.FinalResult() or "{}").get("text", ""))
            text_en = strip_laughter_noises(" ".join(chunks).strip())
        except Exception:
            text_en = ""

    # Limpieza de emoji-words ANTES de elegir
    text_es = remove_emoji_words(text_es)
    text_en = remove_emoji_words(text_en)

    # Fuerza directa
    if spoken_lang == "es" or (spoken_lang is None and FORCE_STT_LANG == "es"):
        return (text_es, "es" if text_es else "unknown", text_es, text_en)
    if spoken_lang == "en" or (spoken_lang is None and FORCE_STT_LANG == "en"):
        return (text_en, "en" if text_en else "unknown", text_es, text_en)

    # AUTO: escoger mejor por score
    best, hint = pick_lang_by_score(text_es, text_en)
    best = remove_emoji_words(best)

    if not best:
        if text_es:
            return text_es, "es", text_es, text_en
        if text_en:
            return text_en, "en", text_es, text_en
        return "", "unknown", text_es, text_en

    return best, hint, text_es, text_en

def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    try:
        from langdetect import detect
        return normalize_lang(detect(text))
    except Exception:
        return guess_lang_by_stops(text)

# ========= Traducción (DeepL -> Google) + Glosario =========
DEEPL_API_KEY = getenv_stripped("DEEPL_API_KEY", "")
DEEPL_API_HOST = getenv_stripped("DEEPL_API_HOST", "api-free.deepl.com")
USE_DEEPL = getenv_stripped("USE_DEEPL", "true").lower() in ("1", "true", "yes")
USE_LOCAL_GLOSSARY = getenv_stripped("USE_LOCAL_GLOSSARY", "false").lower() in ("1","true","yes")

def translate_deepl(text: str, target: str, source_lang: Optional[str]) -> str:
    if not USE_DEEPL or not DEEPL_API_KEY:
        return ""
    tgt = "EN" if str(target).lower().startswith("en") else "ES"
    src = None
    if (source_lang or "").lower().startswith("en"):
        src = "EN"
    elif (source_lang or "").lower().startswith("es"):
        src = "ES"
    url = f"https://{DEEPL_API_HOST}/v2/translate"
    data = {"auth_key": DEEPL_API_KEY, "text": text, "target_lang": tgt}
    if src:
        data["source_lang"] = src
    try:
        def _post():
            return requests.post(url, data=data, timeout=30)
        resp = _post()
        if resp.status_code != 200:
            return ""
        js = resp.json()
        return (js.get("translations", [{}])[0].get("text") or "").strip()
    except Exception:
        return ""

def translate_google(text: str, target: str, source_lang: Optional[str]) -> str:
    try:
        from deep_translator import GoogleTranslator
        src = source_lang if source_lang in ("es","en") else "auto"
        return GoogleTranslator(source=src, target=target).translate(text) or ""
    except Exception:
        return ""

# Glosario local
ES_EN_RULES = [
    (r"\bdep[oó]sito[s]?\s+m[ií]nimo[s]?\b", "minimum deposit"),
    (r"\bse[ñn]ales\b", "signals"),
    (r"\bse[ñn]al\b", "signal"),
    (r"\bapalancamiento\b", "leverage"),
    (r"\bcuenta[s]?\b", "account"),
    (r"\bretirad[ao]s?\b", "withdrawal"),
]
EN_ES_RULES = [
    (r"\bminimum\s+deposit(s)?\b", "depósito mínimo"),
    (r"\bsignals\b", "señales"),
    (r"\bsignal\b", "señal"),
    (r"\bleverage\b", "apalancamiento"),
    (r"\baccount(s)?\b", "cuenta"),
    (r"\bwithdrawal(s)?\b", "retiro"),
]

def apply_local_glossary(text: str, direction: tuple[str,str]) -> str:
    if not USE_LOCAL_GLOSSARY or not text:
        return text
    src, dst = direction
    rules = ES_EN_RULES if (src=="es" and dst=="en") else (EN_ES_RULES if (src=="en" and dst=="es") else [])
    out = text
    for pattern, repl in rules:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out

def translate_smart(text: str, target: str, source_lang: Optional[str]) -> str:
    t = translate_deepl(text, target, source_lang)
    if not t:
        t = translate_google(text, target, source_lang)
    src_norm = (source_lang or "unknown")
    if src_norm not in ("es","en"):
        src_norm = detect_lang(text)
    if src_norm not in ("es","en"):
        src_norm = "es" if target.startswith("en") else "en"
    return apply_local_glossary(t, (src_norm, "en" if target.startswith("en") else "es"))

# ========= TTS (ElevenLabs -> gTTS) =========
def synthesize_tts(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    eleven_api_key = os.getenv("ELEVEN_API_KEY", "").strip()
    eleven_voice_id = os.getenv("ELEVEN_VOICE_ID", "").strip()

    if eleven_api_key and eleven_voice_id:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{eleven_voice_id}"
            headers = {
                "xi-api-key": eleven_api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200 and r.content:
                with open(out_mp3_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass  # cae a gTTS

    try:
        from gtts import gTTS
        tld = "com.mx" if lang_code.startswith("es") else "com"
        gTTS(text=text, lang=("es" if lang_code.startswith("es") else "en"), tld=tld, slow=slow).save(out_mp3_path)
        return True
    except Exception:
        return False

def adjust_speed_with_ffmpeg(in_mp3: str, out_mp3: str, speed: float) -> bool:
    """Ajusta velocidad manteniendo tono con atempo; si no hay ffmpeg, copia tal cual."""
    try:
        spd = max(0.5, min(2.0, float(speed)))
    except Exception:
        spd = 1.0
    if abs(spd - 1.0) < 1e-3:
        try:
            shutil.copyfile(in_mp3, out_mp3)
            return True
        except Exception:
            return False
    try:
        cmd = ["ffmpeg", "-y", "-i", in_mp3, "-filter:a", f"atempo={spd}", "-vn", out_mp3]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
            return True
    except FileNotFoundError:
        pass
    # Sin ffmpeg o fallo: copia
    try:
        shutil.copyfile(in_mp3, out_mp3)
        return True
    except Exception:
        return False

def tts_to_mp3(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    raw = out_mp3_path + ".raw.mp3"
    if not synthesize_tts(text, lang_code, raw, slow=slow):
        return False
    speed = getenv_stripped("TTS_SPEED", "0.95")  # un pelín más lento por default
    try:
        spd = float(speed)
    except Exception:
        spd = 1.0
    ok = adjust_speed_with_ffmpeg(raw, out_mp3_path, spd)
    try:
        os.remove(raw)
    except Exception:
        pass
    return ok

# ========= Handlers =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        try:
            await update.message.reply_text("Bot privado: acceso no autorizado.")
        except Exception:
            pass
        return

    msg = (
        "Traductor de audios y texto:\n"
        "• Envía texto y elige traducción en texto o audio.\n"
        "• Envía un audio, indica si se habla en español o inglés y revisa su transcripción.\n"
        "• Luego elige la traducción en texto o audio.\n"
        "\nComandos:\n/health (estado)"
    )
    await update.message.reply_text(msg)

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        try:
            await update.message.reply_text("Bot privado: acceso no autorizado.")
        except Exception:
            pass
        return
    await update.message.reply_text("ok")

# Cada solicitud guarda su propio texto para que los botones no mezclen mensajes.
def _remember(context, text, source="unknown"):
    key = uuid.uuid4().hex[:12]
    items = context.user_data.setdefault("translations", {})
    items[key] = (text, source)
    while len(items) > 20:
        items.pop(next(iter(items)))
    return key


def _buttons(key, source):
    if source == "es":
        directions = [("🇬🇧 Inglés", "en")]
    elif source == "en":
        directions = [("🇪🇸 Español", "es")]
    else:
        directions = [("🇬🇧 Inglés", "en"), ("🇪🇸 Español", "es")]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{label} · texto", callback_data=f"tr:{key}:{dst}:t"),
         InlineKeyboardButton(f"{label} · audio", callback_data=f"tr:{key}:{dst}:a")]
        for label, dst in directions
    ])


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await update.message.reply_text("Bot privado: acceso no autorizado.")
        return
    text_in = (update.message.text or "").strip()
    if not text_in:
        return
    src = detect_lang(text_in)
    key = _remember(context, text_in, src)
    await update.message.reply_text(
        "Elige cómo quieres la traducción:", reply_markup=_buttons(key, src)
    )


async def handle_translation_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not is_allowed(update):
        await query.answer("Acceso no autorizado", show_alert=True)
        return
    try:
        _, key, dst, kind = query.data.split(":")
        if dst not in ("es", "en") or kind not in ("t", "a"):
            raise ValueError("Opción inválida")
        text_in, hint = context.user_data.get("translations", {})[key]
    except (ValueError, KeyError, AttributeError):
        await query.answer("Esta opción expiró. Envía el mensaje otra vez.", show_alert=True)
        return
    await query.answer()
    src = "en" if dst == "es" else "es"
    translated = await asyncio.to_thread(translate_smart, text_in, dst, src)
    if not translated:
        await query.message.reply_text("No pude traducir ahora. Comprueba la clave y el host de DeepL o la conexión del proveedor alternativo.")
        return
    label = "Español" if dst == "es" else "Inglés"
    if kind == "t":
        # Telegram limita los mensajes de texto a 4096 caracteres.
        for start in range(0, len(translated), 3900):
            await query.message.reply_text(f"Traducción ({label}):\n{translated[start:start + 3900]}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out_mp3 = os.path.join(tmp, "traduccion.mp3")
        ok = await asyncio.to_thread(tts_to_mp3, translated, dst, out_mp3)
        if not ok:
            await query.message.reply_text("La traducción está lista, pero no pude generar el audio. Puedes usar el botón de texto.")
            return
        with open(out_mp3, "rb") as f:
            await query.message.reply_document(
                document=InputFile(f, filename=f"Traducción_{dst.upper()}.mp3"),
                caption=f"Traducción en {label}"
            )


async def _queue_audio(update, context, file_id, suffix):
    key = uuid.uuid4().hex[:12]
    pending = context.user_data.setdefault("pending_audio", {})
    pending[key] = (file_id, suffix)
    while len(pending) > 20:
        pending.pop(next(iter(pending)))
    await update.message.reply_text(
        "¿En qué idioma se habla en este audio?",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🇪🇸 Español", callback_data=f"stt:{key}:es"),
            InlineKeyboardButton("🇬🇧 Inglés", callback_data=f"stt:{key}:en")
        ]])
    )


async def handle_audio_language(update, context):
    query = update.callback_query
    if not is_allowed(update):
        await query.answer("Acceso no autorizado", show_alert=True)
        return
    try:
        _, key, src = query.data.split(":")
        if src not in ("es", "en"):
            raise ValueError()
        file_id, suffix = context.user_data["pending_audio"][key]
    except (KeyError, ValueError):
        await query.answer("Esta opción expiró. Envía el audio otra vez.", show_alert=True)
        return
    await query.answer()
    tg_file = await context.bot.get_file(file_id)
    await _process_audio_file(query.message, context, tg_file, suffix, src)


async def _process_audio_file(message, context: ContextTypes.DEFAULT_TYPE, tg_file, tmp_suffix: str, src: str):
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    with tempfile.TemporaryDirectory() as tmp:
        suffix = tmp_suffix if tmp_suffix.lower() in (".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac", ".flac") else ".audio"
        input_path = os.path.join(tmp, "entrada" + suffix)
        wav_path = os.path.join(tmp, "entrada.wav")
        try:
            await tg_file.download_to_drive(custom_path=input_path)
            ok = await asyncio.to_thread(ffmpeg_to_wav_mono16k, input_path, wav_path)
            if not ok:
                await message.reply_text("No pude convertir el audio. Revisa que FFmpeg esté instalado.")
                return
            best, hint, text_es, text_en = await asyncio.to_thread(vosk_transcribe_both, wav_path, src)
        except Exception:
            await message.reply_text("No pude descargar o procesar este audio. Inténtalo de nuevo.")
            return
    if not best:
        await message.reply_text("No pude transcribir este audio. Comprueba que los modelos Vosk estén disponibles y que se escuche la voz.")
        return
    key = _remember(context, best, src)
    for start in range(0, len(best), 3800):
        title = "Transcripción:" if start == 0 else "Transcripción (continuación):"
        await message.reply_text(f"{title}\n{best[start:start + 3800]}")
    await message.reply_text(
        "Revisa la transcripción. Si no corresponde al audio, vuelve a enviarlo o elige el otro idioma. Elige el formato de traducción:",
        reply_markup=_buttons(key, src)
    )

# ---- Nota de voz (VOICE) ----
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        try:
            await update.message.reply_text("Bot privado: acceso no autorizado.")
        except Exception:
            pass
        return
    voice = update.message.voice
    if not voice:
        return
    await _queue_audio(update, context, voice.file_id, ".ogg")

# ---- Audio normal (AUDIO) ----
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        try:
            await update.message.reply_text("Bot privado: acceso no autorizado.")
        except Exception:
            pass
        return
    audio = update.message.audio
    if not audio:
        return
    suffix = os.path.splitext(audio.file_name or "audio.mp3")[1] or ".mp3"
    await _queue_audio(update, context, audio.file_id, suffix)

# ---- Documento con audio (DOCUMENT) ----
async def handle_document_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        try:
            await update.message.reply_text("Bot privado: acceso no autorizado.")
        except Exception:
            pass
        return
    doc = update.message.document
    if not doc:
        return
    name = (doc.file_name or "").lower()
    is_audio_doc = (doc.mime_type or "").startswith("audio/") or any(
        name.endswith(ext) for ext in (".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus")
    )
    if not is_audio_doc:
        return
    suffix = os.path.splitext(doc.file_name or "file.mp3")[1] or ".mp3"
    await _queue_audio(update, context, doc.file_id, suffix)

def build_app():
    bot_token = getenv_stripped("BOT_TOKEN", "")
    if not bot_token:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno (o vacío).")
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CallbackQueryHandler(handle_translation_choice, pattern=r"^tr:"))
    app.add_handler(CallbackQueryHandler(handle_audio_language, pattern=r"^stt:"))

    # Texto (no comando)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    # Notas de voz
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    # Audios normales
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    # Documentos con audio (al final)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document_audio))
    return app

if __name__ == "__main__":
    app = build_app()
    app.run_polling(drop_pending_updates=True)
