import os
import io
import json
import zipfile
import subprocess
import tempfile
import urllib.request
import shutil
import re
from pathlib import Path

# NUEVOS IMPORTS (para DeepL sin aiohttp)
import asyncio
import requests

from telegram import Update, InputFile
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ========= Utilidades de entorno =========
def getenv_stripped(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val

def normalize_lang(code: str) -> str:
    """Normaliza el código de idioma a 'es', 'en' o 'unknown'."""
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
ES_MODEL_DIR = MODELS_DIR / "vosk-model-small-es-0.42"
EN_MODEL_DIR = MODELS_DIR / "vosk-model-small-en-us-0.15"

ES_URL = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
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
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0

# ---------- heurística de idioma basada en stopwords ----------
ES_STOPS = {" el ", " la ", " de ", " que ", " y ", " para ", " con ", " por ", " los ", " las ", " una ", " un ", " en ", " como "}
EN_STOPS = {" the ", " and ", " you ", " is ", " are ", " this ", " that ", " for ", " with ", " to ", " in ", " of "}

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
    """
    Puntuación robusta:
      score = num_palabras + 2 * num_stopwords_propias
    Preferencia: si scores cercanos (<=2) y ES tiene >=1 stopword, favorecer ES.
    """
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

# === Utilidades extra para robustez ES→EN ===
def is_spanishish(text: str) -> bool:
    """Señales rápidas de español: tildes o muchas stopwords ES."""
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

# === Transcripción Vosk: devuelve (text_best, src_hint, text_es, text_en) ===
def vosk_transcribe_both(wav_path: str):
    """
    Retorna (text_best, src_hint, text_es, text_en)
    - src_hint: 'es' | 'en' | 'unknown'
    """
    try:
        import vosk
    except Exception:
        return "", "unknown", "", ""

    text_es = ""
    text_en = ""

    # ES
    try:
        ensure_model(ES_MODEL_DIR, ES_URL)
        model_es = __import__("vosk").Model(str(ES_MODEL_DIR))
        rec_es = __import__("vosk").KaldiRecognizer(model_es, 16000)
        with open(wav_path, "rb") as f:
            while True:
                data = f.read(4000)
                if not data:
                    break
                rec_es.AcceptWaveform(data)
        res = json.loads(rec_es.FinalResult() or "{}")
        text_es = (res.get("text") or "").strip()
    except Exception:
        text_es = ""

    # EN
    try:
        ensure_model(EN_MODEL_DIR, EN_URL)
        model_en = __import__("vosk").Model(str(EN_MODEL_DIR))
        rec_en = __import__("vosk").KaldiRecognizer(model_en, 16000)
        with open(wav_path, "rb") as f:
            while True:
                data = f.read(4000)
                if not data:
                    break
                rec_en.AcceptWaveform(data)
        res = json.loads(rec_en.FinalResult() or "{}")
        text_en = (res.get("text") or "").strip()
    except Exception:
        text_en = ""

    # Selección por puntuación robusta
    best, hint = pick_lang_by_score(text_es, text_en)
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

# ========= Traducción (DeepL -> Google) + Glosario Local =========
DEEPL_API_KEY = getenv_stripped("DEEPL_API_KEY", "")
DEEPL_API_HOST = getenv_stripped("DEEPL_API_HOST", "api-free.deepl.com")
USE_LOCAL_GLOSSARY = getenv_stripped("USE_LOCAL_GLOSSARY", "true").lower() in ("1","true","yes")

def translate_deepl(text: str, target: str, source_lang: str | None) -> str:
    """Traduce con DeepL si hay API; si falla, devuelve '' para que otro motor tome el relevo.
       Implementación con requests (sin aiohttp)."""
    if not DEEPL_API_KEY:
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
        # Para no bloquear demasiado, usamos to_thread (opcional); si falla, hacemos post directo.
        def _post():
            return requests.post(url, data=data, timeout=30)
        try:
            resp = asyncio.get_event_loop().run_until_complete(asyncio.to_thread(_post))  # si no hay loop activo, caerá al except
        except Exception:
            resp = requests.post(url, data=data, timeout=30)

        if resp.status_code != 200:
            return ""
        js = resp.json()
        return (js.get("translations", [{}])[0].get("text") or "").strip()
    except Exception:
        return ""

def translate_google(text: str, target: str, source_lang: str | None) -> str:
    try:
        from deep_translator import GoogleTranslator
        src = source_lang if source_lang in ("es","en") else "auto"
        return GoogleTranslator(source=src, target=target).translate(text) or ""
    except Exception:
        return ""

# --- Glosario local (post-proceso) ---
ES_EN_RULES = [
    (r"\bdep[oó]sito[s]?\s+m[ií]nimo[s]?\b", "minimum deposit"),
    (r"\bse[ñn]al(es)?\b", "signal"),
    (r"\bapalancamiento\b", "leverage"),
    (r"\bcuenta[s]?\b", "account"),
    (r"\bretirad[ao]s?\b", "withdrawal"),
]
EN_ES_RULES = [
    (r"\bminimum\s+deposit(s)?\b", "depósito mínimo"),
    (r"\bsignal(s)?\b", "señal"),
    (r"\bleverage\b", "apalancamiento"),
    (r"\baccount(s)?\b", "cuenta"),
    (r"\bwithdrawal(s)?\b", "retiro"),
]

def apply_local_glossary(text: str, direction: tuple[str,str]) -> str:
    """direction = ('es','en') o ('en','es')"""
    if not USE_LOCAL_GLOSSARY or not text:
        return text
    src, dst = direction
    rules = ES_EN_RULES if (src=="es" and dst=="en") else (EN_ES_RULES if (src=="en" and dst=="es") else [])
    out = text
    for pattern, repl in rules:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out

def translate_smart(text: str, target: str, source_lang: str | None) -> str:
    """Intenta DeepL y cae a Google, luego aplica glosario local."""
    translated = translate_deepl(text, target, source_lang)
    if not translated:
        translated = translate_google(text, target, source_lang)
    src_norm = (source_lang or "unknown")
    if src_norm not in ("es","en"):
        src_norm = detect_lang(text)
    if src_norm not in ("es","en"):
        src_norm = "es" if target.startswith("en") else "en"
    fixed = apply_local_glossary(translated, (src_norm, "en" if target.startswith("en") else "es"))
    return fixed

# ========= TTS (ElevenLabs -> fallback gTTS) =========
def synthesize_tts(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    """
    1) Si ELEVEN_API_KEY + ELEVEN_VOICE_ID -> ElevenLabs (voz principal).
    2) Si falla o no hay credenciales -> gTTS (fallback).
    """
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
            pass  # fallback a gTTS

    try:
        from gtts import gTTS
        tld = "com.mx" if lang_code.startswith("es") else "com"
        tts = gTTS(text=text, lang=("es" if lang_code.startswith("es") else "en"), tld=tld, slow=slow)
        tts.save(out_mp3_path)
        return True
    except Exception:
        return False

def adjust_speed_with_ffmpeg(in_mp3: str, out_mp3: str, speed: float) -> bool:
    """Ajusta la velocidad manteniendo tono con FFmpeg atempo (0.5–2.0)."""
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
    cmd = ["ffmpeg", "-y", "-i", in_mp3, "-filter:a", f"atempo={spd}", "-vn", out_mp3]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
        return True
    try:
        shutil.copyfile(in_mp3, out_mp3)
        return True
    except Exception:
        return False

def tts_to_mp3(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    """Orquesta: sintetiza (Eleven/gTTS) y luego aplica TTS_SPEED con FFmpeg atempo."""
    raw_mp3 = out_mp3_path + ".raw.mp3"
    ok = synthesize_tts(text, lang_code, raw_mp3, slow=slow)
    if not ok:
        return False
    speed = getenv_stripped("TTS_SPEED", "1.0")
    try:
        spd = float(speed)
    except Exception:
        spd = 1.0
    ok2 = adjust_speed_with_ffmpeg(raw_mp3, out_mp3_path, spd)
    try:
        os.remove(raw_mp3)
    except Exception:
        pass
    return ok2

# ========= Handlers =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Traductor de audios y texto:\n"
        "• Nota de voz ES → EN (texto + audio)\n"
        "• Nota de voz EN → ES (texto + audio)\n"
        "• Texto ES → audio EN\n"
        "• Texto EN → audio ES\n"
        "\nComandos:\n/health  (estado)"
    )
    await update.message.reply_text(msg)

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ok")

# ---- Texto a audio (ES<->EN) ----
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_in = (update.message.text or "").strip()
    if not text_in:
        return
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    src = detect_lang(text_in)
    if src == "es":
        dst = "en"
    elif src == "en":
        dst = "es"
    else:
        dst = "es"

    translated = translate_smart(text_in, target=dst, source_lang=src if src in ("es","en") else None)

    human_src = "Español" if src == "es" else "Inglés" if src == "en" else "desconocido"
    human_dst = "Inglés" if dst == "en" else "Español"
    reply_lines = [
        f"Idioma detectado (texto): {human_src}",
        "",
        "Original:",
        text_in,
        "",
        f"Traducción ({human_dst}):",
        translated if translated else "(no disponible)"
    ]
    await update.message.reply_text("\n".join(reply_lines))

    if translated:
        out_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        if tts_to_mp3(translated, dst, out_mp3):
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_AUDIO)
            except Exception:
                pass
            performer = (update.effective_user.first_name or "Johanna").strip()
            title = f"Traducción ({'EN' if dst == 'en' else 'ES'})"
            with open(out_mp3, "rb") as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"{title}.mp3"),
                    caption=f"{title} — {performer}"
                )
        else:
            await update.message.reply_text("No pude generar el audio de la traducción (TTS).")

# ---- Voz a texto+audio (ES<->EN) ----
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    if not voice:
        return

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    tg_file = await voice.get_file()
    await tg_file.download_to_drive(custom_path=tf_in.name)

    wav_path = tf_in.name + ".wav"
    ok = ffmpeg_to_wav_mono16k(tf_in.name, wav_path)
    if not ok:
        await update.message.reply_text("No pude convertir el audio. Revisa FFmpeg.")
        return

    text_best, src_hint, text_es, text_en = vosk_transcribe_both(wav_path)
    if not text_best:
        await update.message.reply_text("No pude transcribir el audio.")
        return

    es_hits = stop_hits(text_es, ES_STOPS)
    en_hits = stop_hits(text_en, EN_STOPS)
    if es_hits >= max(2, en_hits + 1) and len(text_es.split()) >= 2:
        text_best = text_es
        src_hint = "es"
    elif en_hits >= max(3, es_hits + 2) and len(text_en.split()) >= 2:
        text_best = text_en
        src_hint = "en"
    elif is_spanishish(text_es) and len(text_es.split()) >= 2 and len(text_es.split()) >= len(text_en.split()) - 1:
        text_best = text_es
        src_hint = "es"

    src = normalize_lang(src_hint)
    if src == "unknown":
        sguess = guess_lang_by_stops(text_best)
        src = sguess if sguess != "unknown" else detect_lang(text_best)

    dst = "en" if src == "es" else ("es" if src == "en" else "es")

    translated = translate_smart(text_best, target=dst, source_lang=src if src in ("es","en") else None)

    if src == "en":
        sim = jaccard_similarity(text_best, translated)
        if is_spanishish(text_best) or sim >= 0.75:
            src = "es"
            dst = "en"
            translated = translate_smart(text_best, target=dst, source_lang="es")

    human_src = "Español" if src == "es" else "Inglés" if src == "en" else "desconocido"
    human_dst = "Inglés" if dst == "en" else "Español"
    reply_lines = []
    reply_lines.append(f"Idioma detectado: {human_src}")
    reply_lines.append("")
    reply_lines.append("Transcripción:")
    reply_lines.append(text_best if text_best else "(vacío)")
    reply_lines.append("")
    reply_lines.append(f"Traducción ({human_dst}):")
    reply_lines.append(translated if translated else "(no disponible)")
    await update.message.reply_text("\n".join(reply_lines))

    if translated:
        out_mp3 = tf_in.name + ".mp3"
        if tts_to_mp3(translated, dst, out_mp3):
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_AUDIO)
            except Exception:
                pass
            performer = (update.effective_user.first_name or "Johanna").strip()
            title = f"Traducción ({'EN' if dst == 'en' else 'ES'})"
            with open(out_mp3, "rb") as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"{title}.mp3"),
                    caption=f"{title} — {performer}"
                )
        else:
            await update.message.reply_text("No pude generar el audio de la traducción (TTS).")

def build_app():
    bot_token = getenv_stripped("BOT_TOKEN", "")
    if not bot_token:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno (o vacío).")
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return app

if __name__ == "__main__":
    app = build_app()
    app.run_polling(drop_pending_updates=True)
