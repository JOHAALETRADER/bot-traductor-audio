import os
import io
import json
import zipfile
import subprocess
import tempfile
import urllib.request
from pathlib import Path

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
ES_STOPS = {" el ", " la ", " de ", " que ", " y ", " para ", " con ", " por ", " los ", " las ", " una ", " un "}
EN_STOPS = {" the ", " and ", " you ", " is ", " are ", " this ", " that ", " for ", " with ", " to "}

def guess_lang_by_stops(text: str) -> str:
    s = f" {text.lower()} "
    es_hits = sum(1 for w in ES_STOPS if w in s)
    en_hits = sum(1 for w in EN_STOPS if w in s)
    if es_hits >= 2 and es_hits > en_hits:
        return "es"
    if en_hits >= 2 and en_hits > es_hits:
        return "en"
    return "unknown"

# === Transcripción Vosk: devuelve (text_es, text_en) y decide mejor ===
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

    n_es = len(text_es.split())
    n_en = len(text_en.split())

    # Empate o cercanos: usa stops para inclinar a ES si corresponde
    if n_es == 0 and n_en == 0:
        return "", "unknown", text_es, text_en

    # Si ambos existen, inclinamos a ES si está cerca (80%) y contiene stops de ES
    if n_es >= 3 and (n_es >= int(max(1, n_en * 0.8))):
        hint_by_stops = guess_lang_by_stops(text_es)
        if hint_by_stops == "es":
            return text_es, "es", text_es, text_en

    # Caso general: más largo gana
    if n_es > n_en:
        return text_es, "es" if n_es >= 3 else "unknown", text_es, text_en
    elif n_en > n_es:
        return text_en, "en" if n_en >= 3 else "unknown", text_es, text_en
    else:
        # misma longitud > 0
        # usa stops
        lang = guess_lang_by_stops(text_es or text_en)
        if lang != "unknown":
            return (text_es or text_en), lang, text_es, text_en
        # última carta: langdetect
        try:
            from langdetect import detect
            lang = normalize_lang(detect(text_es or text_en))
        except Exception:
            lang = "unknown"
        return (text_es or text_en), lang, text_es, text_en

def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    try:
        from langdetect import detect
        return normalize_lang(detect(text))
    except Exception:
        return guess_lang_by_stops(text)

def translate(text: str, target: str, source_lang: str = None) -> str:
    """
    Traduce con deep-translator.
    source_lang puede ser 'es' o 'en'; si no se sabe, usa 'auto'.
    """
    try:
        from deep_translator import GoogleTranslator
        src = source_lang if source_lang in ("es", "en") else "auto"
        return GoogleTranslator(source=src, target=target).translate(text) or ""
    except Exception:
        return ""

# ========= TTS (ElevenLabs -> fallback gTTS) =========
def tts_to_mp3(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    """
    1) Si ELEVEN_API_KEY + ELEVEN_VOICE_ID -> ElevenLabs (voz clonada).
    2) Si falla o no hay credenciales -> gTTS.
    """
    eleven_api_key = os.getenv("ELEVEN_API_KEY", "").strip()
    eleven_voice_id = os.getenv("ELEVEN_VOICE_ID", "").strip()

    # ElevenLabs
    if eleven_api_key and eleven_voice_id:
        try:
            import requests
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

    # gTTS
    try:
        from gtts import gTTS
        tld = "com.mx" if lang_code.startswith("es") else "com"
        tts = gTTS(text=text, lang=("es" if lang_code.startswith("es") else "en"), tld=tld, slow=slow)
        tts.save(out_mp3_path)
        return True
    except Exception:
        return False

# ========= Handlers =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Traductor de audios:\n"
        "1) Envía una nota de voz en español o inglés.\n"
        "2) Respondo con transcripción, traducción y audio en el idioma destino."
    )
    await update.message.reply_text(msg)

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ok")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    if not voice:
        return

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    # 1) Descargar OGG
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    tg_file = await voice.get_file()
    await tg_file.download_to_drive(custom_path=tf_in.name)

    # 2) Convertir a WAV 16k mono
    wav_path = tf_in.name + ".wav"
    ok = ffmpeg_to_wav_mono16k(tf_in.name, wav_path)
    if not ok:
        await update.message.reply_text("No pude convertir el audio. Revisa FFmpeg.")
        return

    # 3) Transcribir con ambos modelos (y obtener pista de idioma robusta)
    text_best, src_hint, text_es, text_en = vosk_transcribe_both(wav_path)
    if not text_best:
        await update.message.reply_text("No pude transcribir el audio.")
        return

    # 4) Decidir destino: ES ↔ EN garantizado
    src = normalize_lang(src_hint)
    if src == "unknown":
        src = detect_lang(text_best)

    if src == "es":
        dst = "en"
    elif src == "en":
        dst = "es"
    else:
        # si aún dudamos, forzamos a español
        dst = "es"

    # 5) Traducir (indicamos source explícito si lo sabemos)
    translated = translate(text_best, target=dst, source_lang=src if src in ("es", "en") else None)

    # 6) Responder texto
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

    # 7) Audio de la traducción (con metadata y opción documento)
    if translated:
        out_mp3 = tf_in.name + ".mp3"
        if tts_to_mp3(translated, dst, out_mp3):
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_AUDIO)
            except Exception:
                pass

            send_as_document = os.getenv("SEND_AS_DOCUMENT", "").strip().lower() in ("1", "true", "yes")
            performer = (update.effective_user.first_name or "Johanna").strip()
            title = f"Traducción ({'EN' if dst == 'en' else 'ES'})"

            with open(out_mp3, "rb") as f:
                if send_as_document:
                    # Evita autoplay
                    await update.message.reply_document(
                        document=InputFile(f, filename=f"{title}.mp3"),
                        caption=f"{title} — {performer}"
                    )
                else:
                    # Audio con metadatos
                    await update.message.reply_audio(
                        audio=InputFile(f, filename=f"{title}.mp3"),
                        title=title,
                        performer=performer,
                        caption=title
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
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return app

if __name__ == "__main__":
    app = build_app()
    app.run_polling(drop_pending_updates=True)
