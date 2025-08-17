import os
import io
import json
import zipfile
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from telegram import Update
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

def vosk_transcribe(wav_path: str) -> str:
    try:
        import vosk
    except Exception:
        return ""

    # Intento ES primero
    text_es = ""
    try:
        ensure_model(ES_MODEL_DIR, ES_URL)
        rec_es = vosk.KaldiRecognizer(vosk.Model(str(ES_MODEL_DIR)), 16000)
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

    if len(text_es.split()) >= 3:
        return text_es

    # Intento EN si ES fue flojo
    text_en = ""
    try:
        ensure_model(EN_MODEL_DIR, EN_URL)
        rec_en = vosk.KaldiRecognizer(__import__("vosk").Model(str(EN_MODEL_DIR)), 16000)
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

    return text_en if len(text_en.split()) > len(text_es.split()) else text_es

def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        s = " " + text.lower() + " "
        # heurística simple para español
        if any(w in s for w in [" el ", " la ", " de ", " que ", " y ", " para ", " con ", " por ", " los ", " las "]):
            return "es"
        return "en"

def translate(text: str, target: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target).translate(text) or ""
    except Exception:
        return ""

# ========= TTS (gTTS) =========
def tts_to_mp3(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
    """
    lang_code: 'es' o 'en'
    Para español uso tld 'com.mx' (acento más latino). Para inglés dejo 'com'.
    """
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
        "2) Te devuelvo transcripción, traducción y audio en el idioma destino."
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

    # 3) Transcribir
    text = vosk_transcribe(wav_path)
    if not text:
        await update.message.reply_text("No pude transcribir el audio.")
        return

    # 4) Detectar idioma y FORZAR mapeo ES ↔ EN
    src_raw = detect_lang(text)            # p.ej. 'es', 'en', 'unknown'
    src = normalize_lang(src_raw)          # 'es' | 'en' | 'unknown'

    # ---- BLOQUE FORZADO ES ↔ EN ----
    if src == "es":
        dst = "en"
    elif src == "en":
        dst = "es"
    else:
        # Si no estamos seguros, forzamos a traducir SIEMPRE al español
        # (puedes cambiar a 'en' si prefieres)
        dst = "es"
        src = "unknown"
    # --------------------------------

    translated = translate(text, dst)

    # 5) Responder texto
    human_src = "Español" if src == "es" else "Inglés" if src == "en" else src
    human_dst = "Inglés" if dst == "en" else "Español"

    reply_lines = []
    reply_lines.append("Idioma detectado: {}".format(human_src))
    reply_lines.append("")
    reply_lines.append("Transcripción:")
    reply_lines.append(text)
    reply_lines.append("")
    reply_lines.append("Traducción ({}):".format(human_dst))
    reply_lines.append(translated if translated else "(no disponible)")
    await update.message.reply_text("\n".join(reply_lines))

    # 6) Generar audio de la traducción (TTS) y enviarlo
    if translated:
        out_mp3 = tf_in.name + ".mp3"
        tts_ok = tts_to_mp3(translated, dst, out_mp3)
        if tts_ok:
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_AUDIO)
            except Exception:
                pass
            with open(out_mp3, "rb") as f:
                await update.message.reply_audio(
                    audio=f,
                    title="Traducción",
                    caption="Audio generado ({})".format("en" if dst == "en" else "es")
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
