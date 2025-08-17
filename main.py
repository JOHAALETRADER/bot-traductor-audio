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

# === Transcripción Vosk: devuelve (texto, src_hint) ===
def vosk_transcribe(wav_path: str):
    """
    Retorna (text, src_hint) donde src_hint es 'es', 'en' o 'unknown'
    según qué modelo produjo más palabras.
    """
    try:
        import vosk
    except Exception:
        return "", "unknown"

    text_es = ""
    text_en = ""

    # Intento ES
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

    # Intento EN
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

    # Elegimos el más largo (mejor ajuste)
    if len(text_es.split()) > len(text_en.split()):
        return text_es, "es" if len(text_es.split()) >= 3 else "unknown"
    elif len(text_en.split()) > len(text_es.split()):
        return text_en, "en" if len(text_en.split()) >= 3 else "unknown"
    else:
        # empate o ambos vacíos
        if text_es:
            return text_es, "es"
        if text_en:
            return text_en, "en"
        return "", "unknown"

def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        s = " " + text.lower() + " "
        if any(w in s for w in [" el ", " la ", " de ", " que ", " y ", " para ", " con ", " los ", " las ", " por "]):
            return "es"
        if any(w in s for w in [" the ", " and ", " you ", " is ", " are ", " this ", " that "]):
            return "en"
        return "unknown"

def translate(text: str, target: str, source_lang: str = None) -> str:
    """
    Traduce con deep-translator. Si source_lang es 'es' o 'en', lo fijamos;
    si no, usamos 'auto'.
    """
    try:
        from deep_translator import GoogleTranslator
        src = source_lang if source_lang in ("es", "en") else "auto"
        return GoogleTranslator(source=src, target=target).translate(text) or ""
    except Exception:
        return ""

# ========= TTS (gTTS) =========
def tts_to_mp3(text: str, lang_code: str, out_mp3_path: str, slow: bool = False) -> bool:
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

    # 3) Transcribir (y obtener pista de idioma por modelo)
    text, src_hint = vosk_transcribe(wav_path)
    if not text:
        await update.message.reply_text("No pude transcribir el audio.")
        return

    # 4) Decidir idioma destino con base en src_hint (forzado ES ↔ EN)
    src = normalize_lang(src_hint)
    if src == "unknown":
        # fallback con langdetect si Vosk no dio pista clara
        src = normalize_lang(detect_lang(text))

    if src == "es":
        dst = "en"
    elif src == "en":
        dst = "es"
    else:
        # Si aun así no estamos seguros, forzamos español
        dst = "es"

    # 5) Traducir (indicando source si lo sabemos)
    translated = translate(text, target=dst, source_lang=src if src in ("es", "en") else None)

    # 6) Responder texto
    human_src = "Español" if src == "es" else "Inglés" if src == "en" else "desconocido"
    human_dst = "Inglés" if dst == "en" else "Español"

    reply_lines = []
    reply_lines.append(f"Idioma detectado: {human_src}")
    reply_lines.append("")
    reply_lines.append("Transcripción:")
    reply_lines.append(text if text else "(vacío)")
    reply_lines.append("")
    reply_lines.append(f"Traducción ({human_dst}):")
    reply_lines.append(translated if translated else "(no disponible)")
    await update.message.reply_text("\n".join(reply_lines))

    # 7) Audio de la traducción
    if translated:
        out_mp3 = tf_in.name + ".mp3"
        if tts_to_mp3(translated, dst, out_mp3):
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_AUDIO)
            except Exception:
                pass
            with open(out_mp3, "rb") as f:
                await update.message.reply_audio(
                    audio=f,
                    title="Traducción",
                    caption=f"Audio generado ({'en' if dst == 'en' else 'es'})"
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
