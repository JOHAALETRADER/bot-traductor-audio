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
from typing import Optional

# DeepL (requests, sin aiohttp)
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

# ========= Reconocimiento Vosk =========
FORCE_STT_LANG = getenv_stripped("FORCE_STT_LANG", "es").lower()  # 'es' | 'en' | 'auto'

def vosk_transcribe_both(wav_path: str):
    """
    Retorna (text_best, src_hint, text_es, text_en)
    src_hint: 'es' | 'en' | 'unknown'
    Respeta FORCE_STT_LANG (por defecto 'es').
    """
    try:
        import vosk
    except Exception:
        return "", "unknown", "", ""

    text_es = ""
    text_en = ""

    # Siempre cargamos ES (principal)
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
        text_es = strip_laughter_noises((res.get("text") or "").strip())
    except Exception:
        text_es = ""

    # Dependiendo del modo, ejecutamos EN también
    run_en = (FORCE_STT_LANG == "auto") or (FORCE_STT_LANG == "en")
    if run_en:
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
            text_en = strip_laughter_noises((res.get("text") or "").strip())
        except Exception:
            text_en = ""

    # Si está forzado a ES/EN, elige directo
    if FORCE_STT_LANG == "es":
        return (text_es, "es" if text_es else "unknown", text_es, text_en)
    if FORCE_STT_LANG == "en":
        return (text_en, "en" if text_en else "unknown", text_es, text_en)

    # AUTO: elegir por puntuación y refuerzo
    best, hint = pick_lang_by_score(text_es, text_en)
    if not best:
        if text_es:
            return text_es, "es", text_es, text_en
        if text_en:
            return text_en, "en", text_es, text_en
        return "", "unknown", text_es, text_en

    # Refuerzo fuerte a ES si hay señales
    n_es, n_en = len(text_es.split()), len(text_en.split())
    es_hits = stop_hits(text_es, ES_STOPS)
    en_hits = stop_hits(text_en, EN_STOPS)
    if es_hits >= 1 and n_es >= max(3, n_en - 2):
        return text_es, "es", text_es, text_en
    if en_hits >= 2 and n_en >= max(3, n_es + 1):
        return text_en, "en", text_es, text_en
    if n_es >= n_en * 1.25 and n_es >= 3:
        return text_es, "es", text_es, text_en
    if n_en >= n_es * 1.25 and n_en >= 3:
        return text_en, "en", text_es, text_en
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
USE_LOCAL_GLOSSARY = getenv_stripped("USE_LOCAL_GLOSSARY", "true").lower() in ("1","true","yes")

def translate_deepl(text: str, target: str, source_lang: Optional[str]) -> str:
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
        def _post():
            return requests.post(url, data=data, timeout=30)
        try:
            resp = asyncio.get_event_loop().run_until_complete(asyncio.to_thread(_post))
        except Exception:
            resp = requests.post(url, data=data, timeout=30)
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
    msg = (
        "Traductor de audios y texto:\n"
        "• Nota de voz ES → EN (texto + audio)\n"
        "• Nota de voz EN → ES (texto + audio)\n"
        "• Texto ES → audio EN\n"
        "• Texto EN → audio ES\n"
        "\nComandos:\n/health (estado)"
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
    dst = "en" if src == "es" else ("es" if src == "en" else "es")
    translated = translate_smart(text_in, target=dst, source_lang=src if src in ("es","en") else None)

    human_src = "Español" if src == "es" else "Inglés" if src == "en" else "desconocido"
    human_dst = "Inglés" if dst == "en" else "Español"
    reply = [
        f"Idioma detectado (texto): {human_src}",
        "",
        "Original:",
        text_in,
        "",
        f"Traducción ({human_dst}):",
        translated if translated else "(no disponible)"
    ]
    await update.message.reply_text("\n".join(reply))

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

# ---------- Pipeline común para VOICE/AUDIO/DOCUMENT ----------
async def _process_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file, tmp_suffix: str):
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    # 1) Guardar entrada
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix)
    await tg_file.download_to_drive(custom_path=tf_in.name)

    # 2) Convertir a WAV 16k mono
    wav_path = tf_in.name + ".wav"
    ok = ffmpeg_to_wav_mono16k(tf_in.name, wav_path)
    if not ok:
        await update.message.reply_text("No pude convertir el audio. Revisa FFmpeg (agrega apt.txt con 'ffmpeg').")
        return

    # 3) Transcribir
    text_best, src_hint, text_es, text_en = vosk_transcribe_both(wav_path)
    if not text_best:
        await update.message.reply_text("No pude transcribir el audio.")
        return

    # 4) Decidir direcciones
    src = normalize_lang(src_hint)
    if src == "unknown":
        sguess = guess_lang_by_stops(text_best)
        src = sguess if sguess != "unknown" else detect_lang(text_best)
    dst = "en" if src == "es" else ("es" if src == "en" else "es")

    # 5) Traducir
    translated = translate_smart(text_best, target=dst, source_lang=src if src in ("es","en") else None)

    # fallback si detectó mal
    if src == "en":
        sim = jaccard_similarity(text_best, translated)
        if is_spanishish(text_best) or sim >= 0.75:
            src, dst = "es", "en"
            translated = translate_smart(text_best, target=dst, source_lang="es")

    # 6) Respuesta textual
    human_src = "Español" if src == "es" else "Inglés" if src == "en" else "desconocido"
    human_dst = "Inglés" if dst == "en" else "Español"
    reply = [
        f"Idioma detectado: {human_src}",
        "",
        "Transcripción:",
        text_best if text_best else "(vacío)",
        "",
        f"Traducción ({human_dst}):",
        translated if translated else "(no disponible)"
    ]
    await update.message.reply_text("\n".join(reply))

    # 7) Audio de la traducción (como documento, sin autoplay)
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

# ---- Nota de voz (VOICE) ----
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    if not voice:
        return
    tg_file = await voice.get_file()
    await _process_audio_file(update, context, tg_file, ".ogg")

# ---- Audio normal (AUDIO) ----
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    audio = update.message.audio
    if not audio:
        return
    tg_file = await audio.get_file()
    suffix = os.path.splitext(audio.file_name or "audio.mp3")[1] or ".mp3"
    await _process_audio_file(update, context, tg_file, suffix)

# ---- Documento con audio (DOCUMENT) ----
async def handle_document_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    name = (doc.file_name or "").lower()
    is_audio_doc = (doc.mime_type or "").startswith("audio/") or any(
        name.endswith(ext) for ext in (".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus")
    )
    if not is_audio_doc:
        return
    tg_file = await doc.get_file()
    suffix = os.path.splitext(doc.file_name or "file.mp3")[1] or ".mp3"
    await _process_audio_file(update, context, tg_file, suffix)

def build_app():
    bot_token = getenv_stripped("BOT_TOKEN", "")
    if not bot_token:
        raise RuntimeError("Falta BOT_TOKEN en variables de entorno (o vacío).")
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))

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
