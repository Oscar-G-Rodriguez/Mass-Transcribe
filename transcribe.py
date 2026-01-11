# transcribe_videos.py
# Batch-transcribe a folder of videos to TXT/SRT/JSON using local Whisper.
#
# Requirements:
#   - ffmpeg installed and available on PATH
#   - ONE backend installed (recommended: faster-whisper; fallback: openai-whisper)
#
# Install (pick one):
#   pip install faster-whisper
#   pip install openai-whisper
#
# Notes about internet:
#   - Transcription itself is local/offline.
#   - The first run for a model may download weights (needs internet once).
#   - After the model is cached, you can run offline.

from pathlib import Path
import json
import subprocess
import tempfile
import shutil

# =========================
# CONFIG (edit these)
# =========================

# INPUT_DIR:
#   - Possible types: str, Path
#   - Examples:
#       INPUT_DIR = r"C:\Users\you\Videos"
#       INPUT_DIR = "/home/you/videos"
#       INPUT_DIR = Path("./videos")
INPUT_DIR = r"./videos"

# OUTPUT_DIR:
#   - Possible types: str, Path
#   - Where transcript files will be written
OUTPUT_DIR = r"./transcripts"

# OUTPUT_FORMATS:
#   - Possible values: "txt", "srt", "json"
#   - Possible types: list[str]
#   - Examples:
#       ["txt"]
#       ["srt"]
#       ["txt", "srt", "json"]
OUTPUT_FORMATS = ["txt", "srt", "json"]

# RECURSIVE:
#   - Possible types: bool
#   - If True, scans subfolders inside INPUT_DIR
RECURSIVE = True

# SKIP_EXISTING:
#   - Possible types: bool
#   - If True, skips a video when ALL requested output files already exist
SKIP_EXISTING = True

# VIDEO_EXTENSIONS:
#   - Possible types: set[str], list[str]
#   - Common container types to scan for
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"
}

# BACKEND:
#   - Possible values:
#       "auto"           (try faster-whisper, else openai-whisper)
#       "faster-whisper"
#       "openai-whisper"
#   - Possible types: str
BACKEND = "auto"

# MODEL_NAME:
#   - Possible values (common Whisper sizes):
#       "tiny", "base", "small", "medium", "large-v3"
#   - Possible types: str
MODEL_NAME = "small"

# DEVICE:
#   - Possible values: "cpu", "cuda"
#   - Possible types: str
DEVICE = "cpu"

# COMPUTE_TYPE (faster-whisper only):
#   - Possible values: "int8", "int8_float16", "float16", "float32"
#   - Possible types: str
COMPUTE_TYPE = "int8"

# LANGUAGE:
#   - Possible values:
#       None (auto-detect), or a language code like "en", "es"
#   - Possible types: str | None
LANGUAGE = None

# BEAM_SIZE (faster-whisper only):
#   - Possible types: int
BEAM_SIZE = 5

# SAMPLE_RATE (audio extraction):
#   - Possible types: int
#   - Whisper commonly uses 16000 Hz mono
SAMPLE_RATE = 16000


# =========================
# Helpers
# =========================

def _require_ffmpeg():
    # Ensure ffmpeg is installed
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.\n"
            "Install ffmpeg and ensure 'ffmpeg' is available in your terminal."
        )

def _list_videos(root: Path):
    # Collect video files under root
    if not root.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {root}")

    if root.is_file():
        return [root] if root.suffix.lower() in VIDEO_EXTENSIONS else []

    if RECURSIVE:
        vids = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    else:
        vids = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]

    vids.sort()
    return vids

def _extract_audio_to_wav(video_path: Path, wav_path: Path):
    # Extract mono WAV audio using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",                      # overwrite output
        "-i", str(video_path),
        "-vn",                     # disable video
        "-ac", "1",                # mono
        "-ar", str(SAMPLE_RATE),   # sample rate
        "-c:a", "pcm_s16le",       # PCM 16-bit little-endian WAV
        str(wav_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}\n\n{p.stderr}")

def _srt_timestamp(seconds: float):
    # Convert seconds -> "HH:MM:SS,mmm"
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))

    hours = ms_total // 3_600_000
    ms_total -= hours * 3_600_000

    minutes = ms_total // 60_000
    ms_total -= minutes * 60_000

    secs = ms_total // 1000
    ms = ms_total - secs * 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def _write_txt(path: Path, segments):
    # Write plain transcript text
    path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join((s["text"] or "").strip() for s in segments if (s.get("text") or "").strip())
    path.write_text(text.strip() + "\n", encoding="utf-8")

def _write_srt(path: Path, segments):
    # Write SRT captions
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    idx = 1
    for s in segments:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        start = _srt_timestamp(float(s["start"]))
        end = _srt_timestamp(float(s["end"]))
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(t)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def _write_json(path: Path, segments, meta):
    # Write JSON with segments + metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "segments": segments}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# =========================
# Transcription backends
# =========================

def _pick_backend():
    # Decide backend based on BACKEND + installed packages
    if BACKEND in ("faster-whisper", "openai-whisper"):
        return BACKEND

    # BACKEND == "auto"
    try:
        import faster_whisper  # noqa: F401
        return "faster-whisper"
    except Exception:
        pass

    try:
        import whisper  # noqa: F401
        return "openai-whisper"
    except Exception:
        pass

    raise RuntimeError(
        "No backend available.\n"
        "Install one:\n"
        "  pip install faster-whisper\n"
        "or\n"
        "  pip install openai-whisper"
    )

def _transcribe_faster_whisper(wav_path: Path):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    seg_iter, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=True,
    )

    segments = []
    for s in seg_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    meta = {
        "backend": "faster-whisper",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }
    return segments, meta

def _transcribe_openai_whisper(wav_path: Path):
    import whisper

    model = whisper.load_model(MODEL_NAME)

    # fp16 is only safe on CUDA; on CPU use fp16=False
    fp16 = (DEVICE.lower() == "cuda")

    result = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        fp16=fp16,
        verbose=False,
    )

    segments = []
    for s in (result.get("segments") or []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip(),
        })

    meta = {
        "backend": "openai-whisper",
        "model": MODEL_NAME,
        "device": DEVICE,
        "language": result.get("language"),
    }
    return segments, meta


# =========================
# Main run
# =========================

def run():
    _require_ffmpeg()

    in_root = Path(INPUT_DIR).expanduser().resolve()
    out_root = Path(OUTPUT_DIR).expanduser().resolve()

    videos = _list_videos(in_root)
    if not videos:
        print("No video files found.")
        return

    backend = _pick_backend()
    print(f"Backend: {backend}")
    print(f"Found {len(videos)} video(s).")

    valid_formats = {"txt", "srt", "json"}
    formats = [f.lower().strip() for f in OUTPUT_FORMATS]
    for f in formats:
        if f not in valid_formats:
            raise ValueError(f"Invalid format '{f}'. Use only: {sorted(valid_formats)}")

    for i, video in enumerate(videos, start=1):
        # Preserve relative folder structure inside OUTPUT_DIR (when INPUT_DIR is a directory)
        if in_root.is_dir():
            rel = video.relative_to(in_root)
            base = out_root / rel
        else:
            base = out_root / video.name

        out_txt = base.with_suffix(".txt")
        out_srt = base.with_suffix(".srt")
        out_json = base.with_suffix(".json")

        # Skip if all requested outputs exist
        if SKIP_EXISTING:
            all_exist = True
            if "txt" in formats and not out_txt.exists():
                all_exist = False
            if "srt" in formats and not out_srt.exists():
                all_exist = False
            if "json" in formats and not out_json.exists():
                all_exist = False
            if all_exist:
                print(f"[{i}/{len(videos)}] Skipping (exists): {video}")
                continue

        print(f"[{i}/{len(videos)}] Transcribing: {video}")

        try:
            with tempfile.TemporaryDirectory(prefix="transcribe_") as td:
                td_path = Path(td)
                wav_path = td_path / f"{video.stem}.wav"

                _extract_audio_to_wav(video, wav_path)

                if backend == "faster-whisper":
                    segments, meta = _transcribe_faster_whisper(wav_path)
                else:
                    segments, meta = _transcribe_openai_whisper(wav_path)

            meta["source_video"] = str(video)

            # Write outputs requested
            if "txt" in formats:
                _write_txt(out_txt, segments)
            if "srt" in formats:
                _write_srt(out_srt, segments)
            if "json" in formats:
                _write_json(out_json, segments, meta)

        except Exception as e:
            print(f"[ERROR] {video}\n{e}")

    print("Done.")



run()
