import os
import re
import tempfile
import numpy as np
import librosa
import yt_dlp
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates – major and minor triads for all 12 notes
def _build_chord_templates():
    templates = {}
    for i, name in enumerate(NOTE_NAMES):
        # Major triad: root + major third (4) + perfect fifth (7)
        major = np.zeros(12)
        major[i] = 1.0
        major[(i + 4) % 12] = 1.0
        major[(i + 7) % 12] = 1.0
        templates[name] = major / np.linalg.norm(major)

        # Minor triad: root + minor third (3) + perfect fifth (7)
        minor = np.zeros(12)
        minor[i] = 1.0
        minor[(i + 3) % 12] = 1.0
        minor[(i + 7) % 12] = 1.0
        templates[f"{name}m"] = minor / np.linalg.norm(minor)

    return templates

CHORD_TEMPLATES = _build_chord_templates()

def detect_chords(y, sr, hop_length=512) -> list[dict]:
    """
    Detect chords using beat-synchronous analysis for natural timing,
    with sub-beat resolution for fast passages.
    """
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # 1. Two chroma types combined for robustness
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr, hop_length=hop_length)

    # Resize to match (chroma_cens can differ by a frame)
    min_frames = min(chroma_cqt.shape[1], chroma_cens.shape[1])
    chroma = chroma_cqt[:, :min_frames] * 0.6 + chroma_cens[:, :min_frames] * 0.4

    # 2. Light smoothing — small median to remove only spikes, not real changes
    from scipy.ndimage import median_filter
    chroma = median_filter(chroma, size=(1, 5))

    # 3. Detect beats for beat-synchronous boundaries
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    # 4. Add sub-beat boundaries: split each beat interval in half
    #    This allows detecting chord changes that happen on the "and" (off-beat)
    all_boundaries = [0]
    for i in range(len(beat_frames)):
        if i > 0:
            mid = (beat_frames[i - 1] + beat_frames[i]) // 2
            all_boundaries.append(int(mid))
        all_boundaries.append(int(beat_frames[i]))
    all_boundaries.append(chroma.shape[1])
    all_boundaries = sorted(set(all_boundaries))

    # 5. Analyze each segment
    raw = []  # list of (time, chord, score)
    for i in range(len(all_boundaries) - 1):
        start_f = all_boundaries[i]
        end_f = all_boundaries[i + 1]
        if end_f - start_f < 2:
            continue

        seg_chroma = chroma[:, start_f:end_f].mean(axis=1)
        norm = np.linalg.norm(seg_chroma)
        if norm < 1e-6:
            continue
        seg_chroma = seg_chroma / norm

        best_chord = None
        best_score = -1
        for chord_name, template in CHORD_TEMPLATES.items():
            score = np.dot(seg_chroma, template)
            if score > best_score:
                best_score = score
                best_chord = chord_name

        t = round(start_f * hop_length / sr, 2)
        raw.append((t, best_chord, best_score))

    if not raw:
        return []

    # 6. Confidence-based hysteresis filter:
    #    Only switch chord if the NEW chord scores significantly better,
    #    OR if the current chord scores poorly at this segment.
    HYSTERESIS = 0.03  # minimum score advantage to trigger a switch
    chords = []
    current_chord = raw[0][1]
    current_score = raw[0][2]
    chords.append({"time": raw[0][0], "chord": current_chord})

    for t, chord, score in raw[1:]:
        if chord != current_chord:
            # Recompute current chord's score at this segment to compare fairly
            # Switch if: new chord is clearly better OR current chord fits poorly
            if score - current_score > HYSTERESIS or score > 0.85:
                chords.append({"time": t, "chord": chord})
                current_chord = chord
                current_score = score
        else:
            current_score = score  # update running score

    return chords

def extract_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def detect_key(y, sr) -> tuple[str, float]:
    # 1. Separar harmônicos da percussão
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # 2. Usar múltiplos cromagramas e combinar
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    # Média ponderada dos 3 cromagramas
    chroma_mean = (
        chroma_cqt.mean(axis=1) * 0.5 +
        chroma_stft.mean(axis=1) * 0.3 +
        chroma_cens.mean(axis=1) * 0.2
    )

    # 3. Normalizar antes de correlacionar
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)

    major_scores = []
    minor_scores = []
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        major_scores.append(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        minor_scores.append(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

    best_major = max(range(12), key=lambda i: major_scores[i])
    best_minor = max(range(12), key=lambda i: minor_scores[i])

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{NOTE_NAMES[best_major]} Major", float(major_scores[best_major])
    else:
        return f"{NOTE_NAMES[best_minor]} Minor", float(minor_scores[best_minor])


def detect_bpm(y, sr) -> tuple[float, list[float]]:
    # 1. Separar percussão para melhor detecção de beats
    y_percussive = librosa.effects.percussive(y, margin=4)

    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    # 2. Múltiplos estimadores
    tempo_default = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    tempo_plp = librosa.beat.tempo(onset_envelope=pulse, sr=sr)[0]

    # 3. Usar tempograma para estimativa mais robusta
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo_tg = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                   prior=None, aggregate=np.median)[0]

    # 4. Mediana ao invés de média (mais robusta a outliers)
    candidates = [tempo_default, tempo_plp, tempo_tg]
    final_tempo = round(float(np.median(candidates)), 1)

    # 5. Corrigir subharmônicos: se BPM < 60, provavelmente é o dobro
    if final_tempo < 60:
        final_tempo *= 2
    elif final_tempo > 200:
        final_tempo /= 2

    _, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

    return round(final_tempo, 1), beat_times[:20]

def analyze_audio(audio_path: str) -> dict:
    """Run full audio analysis on a file."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=120)

    bpm, beats = detect_bpm(y, sr)
    key, confidence = detect_key(y, sr)
    chords = detect_chords(y, sr)  # ← ADICIONAR

    # Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms_energy = float(np.mean(librosa.feature.rms(y=y)))

    # Danceability proxy: beat strength consistency
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_consistency = float(1 - np.std(onset_env) / (np.mean(onset_env) + 1e-6))
    danceability = round(min(max(beat_consistency * 100, 0), 100), 1)

    # Energy level
    energy = round(min(rms_energy * 2000, 100), 1)

    return {
        "bpm": bpm,
        "key": key,
        "key_confidence": round(confidence * 100, 1),
        "beats": beats,
        "chords": chords,  
        "danceability": danceability,
        "energy": energy,
        "spectral_centroid": round(spectral_centroid, 1),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL não fornecida."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "URL do YouTube inválida."}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            return jsonify({"error": f"Erro ao baixar áudio: {str(e)}"}), 500

        wav_path = os.path.join(tmpdir, "audio.wav")
        if not os.path.exists(wav_path):
            # Find whatever audio file was downloaded
            files = os.listdir(tmpdir)
            if not files:
                return jsonify({"error": "Falha ao extrair áudio."}), 500
            wav_path = os.path.join(tmpdir, files[0])

        try:
            result = analyze_audio(wav_path)
            result["video_id"] = video_id
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Erro na análise: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)