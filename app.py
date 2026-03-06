import os
import re
import tempfile
import numpy as np
import librosa
import yt_dlp
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from hmmlearn import hmm

app = Flask(__name__)
CORS(app)

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# ── Chord templates: major, minor, 7, m7, dim, aug, sus2, sus4 ──
def _build_chord_templates():
    # intervals relative to root (in semitones)
    CHORD_TYPES = {
        '':     [0, 4, 7],       # major
        'm':    [0, 3, 7],       # minor
        '7':    [0, 4, 7, 10],   # dominant 7th
        'm7':   [0, 3, 7, 10],   # minor 7th
        '7M':   [0, 4, 7, 11],   # major 7th
        'dim':  [0, 3, 6],       # diminished
        'aug':  [0, 4, 8],       # augmented
        'sus2': [0, 2, 7],       # suspended 2nd
        'sus4': [0, 5, 7],       # suspended 4th
    }
    templates = {}  # name -> normalized 12-d vector
    for i, root in enumerate(NOTE_NAMES):
        for suffix, intervals in CHORD_TYPES.items():
            vec = np.zeros(12)
            for iv in intervals:
                vec[(i + iv) % 12] = 1.0
            templates[f"{root}{suffix}"] = vec / np.linalg.norm(vec)
    return templates

CHORD_TEMPLATES = _build_chord_templates()
CHORD_NAMES = list(CHORD_TEMPLATES.keys())      # deterministic order
CHORD_MATRIX = np.array([CHORD_TEMPLATES[n] for n in CHORD_NAMES])  # (N_chords, 12)
N_CHORDS = len(CHORD_NAMES)

# Simplicity prior: triads are far more common than complex chords.
# Complex chords must score significantly higher to win over a simple triad.
CHORD_PRIOR = np.array([
    {
        '': 1.0,        # major triad — no penalty
        'm': 1.0,       # minor triad — no penalty
        '7': 0.82,      # dominant 7th — slight penalty
        'm7': 0.80,     # minor 7th
        '7M': 0.78,     # major 7th
        'sus2': 0.75,   # suspended chords are rarer
        'sus4': 0.75,
        'dim': 0.72,    # diminished
        'aug': 0.72,    # augmented
    }[suffix]
    for root in NOTE_NAMES
    for suffix in ['', 'm', '7', 'm7', '7M', 'dim', 'aug', 'sus2', 'sus4']
])


def _extract_root(chord_name: str) -> str:
    """Extract root note from chord name like 'C#m7' -> 'C#'."""
    if len(chord_name) >= 2 and chord_name[1] == '#':
        return chord_name[:2]
    return chord_name[0]


def _build_hmm_model() -> hmm.CategoricalHMM:
    """
    Build a Hidden Markov Model for chord sequence decoding.
    - Transition matrix: high self-transition (chords sustain), small uniform
      probability for switching, with bonus for musically common transitions.
    - Emission is handled externally (we use the Viterbi path on score indices).
    """
    SELF_PROB = 0.70  # probability of staying on same chord
    model = hmm.CategoricalHMM(n_components=N_CHORDS, n_iter=0, init_params="")

    # Uniform start
    model.startprob_ = np.ones(N_CHORDS) / N_CHORDS

    # Transition matrix with musical priors
    trans = np.full((N_CHORDS, N_CHORDS), (1.0 - SELF_PROB) / (N_CHORDS - 1))
    np.fill_diagonal(trans, SELF_PROB)

    # Boost common harmonic movements (4th, 5th, relative minor/major)
    COMMON_INTERVALS = [5, 7, 3, 4, 9, 8, 2]  # P4, P5, m3, M3, M6, m6, M2
    for i, name_i in enumerate(CHORD_NAMES):
        root_i_str = _extract_root(name_i)
        if root_i_str not in NOTE_NAMES:
            continue
        root_i = NOTE_NAMES.index(root_i_str)
        for j, name_j in enumerate(CHORD_NAMES):
            if i == j:
                continue
            root_j_str = _extract_root(name_j)
            if root_j_str not in NOTE_NAMES:
                continue
            root_j = NOTE_NAMES.index(root_j_str)
            interval = (root_j - root_i) % 12
            if interval in COMMON_INTERVALS:
                trans[i, j] *= 2.5  # boost common transitions

    # Re-normalize rows
    trans /= trans.sum(axis=1, keepdims=True)
    model.transmat_ = trans

    # Emission probs are set dynamically per song (dummy here)
    model.emissionprob_ = np.eye(N_CHORDS)
    model.n_features = N_CHORDS
    return model

_HMM_MODEL = _build_hmm_model()


def detect_chords(y, sr, hop_length=512) -> list[dict]:
    """
    Detect chords using NNLS chroma + beat-synchronous segmentation + HMM decoding.
    """
    # ── 1. Feature extraction ──
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # NNLS chroma: fits harmonics precisely, much cleaner than raw CQT
    chroma_nnls = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=hop_length,
        bins_per_octave=36, n_octaves=6
    )
    chroma_cens = librosa.feature.chroma_cens(
        y=y_harmonic, sr=sr, hop_length=hop_length
    )
    min_frames = min(chroma_nnls.shape[1], chroma_cens.shape[1])
    chroma = chroma_nnls[:, :min_frames] * 0.65 + chroma_cens[:, :min_frames] * 0.35

    # Light median smoothing (size=3: removes only single-frame spikes)
    from scipy.ndimage import median_filter
    chroma = median_filter(chroma, size=(1, 3))

    # ── 2. Beat-synchronous + sub-beat segmentation ──
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    # Create boundaries at each beat AND at each half-beat
    all_boundaries = [0]
    for i in range(len(beat_frames)):
        if i > 0:
            mid = (beat_frames[i - 1] + beat_frames[i]) // 2
            all_boundaries.append(int(mid))
        all_boundaries.append(int(beat_frames[i]))
    all_boundaries.append(chroma.shape[1])
    all_boundaries = sorted(set(all_boundaries))

    # ── 3. Score each segment against all chord templates ──
    segment_times = []
    score_matrix = []  # (n_segments, N_CHORDS)

    for i in range(len(all_boundaries) - 1):
        sf = all_boundaries[i]
        ef = all_boundaries[i + 1]
        if ef - sf < 2:
            continue

        seg = chroma[:, sf:ef].mean(axis=1)
        norm = np.linalg.norm(seg)
        if norm < 1e-6:
            continue
        seg = seg / norm

        # Dot product against all templates at once (vectorized)
        scores = (CHORD_MATRIX @ seg) * CHORD_PRIOR  # apply simplicity bias
        score_matrix.append(scores)
        segment_times.append(round(sf * hop_length / sr, 2))

    if not score_matrix:
        return []

    score_matrix = np.array(score_matrix)  # (n_segments, N_CHORDS)

    # ── 4. HMM Viterbi decoding for optimal chord sequence ──
    # Convert scores to pseudo-probabilities via softmax
    temperature = 0.15  # lower = more peaky = more confident
    exp_scores = np.exp(score_matrix / temperature)
    emission_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Build per-song emission matrix and decode
    model = _HMM_MODEL
    # Use log-likelihood Viterbi via hmmlearn's decode
    # We need to pass observations as indices, but we have continuous emissions.
    # Workaround: use GaussianHMM-style or manual Viterbi.
    best_path = _viterbi_decode(model.startprob_, model.transmat_, emission_probs)

    # ── 5. Convert to output format, emit only on changes ──
    chords = []
    prev_chord = None
    for idx, seg_idx in enumerate(best_path):
        chord = CHORD_NAMES[seg_idx]
        if chord != prev_chord:
            chords.append({"time": segment_times[idx], "chord": chord})
            prev_chord = chord

    return chords


def _viterbi_decode(startprob, transmat, emission_probs):
    """
    Standard Viterbi algorithm for HMM decoding with per-frame emission probs.
    emission_probs: (T, N) matrix of P(observation_t | state_n).
    Returns: list of state indices (length T).
    """
    T, N = emission_probs.shape
    log_start = np.log(startprob + 1e-12)
    log_trans = np.log(transmat + 1e-12)
    log_emiss = np.log(emission_probs + 1e-12)

    # Viterbi tables
    V = np.zeros((T, N))
    ptr = np.zeros((T, N), dtype=int)

    V[0] = log_start + log_emiss[0]

    for t in range(1, T):
        for j in range(N):
            candidates = V[t - 1] + log_trans[:, j]
            ptr[t, j] = np.argmax(candidates)
            V[t, j] = candidates[ptr[t, j]] + log_emiss[t, j]

    # Backtrace
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return path

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
    chords = detect_chords(y, sr)

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